"""Run end-to-end tracker profiles over project videos and export comparison data."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import torch
import ultralytics

from experiments.tracker_benchmark.metrics import (
    TrackPoint,
    find_fragment_candidates,
    split_tracks,
    summarize_video,
    track_row,
)
from experiments.tracker_benchmark.profiles import PROFILES, select_profiles
from experiments.tracker_benchmark.report import write_report
from src.core.detector import YOLODetector
from src.core.tracker import ByteTracker
from src.utils.visualizer import Visualizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Ultralytics tracker profiles on project videos without modifying the main DB."
    )
    parser.add_argument("--videos", default="data/videos", help="Video file or directory")
    parser.add_argument("--model", default="models/detection/best.pt")
    parser.add_argument("--output", default="data/tracker_benchmark")
    parser.add_argument("--run-name", default=None, help="Output subdirectory; timestamp if omitted")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Profile names. Omit to run the six representative algorithms.",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--track-conf", type=float, default=0.1)
    parser.add_argument("--person-conf", type=float, default=0.5)
    parser.add_argument("--bag-conf", type=float, default=0.25)
    parser.add_argument("--default-conf", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit per video")
    parser.add_argument("--max-fragment-gap", type=int, default=15)
    parser.add_argument("--save-videos", action="store_true", help="Write annotated comparison videos")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def discover_videos(value: str) -> list[Path]:
    source = resolve_repo_path(value)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Video source does not exist: {source}")
    videos = sorted(path for path in source.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No supported video files found under: {source}")
    return videos


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_profile_video(
    profile_name: str,
    tracker: ByteTracker,
    video_path: Path,
    run_dir: Path,
    max_frames: int | None,
    max_fragment_gap: int,
    save_video: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    metadata_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save_video:
        video_dir = run_dir / "videos" / profile_name
        video_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_dir / f"{video_path.stem}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

    tracker.reset()
    points: list[TrackPoint] = []
    frame_idx = 0
    tracker_seconds = 0.0
    first_frame_seconds = 0.0
    wall_start = perf_counter()
    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            update_start = perf_counter()
            tracked_objects = tracker.update(frame, frame_idx, timestamp_ms)
            update_seconds = perf_counter() - update_start
            tracker_seconds += update_seconds
            if frame_idx == 0:
                first_frame_seconds = update_seconds

            for obj in tracked_objects:
                points.append(TrackPoint(
                    frame_idx=obj.frame_idx,
                    timestamp_ms=obj.timestamp_ms,
                    track_id=obj.track_id,
                    label=obj.label,
                    class_id=obj.class_id,
                    confidence=obj.confidence,
                    bbox=obj.bbox,
                ))
            if writer is not None:
                writer.write(Visualizer.draw_tracked(frame.copy(), tracked_objects))
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"    {video_path.name}: {frame_idx} frames", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    wall_seconds = perf_counter() - wall_start

    tracks = split_tracks(points)
    fragments = find_fragment_candidates(
        profile_name,
        video_path.name,
        tracks,
        max_gap=max_fragment_gap,
    )
    summary = summarize_video(
        tracker_name=profile_name,
        video_name=video_path.name,
        points=points,
        decoded_frames=frame_idx,
        metadata_frames=metadata_frames,
        source_fps=source_fps,
        tracker_seconds=tracker_seconds,
        first_frame_seconds=first_frame_seconds,
        wall_seconds=wall_seconds,
        fragment_count=len(fragments),
    )
    track_rows = [
        track_row(profile_name, video_path.name, track_id, track_points)
        for track_id, track_points in tracks.items()
    ]
    observation_rows = [{
        "tracker": profile_name,
        "video": video_path.name,
        "frame_idx": point.frame_idx,
        "timestamp_ms": point.timestamp_ms,
        "track_id": point.track_id,
        "label": point.label,
        "class_id": point.class_id,
        "confidence": point.confidence,
        "x1": point.bbox[0],
        "y1": point.bbox[1],
        "x2": point.bbox[2],
        "y2": point.bbox[3],
    } for point in points]
    return summary, track_rows, fragments, observation_rows


def main() -> int:
    args = parse_args()
    if args.list_profiles:
        for name, profile in PROFILES.items():
            print(f"{name:24} {profile.algorithm:14} config={profile.config} reid={profile.reid}")
        return 0

    profiles = select_profiles(args.profiles)
    videos = discover_videos(args.videos)
    model_path = resolve_repo_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    output_root = resolve_repo_path(args.output)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    thresholds = {
        "person": args.person_conf,
        "bag": args.bag_conf,
        "default": args.default_conf,
    }
    summary_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    fragment_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    print(f"Videos: {len(videos)} | Profiles: {len(profiles)} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Output: {run_dir}")

    for profile_index, profile in enumerate(profiles, start=1):
        config = profile.resolved_config(REPO_ROOT)
        print(f"[{profile_index}/{len(profiles)}] {profile.name} ({config})", flush=True)
        try:
            detector = YOLODetector(str(model_path), thresholds)
            tracker = ByteTracker(detector, config, args.track_conf)
        except Exception as exc:
            error_rows.append({"tracker": profile.name, "video": "", "error": repr(exc)})
            print(f"  ERROR initializing: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
            continue

        for video_index, video_path in enumerate(videos, start=1):
            print(f"  [{video_index}/{len(videos)}] {video_path.name}", flush=True)
            try:
                summary, tracks, fragments, observations = run_profile_video(
                    profile.name,
                    tracker,
                    video_path,
                    run_dir,
                    args.max_frames,
                    args.max_fragment_gap,
                    args.save_videos,
                )
                summary_rows.append(summary)
                track_rows.extend(tracks)
                fragment_rows.extend(fragments)
                observation_rows.extend(observations)
                print(
                    f"    done: {summary['decoded_frames']} frames, "
                    f"{float(summary['steady_tracker_fps']):.2f} FPS, "
                    f"{summary['unique_tracks']} tracks, "
                    f"{summary['fragment_candidates']} fragment candidates",
                    flush=True,
                )
            except Exception as exc:
                error_rows.append({"tracker": profile.name, "video": video_path.name, "error": repr(exc)})
                print(f"    ERROR: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise

    write_csv(run_dir / "summary.csv", summary_rows)
    write_csv(run_dir / "tracks.csv", track_rows)
    write_csv(run_dir / "fragment_candidates.csv", fragment_rows)
    write_csv(run_dir / "observations.csv", observation_rows)
    write_csv(run_dir / "errors.csv", error_rows)
    write_report(run_dir / "report.md", summary_rows, track_rows, profiles, str(model_path))

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model": str(model_path),
        "thresholds": thresholds,
        "track_conf": args.track_conf,
        "profiles": [profile.__dict__ | {"resolved_config": profile.resolved_config(REPO_ROOT)} for profile in profiles],
        "videos": [str(video) for video in videos],
        "max_frames": args.max_frames,
        "proxy_metric_warning": "Fragment candidates and continuity metrics are not HOTA/IDF1 without ground truth.",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished. Report: {run_dir / 'report.md'}")
    return 1 if error_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
