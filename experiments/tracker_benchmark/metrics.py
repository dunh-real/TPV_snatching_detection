"""Ground-truth-free continuity metrics for comparing tracker outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp, hypot, sqrt
from statistics import mean, median
from typing import Iterable


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackPoint:
    """One thresholded tracker observation."""

    frame_idx: int
    timestamp_ms: float
    track_id: int
    label: str
    class_id: int
    confidence: float
    bbox: BBox

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


def intersection_over_union(a: BBox, b: BBox) -> float:
    """Compute IoU for two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def split_tracks(points: Iterable[TrackPoint]) -> dict[int, list[TrackPoint]]:
    """Group observations by tracker ID and sort them by frame."""
    grouped: dict[int, list[TrackPoint]] = defaultdict(list)
    for point in points:
        grouped[point.track_id].append(point)
    return {
        track_id: sorted(track_points, key=lambda item: item.frame_idx)
        for track_id, track_points in grouped.items()
    }


def dominant_label(points: list[TrackPoint]) -> str:
    """Return the most frequent label in a track."""
    return Counter(point.label for point in points).most_common(1)[0][0]


def count_label_switches(points: list[TrackPoint]) -> int:
    """Count consecutive label changes within one tracker ID."""
    return sum(a.label != b.label for a, b in zip(points, points[1:]))


def track_row(
    tracker_name: str,
    video_name: str,
    track_id: int,
    points: list[TrackPoint],
) -> dict[str, object]:
    """Summarize one track for CSV output."""
    first, last = points[0], points[-1]
    frames = sorted({point.frame_idx for point in points})
    lifespan = last.frame_idx - first.frame_idx + 1
    return {
        "tracker": tracker_name,
        "video": video_name,
        "track_id": track_id,
        "label": dominant_label(points),
        "start_frame": first.frame_idx,
        "end_frame": last.frame_idx,
        "observed_frames": len(frames),
        "lifespan_frames": lifespan,
        "internal_gap_frames": max(0, lifespan - len(frames)),
        "observation_ratio": len(frames) / lifespan if lifespan else 0.0,
        "mean_confidence": mean(point.confidence for point in points),
        "min_confidence": min(point.confidence for point in points),
        "label_switches": count_label_switches(points),
    }


def _estimated_velocity(points: list[TrackPoint], history: int = 5) -> tuple[float, float]:
    """Estimate pixels/frame velocity from the recent track history."""
    if len(points) < 2:
        return (0.0, 0.0)
    first = points[max(0, len(points) - history)]
    last = points[-1]
    delta_frames = last.frame_idx - first.frame_idx
    if delta_frames <= 0:
        return (0.0, 0.0)
    first_x, first_y = first.center
    last_x, last_y = last.center
    return ((last_x - first_x) / delta_frames, (last_y - first_y) / delta_frames)


def find_fragment_candidates(
    tracker_name: str,
    video_name: str,
    tracks: dict[int, list[TrackPoint]],
    max_gap: int = 15,
    max_center_distance: float = 1.25,
    min_size_ratio: float = 0.2,
) -> list[dict[str, object]]:
    """Find likely tracklet continuations with different IDs.

    This is a heuristic, not a true ID-switch metric. Pairs are greedily selected so
    one ending track and one starting track appear in at most one candidate.
    """
    candidates: list[dict[str, object]] = []
    track_items = list(tracks.items())

    for old_id, old_points in track_items:
        old_last = old_points[-1]
        old_label = dominant_label(old_points)
        velocity_x, velocity_y = _estimated_velocity(old_points)

        for new_id, new_points in track_items:
            if old_id == new_id or dominant_label(new_points) != old_label:
                continue
            new_first = new_points[0]
            gap = new_first.frame_idx - old_last.frame_idx
            if not 1 <= gap <= max_gap:
                continue

            old_x, old_y = old_last.center
            new_x, new_y = new_first.center
            predicted_x = old_x + velocity_x * gap
            predicted_y = old_y + velocity_y * gap
            scale = max(20.0, (old_last.height + new_first.height) / 2.0)
            normalized_distance = hypot(new_x - predicted_x, new_y - predicted_y) / scale
            size_ratio = (
                min(old_last.area, new_first.area) / max(old_last.area, new_first.area)
                if max(old_last.area, new_first.area) > 0.0
                else 0.0
            )
            iou = intersection_over_union(old_last.bbox, new_first.bbox)

            if not (
                (normalized_distance <= max_center_distance and size_ratio >= min_size_ratio)
                or iou >= 0.1
            ):
                continue

            score = (
                0.45 * exp(-normalized_distance)
                + 0.25 * iou
                + 0.20 * sqrt(size_ratio)
                + 0.10 * exp(-gap / max_gap)
            )
            candidates.append({
                "tracker": tracker_name,
                "video": video_name,
                "label": old_label,
                "old_track_id": old_id,
                "new_track_id": new_id,
                "old_end_frame": old_last.frame_idx,
                "new_start_frame": new_first.frame_idx,
                "gap_frames": gap,
                "predicted_center_distance": normalized_distance,
                "bbox_iou": iou,
                "size_ratio": size_ratio,
                "heuristic_score": score,
            })

    selected: list[dict[str, object]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: float(item["heuristic_score"]), reverse=True):
        old_id = int(candidate["old_track_id"])
        new_id = int(candidate["new_track_id"])
        if old_id in used_old or new_id in used_new:
            continue
        used_old.add(old_id)
        used_new.add(new_id)
        selected.append(candidate)
    return selected


def summarize_video(
    tracker_name: str,
    video_name: str,
    points: list[TrackPoint],
    decoded_frames: int,
    metadata_frames: int,
    source_fps: float,
    tracker_seconds: float,
    first_frame_seconds: float,
    wall_seconds: float,
    fragment_count: int,
) -> dict[str, object]:
    """Calculate one tracker/video summary row."""
    tracks = split_tracks(points)
    lengths = [len({point.frame_idx for point in track}) for track in tracks.values()]
    frames_any = {point.frame_idx for point in points}
    frames_person = {point.frame_idx for point in points if point.label == "person"}
    frames_bag = {point.frame_idx for point in points if point.label == "bag"}
    short_tracks = sum(length <= 5 for length in lengths)
    singleton_tracks = sum(length == 1 for length in lengths)
    steady_seconds = max(0.0, tracker_seconds - first_frame_seconds)
    steady_frames = max(0, decoded_frames - 1)

    return {
        "tracker": tracker_name,
        "video": video_name,
        "metadata_frames": metadata_frames,
        "decoded_frames": decoded_frames,
        "decode_ratio": decoded_frames / metadata_frames if metadata_frames else 0.0,
        "source_fps": source_fps,
        "source_duration_s": decoded_frames / source_fps if source_fps else 0.0,
        "tracker_seconds": tracker_seconds,
        "first_frame_seconds": first_frame_seconds,
        "wall_seconds": wall_seconds,
        "tracker_fps": decoded_frames / tracker_seconds if tracker_seconds else 0.0,
        "steady_tracker_fps": steady_frames / steady_seconds if steady_seconds else 0.0,
        "end_to_end_fps": decoded_frames / wall_seconds if wall_seconds else 0.0,
        "observations": len(points),
        "unique_tracks": len(tracks),
        "person_tracks": sum(dominant_label(track) == "person" for track in tracks.values()),
        "bag_tracks": sum(dominant_label(track) == "bag" for track in tracks.values()),
        "mean_track_length": mean(lengths) if lengths else 0.0,
        "median_track_length": median(lengths) if lengths else 0.0,
        "short_tracks_le_5": short_tracks,
        "short_track_ratio": short_tracks / len(lengths) if lengths else 0.0,
        "singleton_tracks": singleton_tracks,
        "internal_gap_frames": sum(
            max(0, track[-1].frame_idx - track[0].frame_idx + 1 - len({p.frame_idx for p in track}))
            for track in tracks.values()
        ),
        "label_switches": sum(count_label_switches(track) for track in tracks.values()),
        "fragment_candidates": fragment_count,
        "frames_with_any_track": len(frames_any),
        "frames_with_person": len(frames_person),
        "frames_with_bag": len(frames_bag),
        "coverage_any": len(frames_any) / decoded_frames if decoded_frames else 0.0,
        "coverage_person": len(frames_person) / decoded_frames if decoded_frames else 0.0,
        "coverage_bag": len(frames_bag) / decoded_frames if decoded_frames else 0.0,
        "mean_confidence": mean(point.confidence for point in points) if points else 0.0,
    }
