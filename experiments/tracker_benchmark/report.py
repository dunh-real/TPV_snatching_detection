"""Generate a human-readable Markdown report from benchmark rows."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from experiments.tracker_benchmark.profiles import TrackerProfile


def _number(value: object, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _aggregate(
    summary_rows: list[dict[str, object]],
    track_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_tracker: dict[str, list[dict[str, object]]] = defaultdict(list)
    lengths_by_tracker: dict[str, list[int]] = defaultdict(list)
    for row in summary_rows:
        by_tracker[str(row["tracker"])].append(row)
    for row in track_rows:
        lengths_by_tracker[str(row["tracker"])].append(int(row["observed_frames"]))

    aggregate_rows: list[dict[str, object]] = []
    for tracker, rows in by_tracker.items():
        frames = sum(int(row["decoded_frames"]) for row in rows)
        tracker_seconds = sum(float(row["tracker_seconds"]) for row in rows)
        steady_seconds = sum(
            max(0.0, float(row["tracker_seconds"]) - float(row["first_frame_seconds"]))
            for row in rows
        )
        steady_frames = sum(max(0, int(row["decoded_frames"]) - 1) for row in rows)
        observations = sum(int(row["observations"]) for row in rows)
        tracks = sum(int(row["unique_tracks"]) for row in rows)
        short_tracks = sum(int(row["short_tracks_le_5"]) for row in rows)
        aggregate_rows.append({
            "tracker": tracker,
            "videos": len(rows),
            "frames": frames,
            "tracker_fps": frames / tracker_seconds if tracker_seconds else 0.0,
            "steady_tracker_fps": steady_frames / steady_seconds if steady_seconds else 0.0,
            "observations": observations,
            "tracks": tracks,
            "median_track_length": median(lengths_by_tracker[tracker]) if lengths_by_tracker[tracker] else 0.0,
            "short_track_ratio": short_tracks / tracks if tracks else 0.0,
            "fragment_candidates": sum(int(row["fragment_candidates"]) for row in rows),
            "fragment_rate_1000_obs": (
                1000.0 * sum(int(row["fragment_candidates"]) for row in rows) / observations
                if observations
                else 0.0
            ),
            "coverage_any": (
                sum(int(row["frames_with_any_track"]) for row in rows) / frames if frames else 0.0
            ),
            "coverage_person": (
                sum(int(row["frames_with_person"]) for row in rows) / frames if frames else 0.0
            ),
            "coverage_bag": (
                sum(int(row["frames_with_bag"]) for row in rows) / frames if frames else 0.0
            ),
            "label_switches": sum(int(row["label_switches"]) for row in rows),
        })
    return aggregate_rows


def write_report(
    destination: Path,
    summary_rows: list[dict[str, object]],
    track_rows: list[dict[str, object]],
    profiles: list[TrackerProfile],
    model_path: str,
) -> None:
    """Write benchmark tables, caveats, and qualitative tracker guidance."""
    aggregate_rows = _aggregate(summary_rows, track_rows)
    profile_by_name = {profile.name: profile for profile in profiles}
    max_coverage = max((float(row["coverage_any"]) for row in aggregate_rows), default=0.0)
    eligible = [
        row for row in aggregate_rows
        if float(row["coverage_any"]) >= 0.9 * max_coverage
    ]
    shortlist = sorted(
        eligible,
        key=lambda row: (
            float(row["short_track_ratio"]),
            float(row["fragment_rate_1000_obs"]),
            -float(row["median_track_length"]),
            -float(row["steady_tracker_fps"]),
        ),
    )[:3]

    lines = [
        "# Tracker benchmark report",
        "",
        f"Model: `{model_path}`",
        "",
        "> **Quan trọng:** chưa có ground-truth track ID nên `fragment_candidates`, track ngắn và coverage chỉ là proxy. "
        "Chúng không thay thế HOTA, IDF1 hoặc số ID switch thật. Tracker tạo ít ID có thể chỉ đang bỏ sót object.",
        "",
        "> Mỗi hàng là một **profile = thuật toán + cấu hình**. Hai profile project đã được tune, trong khi OC/DeepOC/"
        "FastTrack/TrackTrack ở cấu hình Ultralytics mặc định. Bảng này chọn profile chạy thực tế, không phải phép so sánh "
        "thuật toán thuần túy với hyperparameter hoàn toàn công bằng.",
        "",
        "## Tổng hợp toàn bộ video",
        "",
        "| Profile | Algorithm | ReID | Steady FPS | Tracks | Median length | Short ≤5 | Fragment/1k obs | Coverage person | Coverage bag |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate_rows, key=lambda item: str(item["tracker"])):
        profile = profile_by_name[str(row["tracker"])]
        lines.append(
            f"| {profile.name} | {profile.algorithm} | {'yes' if profile.reid else 'no'} | "
            f"{_number(row['steady_tracker_fps'])} | {row['tracks']} | "
            f"{_number(row['median_track_length'], 1)} | {_number(100 * float(row['short_track_ratio']), 1)}% | "
            f"{_number(row['fragment_rate_1000_obs'])} | {_number(100 * float(row['coverage_person']), 1)}% | "
            f"{_number(100 * float(row['coverage_bag']), 1)}% |"
        )

    lines.extend([
        "",
        "## Shortlist cần xem video/ground truth tiếp",
        "",
    ])
    if shortlist:
        for index, row in enumerate(shortlist, start=1):
            lines.append(
                f"{index}. `{row['tracker']}` — short tracks "
                f"{_number(100 * float(row['short_track_ratio']), 1)}%, "
                f"fragment proxy {_number(row['fragment_rate_1000_obs'])}/1k observations, "
                f"steady {_number(row['steady_tracker_fps'])} FPS."
            )
    else:
        lines.append("Không có profile hoàn thành benchmark.")

    lines.extend([
        "",
        "Shortlist chỉ loại các profile có coverage thấp hơn 90% coverage tốt nhất rồi ưu tiên ít track ngắn/fragment proxy. "
        "Hãy kiểm tra trực quan và gán ground truth trước khi chọn tracker production.",
        "",
        "## Theo từng video",
        "",
        "| Video | Profile | Frames decoded/meta | FPS | Tracks P/B | Median length | Short ≤5 | Fragments | Coverage P/B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(summary_rows, key=lambda item: (str(item["video"]), str(item["tracker"]))):
        lines.append(
            f"| {row['video']} | {row['tracker']} | {row['decoded_frames']}/{row['metadata_frames']} | "
            f"{_number(row['steady_tracker_fps'])} | {row['person_tracks']}/{row['bag_tracks']} | "
            f"{_number(row['median_track_length'], 1)} | {row['short_tracks_le_5']} | "
            f"{row['fragment_candidates']} | {_number(100 * float(row['coverage_person']), 1)}%/"
            f"{_number(100 * float(row['coverage_bag']), 1)}% |"
        )

    lines.extend([
        "",
        "## Ưu, nhược điểm trong bài toán cướp giật túi",
        "",
        "| Profile | Ưu điểm | Nhược điểm | Mức phù hợp dự án |",
        "|---|---|---|---|",
    ])
    for profile in profiles:
        lines.append(
            f"| `{profile.name}` | {profile.strengths} | {profile.weaknesses} | {profile.fit_for_project} |"
        )

    lines.extend([
        "",
        "## Cách kết luận chính xác",
        "",
        "1. Chọn một tập nhỏ video đại diện và sửa ground-truth bbox/track ID bằng CVAT.",
        "2. Tính HOTA, IDF1, ID switches và fragmentation riêng cho `person` và `bag`.",
        "3. Loại tracker có recall bag thấp dù continuity trông đẹp.",
        "4. Trong các tracker đạt chất lượng, chọn tracker thỏa FPS/latency của edge device.",
        "5. Đánh giá thêm độ ổn định của feature downstream: vận tốc, gia tốc, person–bag distance và ownership change.",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
