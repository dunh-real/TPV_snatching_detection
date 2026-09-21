"""Benchmark script to measure execution time and Real-Time Factor (RTF).

Calculates:
    RTF = Execution Time / Video Length
    Processing FPS = Total Processed Frames / Execution Time

Breaks down time into:
    - Decode (video reading / demuxing)
    - Inference & Tracking (YOLO + ByteTrack)
    - Analytics (Rule engine / Snatching logic)

Supports:
    - Full video evaluation on video directories (e.g. data/videos/videotest)
    - Chunk-based simulation (e.g. 2s, 3s, 5s, 10s chunks)
    - GPU warm-up & CUDA synchronization for accurate timings on NVIDIA T4 / GPUs
    - CSV and Markdown reporting with capacity sizing recommendations
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analytics.engine import SnatchAnalyticsEngine
from src.core.detector import YOLODetector
from src.core.tracker import ByteTracker

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def sync_cuda() -> None:
    """Synchronize CUDA stream to ensure accurate GPU time measurement."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class VideoBenchmarkResult:
    video_name: str
    width: int
    height: int
    source_fps: float
    total_frames: int
    processed_frames: int
    video_length_s: float
    decode_time_s: float
    infer_track_time_s: float
    analytics_time_s: float
    total_exec_time_s: float
    rtf_total: float  # total_exec_time_s / video_length_s
    rtf_infer_only: float  # infer_track_time_s / video_length_s
    proc_fps: float  # processed_frames / total_exec_time_s
    peak_vram_mb: float
    is_realtime_capable: bool
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark video execution time and RTF (execute_time / video_length) for batching sizing."
    )
    parser.add_argument(
        "--videos",
        default="data/videos/videotest",
        help="Path to video file or directory of test videos (default: data/videos/videotest)",
    )
    parser.add_argument(
        "--model",
        default="models/detection/best.pt",
        help="Path to YOLO weights",
    )
    parser.add_argument(
        "--tracker",
        default="configs/custom_tracker.yaml",
        help="Path to Ultralytics tracker config",
    )
    parser.add_argument(
        "--rules-config",
        default="configs/snatch_rules.yaml",
        help="Path to snatch rule configuration",
    )
    parser.add_argument(
        "--disable-analytics",
        action="store_true",
        help="Measure only detector + tracker without rule analytics",
    )
    parser.add_argument(
        "--track-conf",
        type=float,
        default=0.1,
        help="Minimum detector confidence passed to tracker",
    )
    parser.add_argument("--person-conf", type=float, default=0.5)
    parser.add_argument("--bag-conf", type=float, default=0.25)
    parser.add_argument("--default-conf", type=float, default=0.4)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (1 = process every frame, 2 = skip every 2nd frame)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap per video (useful for fast smoke tests)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="Number of dummy frames to warm up GPU before measuring",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=float,
        nargs="*",
        default=[],
        help="Optional list of chunk lengths in seconds to simulate batching (e.g. --chunk-sizes 2 3 5 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_results",
        help="Directory to save CSV and Markdown reports",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on: '0', 'cuda', or 'cpu'. Auto-detects GPU if available.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def discover_video_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path]
    if not source_path.is_dir():
        raise FileNotFoundError(f"Video path not found: {source_path}")
    files = sorted(
        [
            p
            for p in source_path.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
    if not files:
        raise FileNotFoundError(f"No video files found in: {source_path}")
    return files


def warmup_gpu(
    detector: YOLODetector,
    tracker: ByteTracker,
    warmup_frames: int = 15,
    w: int = 640,
    h: int = 640,
) -> None:
    """Warm up PyTorch, CUDA context, and cuDNN kernels with synthetic frames."""
    if warmup_frames <= 0:
        return
    print(f"[*] Warming up GPU / model with {warmup_frames} dummy frames...", flush=True)
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(warmup_frames):
        tracker.update(dummy, i, i * 33.3, include_unreliable=True)
    sync_cuda()
    tracker.reset()
    print("[*] Warm-up complete.\n", flush=True)


def benchmark_single_video(
    video_path: Path,
    detector: YOLODetector,
    tracker: ByteTracker,
    analytics_engine: SnatchAnalyticsEngine | None,
    stride: int = 1,
    max_frames: int | None = None,
) -> VideoBenchmarkResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker.reset()
    if analytics_engine is not None:
        analytics_engine.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    decode_time_s = 0.0
    infer_track_time_s = 0.0
    analytics_time_s = 0.0

    frame_idx = 0
    processed_count = 0
    last_timestamp_ms = -1.0

    sync_cuda()
    wall_start = time.perf_counter()

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            # 1. Đo Decode / Video Reading
            sync_cuda()
            t0 = time.perf_counter()
            ret, frame = cap.read()
            sync_cuda()
            t1 = time.perf_counter()

            if not ret or frame is None:
                break

            decode_time_s += (t1 - t0)

            # Stride logic: chỉ xử lý nếu khớp stride
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            reported_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            expected_timestamp_ms = frame_idx * 1000.0 / source_fps
            if reported_timestamp_ms > last_timestamp_ms:
                timestamp_ms = reported_timestamp_ms
            else:
                timestamp_ms = max(
                    expected_timestamp_ms,
                    last_timestamp_ms + 1000.0 / source_fps,
                )
            last_timestamp_ms = timestamp_ms

            # 2. Đo Inference & Tracking
            sync_cuda()
            t_inf0 = time.perf_counter()
            tracked = tracker.update(
                frame,
                frame_idx,
                timestamp_ms,
                include_unreliable=analytics_engine is not None,
            )
            sync_cuda()
            t_inf1 = time.perf_counter()
            infer_track_time_s += (t_inf1 - t_inf0)

            # 3. Đo Rule Engine Analytics
            if analytics_engine is not None:
                sync_cuda()
                t_an0 = time.perf_counter()
                analytics_engine.update(
                    tracked,
                    frame_idx,
                    timestamp_ms,
                    frame.shape,
                )
                sync_cuda()
                t_an1 = time.perf_counter()
                analytics_time_s += (t_an1 - t_an0)

            processed_count += 1
            frame_idx += 1

    finally:
        cap.release()

    sync_cuda()
    wall_end = time.perf_counter()
    total_exec_time_s = wall_end - wall_start

    # Tính thời lượng video thực tế được nạp
    video_length_s = (
        (frame_idx / source_fps)
        if frame_idx > 0
        else (total_frames_meta / source_fps)
    )
    if video_length_s <= 0:
        video_length_s = 1e-6

    rtf_total = total_exec_time_s / video_length_s
    rtf_infer_only = infer_track_time_s / video_length_s
    proc_fps = processed_count / total_exec_time_s if total_exec_time_s > 0 else 0.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    is_realtime_capable = rtf_total < 1.0

    return VideoBenchmarkResult(
        video_name=video_path.name,
        width=width,
        height=height,
        source_fps=round(source_fps, 2),
        total_frames=frame_idx,
        processed_frames=processed_count,
        video_length_s=round(video_length_s, 2),
        decode_time_s=round(decode_time_s, 4),
        infer_track_time_s=round(infer_track_time_s, 4),
        analytics_time_s=round(analytics_time_s, 4),
        total_exec_time_s=round(total_exec_time_s, 4),
        rtf_total=round(rtf_total, 4),
        rtf_infer_only=round(rtf_infer_only, 4),
        proc_fps=round(proc_fps, 2),
        peak_vram_mb=round(peak_vram_mb, 2),
        is_realtime_capable=is_realtime_capable,
    )


def simulate_chunk_batching(
    video_path: Path,
    chunk_seconds: float,
    detector: YOLODetector,
    tracker: ByteTracker,
    analytics_engine: SnatchAnalyticsEngine | None,
) -> dict[str, float]:
    """Simulates slicing a video into small chunks of `chunk_seconds` and processing them.

    Measures:
    - Average executing time per chunk
    - Chunk RTF (chunk_exec_time / chunk_seconds)
    - Max chunk latency (Worst-case time from frame arrival to detection alert)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    chunk_frames = max(1, int(round(chunk_seconds * source_fps)))

    tracker.reset()
    if analytics_engine:
        analytics_engine.reset()

    chunk_exec_times: list[float] = []
    chunk_infer_times: list[float] = []

    frame_idx = 0
    while True:
        # Đọc 1 chunk gồm chunk_frames
        chunk_frames_list = []
        sync_cuda()
        t_read0 = time.perf_counter()
        for _ in range(chunk_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            chunk_frames_list.append(frame)
        sync_cuda()
        t_read1 = time.perf_counter()

        if not chunk_frames_list:
            break

        actual_chunk_len_s = len(chunk_frames_list) / source_fps

        # Xử lý chunk
        sync_cuda()
        t_chunk_start = time.perf_counter()
        chunk_infer_time = 0.0

        for frame in chunk_frames_list:
            timestamp_ms = frame_idx * 1000.0 / source_fps

            sync_cuda()
            t_inf0 = time.perf_counter()
            tracked = tracker.update(
                frame,
                frame_idx,
                timestamp_ms,
                include_unreliable=analytics_engine is not None,
            )
            sync_cuda()
            t_inf1 = time.perf_counter()
            chunk_infer_time += (t_inf1 - t_inf0)

            if analytics_engine:
                analytics_engine.update(
                    tracked,
                    frame_idx,
                    timestamp_ms,
                    frame.shape,
                )
            frame_idx += 1

        sync_cuda()
        t_chunk_end = time.perf_counter()

        # Tổng thời gian cho chunk = time đọc + time xử lý
        total_chunk_time = (t_read1 - t_read0) + (t_chunk_end - t_chunk_start)
        chunk_exec_times.append(total_chunk_time)
        chunk_infer_times.append(chunk_infer_time)

    cap.release()

    if not chunk_exec_times:
        return {}

    avg_chunk_exec_s = sum(chunk_exec_times) / len(chunk_exec_times)
    avg_chunk_rtf = avg_chunk_exec_s / chunk_seconds
    worst_latency_s = chunk_seconds + max(chunk_exec_times)

    return {
        "chunk_seconds": chunk_seconds,
        "num_chunks": len(chunk_exec_times),
        "avg_chunk_exec_s": round(avg_chunk_exec_s, 4),
        "avg_chunk_rtf": round(avg_chunk_rtf, 4),
        "worst_latency_s": round(worst_latency_s, 4),
    }


def print_ascii_table(results: list[VideoBenchmarkResult]) -> None:
    headers = [
        "Video Name",
        "Frames",
        "FPS",
        "Length(s)",
        "Decode(s)",
        "Infer(s)",
        "Rules(s)",
        "Total(s)",
        "RTF (Total)",
        "Proc FPS",
        "Realtime?",
    ]

    rows = []
    for r in results:
        status = "YES (Fast)" if r.rtf_total < 1.0 else "NO (Laggy)"
        rows.append(
            [
                r.video_name[:20],
                str(r.total_frames),
                f"{r.source_fps:.1f}",
                f"{r.video_length_s:.2f}",
                f"{r.decode_time_s:.2f}",
                f"{r.infer_track_time_s:.2f}",
                f"{r.analytics_time_s:.2f}",
                f"{r.total_exec_time_s:.2f}",
                f"{r.rtf_total:.3f}",
                f"{r.proc_fps:.1f}",
                status,
            ]
        )

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    print("\n" + separator)
    print(header_str)
    print(separator)
    for row in rows:
        row_str = "| " + " | ".join(val.ljust(w) for val, w in zip(row, col_widths)) + " |"
        print(row_str)
    print(separator + "\n")


def generate_sizing_summary(results: list[VideoBenchmarkResult]) -> dict[str, object]:
    if not results:
        return {}

    rtf_list = [r.rtf_total for r in results]
    infer_rtf_list = [r.rtf_infer_only for r in results]
    proc_fps_list = [r.proc_fps for r in results]

    avg_rtf = float(np.mean(rtf_list))
    median_rtf = float(np.median(rtf_list))
    max_rtf = float(np.max(rtf_list))
    min_rtf = float(np.min(rtf_list))

    avg_infer_rtf = float(np.mean(infer_rtf_list))
    avg_proc_fps = float(np.mean(proc_fps_list))

    # Ước lượng số luồng camera 1 GPU gánh được an toàn
    # Để an toàn (headroom 20%), dùng công thức: (1 / avg_rtf) * 0.8
    concurrent_streams_safe = int(np.floor((1.0 / avg_rtf) * 0.8)) if avg_rtf > 0 else 0
    concurrent_streams_max = int(np.floor(1.0 / avg_rtf)) if avg_rtf > 0 else 0

    return {
        "num_videos": len(results),
        "avg_rtf": round(avg_rtf, 4),
        "median_rtf": round(median_rtf, 4),
        "min_rtf": round(min_rtf, 4),
        "max_rtf": round(max_rtf, 4),
        "avg_infer_rtf": round(avg_infer_rtf, 4),
        "avg_proc_fps": round(avg_proc_fps, 2),
        "safe_concurrent_cctv_per_gpu": max(1, concurrent_streams_safe),
        "theoretical_max_cctv_per_gpu": max(1, concurrent_streams_max),
    }


def export_reports(
    results: list[VideoBenchmarkResult],
    sizing: dict[str, object],
    chunk_results: list[dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"rtf_benchmark_{timestamp}.csv"
    md_path = output_dir / f"rtf_report_{timestamp}.md"

    # Export CSV
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(asdict(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Export Markdown
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐO LƯỜNG EXECUTING TIME & RTF (REAL-TIME FACTOR)\n\n")
        f.write(f"- **Thời gian chạy**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Môi trường**: {platform.system()} {platform.release()} | Python {platform.python_version()}\n")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        f.write(f"- **Thiết bị tính toán**: `{device_name}`\n")
        f.write(f"- **Mô hình**: `{args.model}`\n")
        f.write(f"- **Tracker**: `{args.tracker}`\n")
        f.write(f"- **Thư mục video**: `{args.videos}`\n\n")

        f.write("## 1. Tóm tắt chỉ số & Đánh giá năng lực (Sizing)\n\n")
        f.write("| Chỉ số | Giá trị | Ý nghĩa |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **RTF Trung bình (Total)** | **{sizing.get('avg_rtf')}** | Số giây GPU cần để xử lý 1 giây video |\n")
        f.write(f"| **RTF Trung vị (Median)** | {sizing.get('median_rtf')} | Đại diện cho phân phối thông thường |\n")
        f.write(f"| **RTF Min - Max** | {sizing.get('min_rtf')} - {sizing.get('max_rtf')} | Biên độ dao động phụ thuộc mật độ người/xe |\n")
        f.write(f"| **RTF Chỉ riêng Inference** | {sizing.get('avg_infer_rtf')} | RTF khi loại bỏ I/O decode và rule logic |\n")
        f.write(f"| **Processing Speed (FPS)** | **{sizing.get('avg_proc_fps')} FPS** | Tốc độ xử lý khung hình trung bình |\n")
        f.write(f"| **Số CCTV tối đa / 1 GPU (Lý thuyết)** | **{sizing.get('theoretical_max_cctv_per_gpu')} luồng** | $1.0 / \\text{{RTF}}_{{\\text{{avg}}}}$ |\n")
        f.write(f"| **Số CCTV an toàn / 1 GPU (Thực tế)** | **{sizing.get('safe_concurrent_cctv_per_gpu')} luồng** | Tính thêm 20% dung sai đỉnh điểm |\n\n")

        f.write("## 2. Chi tiết từng Video trong tập kiểm thử\n\n")
        f.write("| Video | Frames | FPS | Độ dài (s) | Decode (s) | Infer (s) | Rules (s) | Tổng Exec (s) | RTF | Proc FPS | Đạt Realtime? |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            status = "✅ Đạt" if r.rtf_total < 1.0 else "❌ Quá tải"
            f.write(
                f"| `{r.video_name}` | {r.total_frames} | {r.source_fps} | {r.video_length_s}s | "
                f"{r.decode_time_s}s | {r.infer_track_time_s}s | {r.analytics_time_s}s | "
                f"{r.total_exec_time_s}s | **{r.rtf_total}** | {r.proc_fps} | {status} |\n"
            )

        if chunk_results:
            f.write("\n## 3. Khảo sát Mô phỏng Chia Batch (Chunking Simulation)\n\n")
            f.write("| Độ dài Chunk (s) | Số Chunk | Exec Time TB / Chunk (s) | RTF Chunk | Độ trễ Cảnh báo Tối đa (Latency) |\n")
            f.write("|---|---|---|---|---|\n")
            for c in chunk_results:
                f.write(
                    f"| {c['chunk_seconds']}s | {c['num_chunks']} | {c['avg_chunk_exec_s']}s | "
                    f"**{c['avg_chunk_rtf']}** | ~**{c['worst_latency_s']}s** |\n"
                )

        f.write("\n## 4. Khuyến nghị cho Kiến trúc Hệ thống (Realtime vs Batching)\n\n")
        avg_rtf = sizing.get("avg_rtf", 1.0)
        if avg_rtf < 0.3:
            f.write(
                "> **Nhận định**: Hệ thống xử lý cực nhanh (RTF < 0.3, tức là tốc độ gấp > 3 lần realtime).\n"
                "> - **Khuyến nghị**: Hoàn toàn có thể chạy **Realtime Streaming** cho cụm 2-4 camera trên 1 GPU T4.\n"
                "> - Nếu muốn chạy **Batching**, chỉ nên cắt đoạn ngắn (3s - 5s) để giữ độ trễ cảnh báo dưới 5s, đủ nhanh để bảo vệ can thiệp khi có cướp giật.\n"
            )
        elif avg_rtf < 1.0:
            f.write(
                "> **Nhận định**: Hệ thống nhanh hơn realtime ở mức vừa phải (0.3 <= RTF < 1.0).\n"
                "> - **Khuyến nghị**: Chạy **Batching với Chunk 4s - 6s** (sliding window 2s-3s) là phương án tối ưu nhất. Phương án này vừa giúp GPU tận dụng batch size tốt hơn, vừa bảo đảm hệ thống không bị crash khi mạng camera chập chờn.\n"
            )
        else:
            f.write(
                "> **Nhận định**: Hệ thống chậm hơn realtime (RTF >= 1.0).\n"
                "> - **Khuyến nghị**: BẮT BUỘC dùng kiến trúc **Batching / Worker Queue**. Cần cân nhắc áp dụng Frame Stride (ví dụ chỉ lấy 15 FPS thay vì 30 FPS) hoặc xuất model sang định dạng TensorRT (`model.export(format='engine')`) để đưa RTF xuống dưới 0.5.\n"
            )

    return csv_path, md_path


def main() -> int:
    args = parse_args()
    print("=" * 65)
    print("      TROMCAP BENCHMARK: EXECUTING TIME & RTF SIZING      ")
    print("=" * 65)

    video_source = resolve_path(args.videos)
    video_files = discover_video_files(video_source)
    print(f"[*] Found {len(video_files)} video(s) to benchmark under: {video_source}")

    # Determine and force device
    if args.device is not None:
        target_device = args.device
    else:
        target_device = "0" if torch.cuda.is_available() else "cpu"

    if target_device == "cpu" or not torch.cuda.is_available():
        device_name = "CPU"
        print("\n" + "!" * 67)
        print(" [!] CẢNH BÁO: BẠN ĐANG CHẠY TRÊN CPU, KHÔNG PHẢI GPU!")
        print("     - Tốc độ xử lý trên CPU của Colab chỉ đạt ~2-3 FPS (RTF > 10).")
        print("     - Cách khắc phục trên Google Colab:")
        print("       1. Vào thanh menu trên cùng: Runtime -> Change runtime type")
        print("       2. Mục 'Hardware accelerator': Chọn 'T4 GPU'")
        print("       3. Bấm 'Save' rồi chạy lại notebook.")
        print("!" * 67 + "\n")
    else:
        device_name = f"{torch.cuda.get_device_name(0)} (CUDA Active)"
        print(f"\n[*] GPU ĐƯỢC KÍCH HOẠT: {device_name}\n")

    print(f"[*] Execution Device: {device_name}")
    print(f"[*] Model: {args.model}")
    print(f"[*] Tracker: {args.tracker}")
    print(f"[*] Rules Enabled: {not args.disable_analytics}")
    print(f"[*] Stride: {args.stride}")

    confidence = {
        "person": args.person_conf,
        "bag": args.bag_conf,
        "default": args.default_conf,
    }

    print("[*] Loading detector and tracker...", flush=True)
    detector = YOLODetector(model_path=args.model, conf=confidence, device=target_device)
    tracker = ByteTracker(
        detector=detector,
        tracker_config=args.tracker,
        tracking_conf=args.track_conf,
        device=target_device,
    )

    analytics_engine = None
    if not args.disable_analytics:
        analytics_engine = SnatchAnalyticsEngine.from_yaml(args.rules_config)

    # Warmup GPU
    warmup_gpu(detector, tracker, warmup_frames=args.warmup_frames)

    # Benchmark individual videos
    results: list[VideoBenchmarkResult] = []
    print(f"[*] Starting benchmark on {len(video_files)} videos...\n")

    for idx, video_path in enumerate(video_files, start=1):
        print(f"[{idx}/{len(video_files)}] Benchmarking: {video_path.name} ... ", end="", flush=True)
        try:
            res = benchmark_single_video(
                video_path=video_path,
                detector=detector,
                tracker=tracker,
                analytics_engine=analytics_engine,
                stride=args.stride,
                max_frames=args.max_frames,
            )
            results.append(res)
            status_symbol = "[OK]" if res.is_realtime_capable else "[!]"
            print(
                f"{status_symbol} Done ({res.total_frames} frames, "
                f"Video: {res.video_length_s}s, Exec: {res.total_exec_time_s}s, "
                f"RTF: {res.rtf_total}, Speed: {res.proc_fps} FPS)"
            )
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("[!] No benchmark results collected.")
        return 1

    # In bảng ASCII ra màn hình console
    print_ascii_table(results)

    # Tổng kết sizing
    sizing = generate_sizing_summary(results)
    print("=" * 65)
    print("                      SIZING SUMMARY                      ")
    print("=" * 65)
    print(f"  - Average RTF (Exec / Video Length): {sizing['avg_rtf']}")
    print(f"  - Median RTF                       : {sizing['median_rtf']}")
    print(f"  - Min ~ Max RTF                    : {sizing['min_rtf']} ~ {sizing['max_rtf']}")
    print(f"  - Pure Inference RTF               : {sizing['avg_infer_rtf']}")
    print(f"  - Average Processing Speed         : {sizing['avg_proc_fps']} FPS")
    print(f"  - Theoretical Max Cameras / 1 GPU  : {sizing['theoretical_max_cctv_per_gpu']} cameras")
    print(f"  - Safe Concurrent Cameras / 1 GPU  : {sizing['safe_concurrent_cctv_per_gpu']} cameras (with 20% margin)")
    print("=" * 65 + "\n")

    # Mô phỏng chia batch nếu người dùng truyền --chunk-sizes
    chunk_results: list[dict[str, object]] = []
    if args.chunk_sizes and len(video_files) > 0:
        print("[*] Running Batching/Chunking Simulation on sample video...")
        sample_video = video_files[0]
        for c_size in args.chunk_sizes:
            print(f"    - Testing chunk length: {c_size}s ... ", end="", flush=True)
            c_res = simulate_chunk_batching(
                video_path=sample_video,
                chunk_seconds=float(c_size),
                detector=detector,
                tracker=tracker,
                analytics_engine=analytics_engine,
            )
            if c_res:
                chunk_results.append(c_res)
                print(
                    f"Done. Avg Exec: {c_res['avg_chunk_exec_s']}s, "
                    f"Chunk RTF: {c_res['avg_chunk_rtf']}, Max Latency: ~{c_res['worst_latency_s']}s"
                )
            else:
                print("Failed")
        print()

    # Xuất file báo cáo
    output_dir = resolve_path(args.output_dir)
    csv_file, md_file = export_reports(results, sizing, chunk_results, output_dir, args)
    print(f"[OK] Saved CSV report to: {csv_file}")
    print(f"[OK] Saved Markdown report to: {md_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
