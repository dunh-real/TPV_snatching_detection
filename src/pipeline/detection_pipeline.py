"""Detection pipeline: input, tracking, rule analytics, storage, visualization."""

import cv2
from pathlib import Path

from src.analytics.engine import SnatchAnalyticsEngine
from src.core.detector import YOLODetector
from src.core.tracker import ByteTracker
from src.database.operations import DetectionDB
from src.utils.latency import LatencyTracker
from src.utils.visualizer import Visualizer


class DetectionPipeline:
    """Orchestrates the full Phase-1 detection & tracking workflow."""

    def __init__(
        self,
        model_path: str = "models/detection/best.pt",
        db_path: str = "data/detections.db",
        conf: float | dict[str, float] = 0.6,
        tracker_config: str = "configs/custom_tracker.yaml",
        tracking_conf: float = 0.1,
        rules_config: str = "configs/snatch_rules.yaml",
        enable_analytics: bool = True,
    ):
        self.detector = YOLODetector(model_path, conf)
        self.tracker = ByteTracker(self.detector, tracker_config, tracking_conf)
        self.analytics = (
            SnatchAnalyticsEngine.from_yaml(rules_config) if enable_analytics else None
        )
        self.db = DetectionDB(db_path)
        self.vis = Visualizer()
        self.latency_tracker = LatencyTracker()

    # ── Public API ────────────────────────────────────────────

    def run(
        self,
        source: str,
        output_dir: str | None = None,
        show: bool = False,
    ) -> int:
        """Run pipeline on a video or image. Returns video_id."""
        ext = Path(source).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            return self._process_image(source, output_dir, show)
        return self._process_video(source, output_dir, show)

    # ── Image ─────────────────────────────────────────────────

    def _process_image(self, source: str, output_dir: str | None, show: bool) -> int:
        self.latency_tracker.reset()
        self.latency_tracker.start_frame()
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source}")

        video_id = self.db.create_video(source, fps=0, total_frames=1)
        detections = self.detector.detect(frame)
        self.latency_tracker.mark_tracking_done()

        vis_frame = self.vis.draw_detections(frame, detections)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            out_filename = f"{Path(source).stem}_vislabel.jpg"
            out_file = out / out_filename
            cv2.imwrite(str(out_file), vis_frame)
            print(f"Saved result to {out_file}")

        if show:
            cv2.imshow("Detection", vis_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.latency_tracker.end_frame()
        self.latency_tracker.print_summary()
        return video_id

    # ── Video ─────────────────────────────────────────────────

    def _process_video(self, source: str, output_dir: str | None, show: bool) -> int:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_id = self.db.create_video(source, fps, total)
        self.tracker.reset()
        if self.analytics:
            self.analytics.reset()
        self.latency_tracker.reset()

        # Prepare video writer
        writer = None
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            out_filename = f"{Path(source).stem}_vislabel.mp4"
            out_file = str(out / out_filename)
            writer = cv2.VideoWriter(
                out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        frame_idx = 0
        last_timestamp_ms = -1.0
        try:
            while True:
                self.latency_tracker.start_frame()
                ret, frame = cap.read()
                if not ret:
                    break

                reported_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                expected_timestamp_ms = frame_idx * 1000.0 / fps
                if reported_timestamp_ms > last_timestamp_ms:
                    timestamp_ms = reported_timestamp_ms
                else:
                    timestamp_ms = max(
                        expected_timestamp_ms,
                        last_timestamp_ms + 1000.0 / fps,
                    )
                last_timestamp_ms = timestamp_ms
                tracked = self.tracker.update(
                    frame,
                    frame_idx,
                    timestamp_ms,
                    include_unreliable=self.analytics is not None,
                )
                self.latency_tracker.mark_tracking_done()

                reliable_tracks = [item for item in tracked if item.is_reliable]
                self.db.insert_detections(
                    video_id,
                    reliable_tracks,
                    commit=self.analytics is None,
                )

                if self.analytics:
                    analytics_result = self.analytics.update(
                        tracked,
                        frame_idx,
                        timestamp_ms,
                        frame.shape,
                    )
                    self.db.insert_analytics(
                        video_id,
                        analytics_result,
                        self.analytics.config.version,
                    )
                    vis_frame = self.vis.draw_analytics(frame, analytics_result)
                else:
                    vis_frame = self.vis.draw_tracked(frame, reliable_tracks)

                if writer:
                    writer.write(vis_frame)
                if show:
                    cv2.imshow("Snatching Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                track_ms, post_ms, total_ms = self.latency_tracker.end_frame()
                frame_idx += 1
                if frame_idx % 100 == 0:
                    fps_est = 1000.0 / total_ms if total_ms > 0 else 0.0
                    print(
                        f"  Processed {frame_idx}/{total} frames ... "
                        f"[Tracking: {track_ms:.1f}ms | Analytics+Post: {post_ms:.1f}ms | Total: {total_ms:.1f}ms (~{fps_est:.1f} FPS)]"
                    )
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        print(f"Done - {frame_idx} frames processed, video_id={video_id}")
        self.latency_tracker.print_summary()
        return video_id
