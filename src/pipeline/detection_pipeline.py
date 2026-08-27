"""Detection pipeline: video/image → detect → track → DB → visualize."""

import cv2
from pathlib import Path

from src.core.detector import YOLODetector
from src.core.tracker import ByteTracker
from src.database.operations import DetectionDB
from src.utils.visualizer import Visualizer


class DetectionPipeline:
    """Orchestrates the full Phase-1 detection & tracking workflow."""

    def __init__(
        self,
        model_path: str = "models/detection/best_yolov8s.pt",
        db_path: str = "data/detections.db",
        conf: float | dict[str, float] = 0.4,
        tracker_config: str = "configs/custom_tracker.yaml",
        tracking_conf: float = 0.1,
    ):
        self.detector = YOLODetector(model_path, conf)
        self.tracker = ByteTracker(self.detector, tracker_config, tracking_conf)
        self.db = DetectionDB(db_path)
        self.vis = Visualizer()

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
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source}")

        video_id = self.db.create_video(source, fps=0, total_frames=1)
        detections = self.detector.detect(frame)
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
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                tracked = self.tracker.update(frame, frame_idx, timestamp_ms)
                self.db.insert_detections(video_id, tracked)

                vis_frame = self.vis.draw_tracked(frame, tracked)

                if writer:
                    writer.write(vis_frame)
                if show:
                    cv2.imshow("Snatching Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"  Processed {frame_idx}/{total} frames ...")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        print(f"Done — {frame_idx} frames processed, video_id={video_id}")
        return video_id
