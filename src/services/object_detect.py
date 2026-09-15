"""Module dịch vụ phát hiện đối tượng."""

import numpy as np

from src.core.vision.detector import YOLODetector
from src.core.types import Detection, TrackedObject
from src.core.vision.tracker import ByteTracker


class ObjectDetectionService:
    """Service wrapping YOLO detection and ByteTrack tracking."""

    def __init__(
        self,
        model_path: str = "models/detection/best.pt",
        conf: float | dict[str, float] = 0.6,
        tracker_config: str = "configs/custom_tracker.yaml",
        tracking_conf: float = 0.1,
    ):
        self.detector = YOLODetector(model_path=model_path, conf=conf)
        self.tracker = ByteTracker(
            detector=self.detector,
            tracker_config=tracker_config,
            tracking_conf=tracking_conf,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a single frame."""
        return self.detector.detect(frame)

    def track(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: float,
        include_unreliable: bool = False,
    ) -> list[TrackedObject]:
        """Track objects across frames."""
        return self.tracker.update(
            frame=frame,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            include_unreliable=include_unreliable,
        )

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.tracker.reset()
