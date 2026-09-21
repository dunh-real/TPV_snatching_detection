"""Bộ bám vết đối tượng sử dụng cấu hình tracker của Ultralytics."""

import numpy as np

from src.core.vision.detector import YOLODetector
from src.core.types import TrackedObject


class ByteTracker:
    """Wrap YOLO ``model.track()`` while preserving state between frames."""

    def __init__(
        self,
        detector: YOLODetector,
        tracker_config: str = "configs/custom_tracker.yaml",
        tracking_conf: float = 0.1,
        device: str | int | None = None,
    ):
        if not 0.0 <= tracking_conf <= 1.0:
            raise ValueError("tracking_conf must be between 0 and 1")

        self.detector = detector
        self.tracker_config = tracker_config
        self.tracking_conf = tracking_conf
        self.device = device if device is not None else getattr(detector, "device", None)

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: float,
        include_unreliable: bool = False,
    ) -> list[TrackedObject]:
        """Detect and track objects in one frame.

        By default this preserves the previous public behavior and emits only
        class-thresholded observations. Analytics can opt into weaker tracked
        boxes to maintain temporal state without treating them as reliable.
        """
        
        results = self.detector.model.track(
            frame,
            # ByteTrack needs low-confidence boxes for its second association pass.
            # Class-specific thresholds are applied only when results are emitted.
            conf=self.tracking_conf,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
            device=self.device,
        )
        tracked = []
        for r in results:
            if r.boxes.id is None:
                continue
            # Tracked rows are [x1, y1, x2, y2, track_id, confidence, class].
            # Copy the batch once instead of synchronizing CUDA for every scalar.
            rows = r.boxes.data.detach().cpu().numpy()
            for row in rows:
                cls_id = int(row[-1])
                confidence = float(row[-2])
                track_id = int(row[-3])
                label = self.detector.class_names[cls_id]
                
                is_reliable = confidence >= self.detector.get_threshold(label)
                if is_reliable or include_unreliable:
                    tracked.append(TrackedObject(
                        track_id=track_id,
                        bbox=tuple(float(value) for value in row[:4]),
                        confidence=confidence,
                        class_id=cls_id,
                        label=label,
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        is_reliable=is_reliable,
                    ))
        return tracked

    def reset(self):
        """Reset tracker state for a new video."""
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            if hasattr(self.detector.model.predictor, "trackers"):
                for t in self.detector.model.predictor.trackers:
                    t.reset()
