"""Object tracker using an Ultralytics tracker configuration."""

import numpy as np

from src.core.detector import YOLODetector
from src.core.models import TrackedObject


class ByteTracker:
    """Wrap YOLO ``model.track()`` while preserving state between frames."""

    def __init__(
        self,
        detector: YOLODetector,
        tracker_config: str = "configs/custom_tracker.yaml",
        tracking_conf: float = 0.1,
    ):
        if not 0.0 <= tracking_conf <= 1.0:
            raise ValueError("tracking_conf must be between 0 and 1")

        self.detector = detector
        self.tracker_config = tracker_config
        self.tracking_conf = tracking_conf

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
        )
        tracked = []
        for r in results:
            if r.boxes.id is None:
                continue
            for box, tid in zip(r.boxes, r.boxes.id):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.detector.class_names[cls_id]
                
                is_reliable = confidence >= self.detector.get_threshold(label)
                if is_reliable or include_unreliable:
                    tracked.append(TrackedObject(
                        track_id=int(tid),
                        bbox=tuple(box.xyxy[0].tolist()),
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
