"""ByteTrack object tracker using ultralytics built-in tracker."""

import numpy as np

from src.core.detector import YOLODetector
from src.core.models import TrackedObject


class ByteTracker:
    """Wraps YOLO model.track() with ByteTrack for multi-object tracking."""

    def __init__(self, detector: YOLODetector):
        self.detector = detector

    def update(
        self, frame: np.ndarray, frame_idx: int, timestamp_ms: float
    ) -> list[TrackedObject]:
        """Detect + track objects in a frame, return TrackedObjects filtered by per-class threshold."""
        
        # Inject missing tracking args if needed due to old model checkpoint
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            if not hasattr(self.detector.model.predictor.args, "fuse_score"):
                setattr(self.detector.model.predictor.args, "fuse_score", False)
                
        results = self.detector.model.track(
            frame,
            conf=self.detector.base_conf,
            tracker="bytetrack.yaml",
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
                
                # Filter by class-specific threshold
                if confidence >= self.detector.get_threshold(label):
                    tracked.append(TrackedObject(
                        track_id=int(tid),
                        bbox=tuple(box.xyxy[0].tolist()),
                        confidence=confidence,
                        class_id=cls_id,
                        label=label,
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                    ))
        return tracked

    def reset(self):
        """Reset tracker state for a new video."""
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            if hasattr(self.detector.model.predictor, "trackers"):
                for t in self.detector.model.predictor.trackers:
                    t.reset()
