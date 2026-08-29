"""Visualization utilities for drawing bounding boxes."""

import cv2
import numpy as np

from src.core.models import Detection, TrackedObject

# Color palette — indexed by track_id % len(COLORS)
COLORS = [
    (46, 204, 113), (231, 76, 60), (52, 152, 219), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


class Visualizer:
    """Draws detection / tracking results onto frames."""

    @staticmethod
    def draw_tracked(frame: np.ndarray, objects: list[TrackedObject]) -> np.ndarray:
        """Draw bounding boxes with track ID and label."""
        for obj in objects:
            color = COLORS[obj.track_id % len(COLORS)]
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{obj.track_id} {obj.label} {obj.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(y1, th + 6)
            cv2.rectangle(frame, (x1, text_y - th - 6), (x1 + tw, text_y), color, -1)
            cv2.putText(frame, label, (x1, text_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes for plain detections (no tracking ID)."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 2)
            label = f"{det.label} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(y1, th + 6)
            cv2.rectangle(frame, (x1, text_y - th - 6), (x1 + tw, text_y), (46, 204, 113), -1)
            cv2.putText(frame, label, (x1, text_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame
