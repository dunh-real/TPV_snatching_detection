"""Data models for detection and tracking."""

from dataclasses import dataclass


@dataclass
class Detection:
    """Raw detection result from YOLO."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str


@dataclass
class TrackedObject:
    """Detection enriched with tracking ID and timestamp."""
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str
    frame_idx: int
    timestamp_ms: float