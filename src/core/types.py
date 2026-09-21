"""Các kiểu dữ liệu dùng chung cho adapter phát hiện và bám vết."""

from dataclasses import dataclass


BBox = tuple[float, float, float, float]


@dataclass
class Detection:
    """Raw detection result from YOLO."""
    bbox: BBox  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str


@dataclass
class TrackedObject:
    """Detection enriched with tracking ID and timestamp."""
    track_id: int
    bbox: BBox  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str
    frame_idx: int
    timestamp_ms: float
    # Low-confidence boxes can maintain analytics state without being emitted
    # as normal detections or triggering an event on their own.
    is_reliable: bool = True
