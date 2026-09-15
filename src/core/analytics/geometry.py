"""Các tiện ích hình học nhỏ, không phụ thuộc ngoài, cho analytics và tracking."""

import math

from src.core.types import BBox


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_size(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def bbox_bottom_center(bbox: BBox) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def normalized_scale(bbox: BBox, frame_shape: tuple[int, ...]) -> float:
    """Return log object scale relative to the frame, not a physical z coordinate."""
    height, width = frame_shape[:2]
    box_width, box_height = bbox_size(bbox)
    relative_area = max((box_width * box_height) / max(width * height, 1), 1e-12)
    return 0.5 * math.log(relative_area)


def expand_bbox(bbox: BBox, ratio: float) -> BBox:
    x1, y1, x2, y2 = bbox
    width, height = bbox_size(bbox)
    return (
        x1 - width * ratio,
        y1 - height * ratio,
        x2 + width * ratio,
        y2 + height * ratio,
    )


def point_to_bbox_distance(point: tuple[float, float], bbox: BBox) -> float:
    """Euclidean distance from a point to a box; zero means the point is inside."""
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def normalized_point_to_person_distance(point: tuple[float, float], person_bbox: BBox) -> float:
    _, person_height = bbox_size(person_bbox)
    return point_to_bbox_distance(point, person_bbox) / person_height


def normalized_center_distance(object_bbox: BBox, person_bbox: BBox) -> float:
    """Center distance measured in units of person height."""
    object_center = bbox_center(object_bbox)
    person_center = bbox_center(person_bbox)
    _, person_height = bbox_size(person_bbox)
    return math.dist(object_center, person_center) / person_height


def normalized_person_distance(first: BBox, second: BBox) -> float:
    """Person center distance measured using the first person's height."""
    _, first_height = bbox_size(first)
    return math.dist(bbox_center(first), bbox_center(second)) / first_height


def relative_bbox_offset(
    object_bbox: BBox,
    person_bbox: BBox,
) -> tuple[float, float, float, float]:
    object_cx, object_cy = bbox_center(object_bbox)
    person_cx, person_cy = bbox_center(person_bbox)
    object_width, object_height = bbox_size(object_bbox)
    person_width, person_height = bbox_size(person_bbox)
    return (
        (object_cx - person_cx) / person_width,
        (object_cy - person_cy) / person_height,
        object_width / person_width,
        object_height / person_height,
    )


def bbox_from_relative_offset(
    offset: tuple[float, float, float, float],
    person_bbox: BBox,
) -> BBox:
    relative_x, relative_y, relative_width, relative_height = offset
    person_cx, person_cy = bbox_center(person_bbox)
    person_width, person_height = bbox_size(person_bbox)
    width = max(1.0, relative_width * person_width)
    height = max(1.0, relative_height * person_height)
    cx = person_cx + relative_x * person_width
    cy = person_cy + relative_y * person_height
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def cosine_similarity(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm < 1e-9 or second_norm < 1e-9:
        return 0.0
    dot_product = first[0] * second[0] + first[1] * second[1]
    return max(-1.0, min(1.0, dot_product / (first_norm * second_norm)))


def translate_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy
