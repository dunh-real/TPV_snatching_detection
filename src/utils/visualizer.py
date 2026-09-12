"""Visualization utilities for drawing bounding boxes."""

import cv2
import numpy as np

from src.analytics.models import AnalyticsResult, EventState, PersonRole
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
            # if not getattr(obj, "is_reliable", True) and obj.label == "person" and not draw_low_conf_person:
            #     continue
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

    @staticmethod
    def draw_analytics(frame: np.ndarray, result: AnalyticsResult) -> np.ndarray:
        """Draw canonical IDs, person roles, holder links, and event alerts."""
        entities_by_id = {item.entity_id: item for item in result.entities}
        relations_by_bag = {item.bag_id: item for item in result.relations}

        for relation in result.relations:
            bag = entities_by_id.get(relation.bag_id)
            holder = entities_by_id.get(relation.holder_person_id)
            if bag is None or holder is None or not bag.observed or not holder.observed:
                continue
            bag_center = Visualizer._bbox_center_int(bag.bbox)
            holder_center = Visualizer._bbox_center_int(holder.bbox)
            color = (0, 200, 255) if relation.candidate_holder_id else (46, 204, 113)
            cv2.line(frame, bag_center, holder_center, color, 2)

        for entity in result.entities:
            if not entity.observed:
                continue
            x1, y1, x2, y2 = map(int, entity.bbox)
            if entity.entity_id in result.roles:
                role = result.roles.get(entity.entity_id)
                role_value = role.role if role else PersonRole.UNKNOWN
                color = Visualizer._role_color(role_value)
                label = f"P#{entity.entity_id} {role_value.value.upper()}"
            elif entity.entity_id in relations_by_bag:
                relation = relations_by_bag.get(entity.entity_id)
                color = (0, 140, 255)
                label = f"B#{entity.entity_id}"
                if relation and relation.holder_person_id is not None:
                    label += f" holder=P#{relation.holder_person_id}"
            else:
                color = COLORS[entity.entity_id % len(COLORS)]
                label = f"E#{entity.entity_id} {entity.label}"

            if not entity.reliable:
                color = tuple(int(channel * 0.6) for channel in color)
                label += " low-conf"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            Visualizer._draw_label(frame, label, x1, y1, color)

        visible_events = [
            event
            for event in result.events
            if event.state
            in {EventState.TRANSFER_PENDING, EventState.ESCAPING, EventState.SUSPECTED}
        ]
        if visible_events:
            event = max(
                visible_events,
                key=lambda item: (item.state == EventState.SUSPECTED, item.score),
            )
            banner = (
                f"{event.state.value.upper()} score={event.score:.1f} "
                f"B#{event.bag_id} P#{event.victim_person_id}->P#{event.suspect_person_id}"
            )
            banner_color = (
                (0, 0, 255) if event.state == EventState.SUSPECTED else (0, 165, 255)
            )
            Visualizer._draw_label(frame, banner, 10, 28, banner_color)
        return frame

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y = max(y, text_height + 6)
        cv2.rectangle(
            frame,
            (x, text_y - text_height - 6),
            (x + text_width, text_y),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    @staticmethod
    def _bbox_center_int(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    @staticmethod
    def _role_color(role: PersonRole) -> tuple[int, int, int]:
        return {
            PersonRole.NORMAL: (46, 204, 113),
            PersonRole.UNKNOWN: (149, 165, 166),
            PersonRole.POSSIBLE_VICTIM: (255, 200, 0),
            PersonRole.POSSIBLE_SUSPECT: (0, 165, 255),
            PersonRole.VICTIM: (255, 120, 30),
            PersonRole.SUSPECT: (0, 0, 255),
        }[role]
