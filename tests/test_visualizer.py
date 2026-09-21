import unittest
from unittest.mock import patch
import numpy as np

from src.core.analytics.types import (
    AnalyticsResult,
    EntityObservation,
    PersonRole,
    RoleAssignment,
)
from src.core.types import TrackedObject
from src.utils.visualizer import Visualizer


class VisualizerTests(unittest.TestCase):
    def test_draw_analytics_skips_low_conf_person(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person_low_conf = EntityObservation(
            entity_id=1,
            raw_track_id=1,
            label="person",
            class_id=0,
            bbox=(10.0, 10.0, 50.0, 100.0),
            confidence=0.2,
            frame_idx=0,
            timestamp_ms=0.0,
            observed=True,
            reliable=False,
        )
        bag = EntityObservation(
            entity_id=2,
            raw_track_id=2,
            label="bag",
            class_id=1,
            bbox=(60.0, 60.0, 90.0, 90.0),
            confidence=0.8,
            frame_idx=0,
            timestamp_ms=0.0,
            observed=True,
            reliable=True,
        )
        result = AnalyticsResult(
            frame_idx=0,
            timestamp_ms=0.0,
            entities=[person_low_conf, bag],
            motions={},
            relations=[],
            events=[],
            roles={1: RoleAssignment(1, PersonRole.NORMAL, 0.2)},
        )

        with patch("cv2.rectangle") as mock_rect:
            Visualizer.draw_analytics(frame, result, draw_low_conf_person=False)
            # Bag is reliable so its box and label are drawn, but low-conf person should be skipped.
            # Let's inspect calls to cv2.rectangle:
            drawn_coords = [call.args[1] for call in mock_rect.call_args_list if len(call.args) >= 2]
            # (10, 10) corresponds to the person bbox top-left
            self.assertNotIn((10, 10), drawn_coords)
            # (60, 60) corresponds to the bag bbox top-left
            self.assertIn((60, 60), drawn_coords)

    def test_draw_analytics_draws_low_conf_person_when_enabled(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person_low_conf = EntityObservation(
            entity_id=1,
            raw_track_id=1,
            label="person",
            class_id=0,
            bbox=(10.0, 10.0, 50.0, 100.0),
            confidence=0.2,
            frame_idx=0,
            timestamp_ms=0.0,
            observed=True,
            reliable=False,
        )
        result = AnalyticsResult(
            frame_idx=0,
            timestamp_ms=0.0,
            entities=[person_low_conf],
            motions={},
            relations=[],
            events=[],
            roles={1: RoleAssignment(1, PersonRole.NORMAL, 0.2)},
        )

        with patch("cv2.rectangle") as mock_rect:
            Visualizer.draw_analytics(frame, result, draw_low_conf_person=True)
            drawn_coords = [call.args[1] for call in mock_rect.call_args_list if len(call.args) >= 2]
            self.assertIn((10, 10), drawn_coords)

    def test_draw_tracked_skips_low_conf_person(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person_obj = TrackedObject(
            track_id=1,
            bbox=(10.0, 10.0, 50.0, 100.0),
            confidence=0.2,
            class_id=0,
            label="person",
            frame_idx=0,
            timestamp_ms=0.0,
            is_reliable=False,
        )

        with patch("cv2.rectangle") as mock_rect:
            Visualizer.draw_tracked(frame, [person_obj], draw_low_conf_person=False)
            drawn_coords = [call.args[1] for call in mock_rect.call_args_list if len(call.args) >= 2]
            self.assertNotIn((10, 10), drawn_coords)


if __name__ == "__main__":
    unittest.main()
