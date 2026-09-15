import unittest

from src.core.analytics.config import SnatchConfig
from src.core.analytics.types import EventState, PersonRole, RelationState
from src.core.analytics.roles import RoleResolver
from src.core.analytics.snatch_detector import SnatchDetector
from tests.helpers import entity, motion, relation


class SnatchDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = SnatchDetector(SnatchConfig())

    @staticmethod
    def _observations(timestamp_ms: float, suspect_x: float, bag_x: float):
        return [
            entity(1, "person", (100.0, 100.0, 300.0, 500.0), timestamp_ms),
            entity(2, "person", (suspect_x, 100.0, suspect_x + 200.0, 500.0), timestamp_ms),
            entity(3, "bag", (bag_x, 300.0, bag_x + 50.0, 380.0), timestamp_ms),
        ]

    @staticmethod
    def _motions(moving: bool = False):
        speed = 0.12 if moving else 0.0
        return {
            1: motion(1, vx=0.03 if moving else 0.0),
            2: motion(2, vx=speed, ax=0.08 if moving else 0.0),
            3: motion(3, vx=speed),
        }

    def _step(self, timestamp_ms, suspect_x, bag_x, relation_value, moving=False):
        return self.detector.update(
            self._observations(timestamp_ms, suspect_x, bag_x),
            self._motions(moving),
            [relation_value],
            timestamp_ms,
            "person",
        )

    def test_transfer_and_escape_produce_suspected_event(self) -> None:
        owner_relation = relation(1)
        self._step(0.0, 500.0, 220.0, owner_relation)
        self._step(200.0, 430.0, 220.0, owner_relation)
        self._step(400.0, 260.0, 220.0, owner_relation)
        pending = relation(1, candidate=2)
        self._step(500.0, 260.0, 220.0, pending)

        switched = relation(
            2,
            previous=1,
            holder_duration=0.0,
            previous_duration=2.0,
        )
        self._step(800.0, 300.0, 330.0, switched, moving=True)
        self._step(900.0, 360.0, 390.0, switched, moving=True)
        self._step(1300.0, 500.0, 530.0, switched, moving=True)
        events = self._step(1700.0, 600.0, 630.0, switched, moving=True)

        suspected = [event for event in events if event.state == EventState.SUSPECTED]
        self.assertEqual(len(suspected), 1)
        self.assertEqual(suspected[0].victim_person_id, 1)
        self.assertEqual(suspected[0].suspect_person_id, 2)

        roles = RoleResolver().update(
            self._observations(1700.0, 600.0, 630.0), suspected, "person"
        )
        self.assertEqual(roles[1].role, PersonRole.VICTIM)
        self.assertEqual(roles[2].role, PersonRole.SUSPECT)

    def test_contact_without_transfer_is_cancelled(self) -> None:
        owner_relation = relation(1)
        self._step(0.0, 500.0, 220.0, owner_relation)
        self._step(200.0, 400.0, 220.0, owner_relation)
        self._step(400.0, 250.0, 220.0, owner_relation)
        events = self._step(2100.0, 250.0, 220.0, owner_relation)
        self.assertFalse(any(event.state == EventState.SUSPECTED for event in events))
        self.assertTrue(any(event.state == EventState.CANCELLED for event in events))


if __name__ == "__main__":
    unittest.main()
