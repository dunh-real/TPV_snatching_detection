import unittest

from src.core.analytics.config import IdentityConfig, LabelsConfig, MotionConfig
from src.core.analytics.identity import EntityIdentityManager
from src.core.analytics.trajectory import TrajectoryManager
from tests.helpers import entity, tracked


class TrajectoryTests(unittest.TestCase):
    def test_velocity_uses_elapsed_time_and_normalized_coordinates(self) -> None:
        manager = TrajectoryManager(
            MotionConfig(ema_alpha=1.0, velocity_alpha=1.0, acceleration_alpha=1.0)
        )
        manager.update(
            [entity(1, "person", (0.0, 0.0, 100.0, 200.0), 0.0)],
            (1000, 1000, 3),
        )
        result = manager.update(
            [entity(1, "person", (100.0, 0.0, 200.0, 200.0), 1000.0)],
            (1000, 1000, 3),
        )
        self.assertAlmostEqual(result[1].vx, 0.1)
        self.assertAlmostEqual(result[1].vy, 0.0)

    def test_inferred_observation_reduces_quality(self) -> None:
        manager = TrajectoryManager(MotionConfig())
        first = manager.update(
            [entity(1, "person", (0.0, 0.0, 100.0, 200.0), 0.0)],
            (1000, 1000, 3),
        )[1]
        inferred = manager.update(
            [entity(1, "person", (0.0, 0.0, 100.0, 200.0), 100.0, observed=False)],
            (1000, 1000, 3),
        )[1]
        self.assertLess(inferred.quality, first.quality)


class IdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = EntityIdentityManager(IdentityConfig(), LabelsConfig())
        self.shape = (1000, 1000, 3)

    def test_new_raw_bag_id_reattaches_to_canonical_bag(self) -> None:
        first = self.manager.update(
            [tracked(10, "bag", (100.0, 100.0, 150.0, 160.0), 0.0)],
            0,
            0.0,
            self.shape,
        )
        bag_id = first[0].entity_id
        missing = self.manager.update([], 1, 100.0, self.shape)
        self.assertEqual(missing[0].entity_id, bag_id)
        self.assertFalse(missing[0].observed)

        reappeared = self.manager.update(
            [tracked(11, "bag", (103.0, 100.0, 153.0, 160.0), 200.0, frame_idx=2)],
            2,
            200.0,
            self.shape,
        )
        self.assertEqual(reappeared[0].entity_id, bag_id)
        self.assertEqual(reappeared[0].raw_track_id, 11)

    def test_expired_track_gets_a_new_entity_id(self) -> None:
        first = self.manager.update(
            [tracked(10, "bag", (100.0, 100.0, 150.0, 160.0), 0.0)],
            0,
            0.0,
            self.shape,
        )[0]
        second = self.manager.update(
            [tracked(11, "bag", (100.0, 100.0, 150.0, 160.0), 3000.0, frame_idx=30)],
            30,
            3000.0,
            self.shape,
        )[0]
        self.assertNotEqual(first.entity_id, second.entity_id)


if __name__ == "__main__":
    unittest.main()
