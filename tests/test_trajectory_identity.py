import unittest

from dataclasses import replace

from src.core.analytics.config import IdentityConfig, LabelsConfig, MotionConfig
from src.core.analytics.identity import EntityIdentityManager
from src.core.analytics.types import RelationState
from src.core.analytics.trajectory import TrajectoryManager
from tests.helpers import entity, relation, tracked


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

    def test_ema_response_is_independent_of_frame_interval(self) -> None:
        config = MotionConfig()
        fast = TrajectoryManager(config)
        slow = TrajectoryManager(config)
        start = entity(1, "person", (0.0, 0.0, 100.0, 200.0), 0.0)
        fast.update([start], (1000, 1000, 3))
        slow.update([start], (1000, 1000, 3))

        fast.update(
            [entity(1, "person", (100.0, 0.0, 200.0, 200.0), 1000 / 30)],
            (1000, 1000, 3),
        )
        fast_result = fast.update(
            [entity(1, "person", (100.0, 0.0, 200.0, 200.0), 2000 / 30)],
            (1000, 1000, 3),
        )[1]
        slow_result = slow.update(
            [entity(1, "person", (100.0, 0.0, 200.0, 200.0), 2000 / 30)],
            (1000, 1000, 3),
        )[1]
        self.assertAlmostEqual(fast_result.x, slow_result.x)


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

    def test_reattach_removes_the_previous_raw_track_mapping(self) -> None:
        original = self.manager.update(
            [tracked(10, "bag", (100.0, 100.0, 150.0, 160.0), 0.0)],
            0, 0.0, self.shape,
        )[0]
        self.manager.update(
            [tracked(11, "bag", (102.0, 100.0, 152.0, 160.0), 100.0)],
            1, 100.0, self.shape,
        )
        observations = self.manager.update(
            [tracked(10, "bag", (800.0, 800.0, 850.0, 860.0), 200.0)],
            2, 200.0, self.shape,
        )
        reused = next(item for item in observations if item.raw_track_id == 10)
        self.assertNotEqual(reused.entity_id, original.entity_id)

    def test_detached_bag_drops_its_holder_anchor(self) -> None:
        observations = self.manager.update(
            [
                tracked(1, "person", (100.0, 100.0, 300.0, 500.0), 0.0),
                tracked(2, "bag", (150.0, 250.0, 200.0, 320.0), 0.0),
            ],
            0, 0.0, self.shape,
        )
        person_id = next(item.entity_id for item in observations if item.label == "person")
        bag_id = next(item.entity_id for item in observations if item.label == "bag")
        attached = relation(person_id, bag_id=bag_id)
        self.manager.update_bag_anchors([attached], observations)
        self.assertIn(bag_id, self.manager._bag_anchors)

        detached = replace(
            attached, holder_person_id=None, state=RelationState.DETACHED
        )
        self.manager.update_bag_anchors([detached], observations)
        self.assertNotIn(bag_id, self.manager._bag_anchors)


if __name__ == "__main__":
    unittest.main()
