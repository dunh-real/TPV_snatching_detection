import unittest

from src.analytics.config import AssociationConfig, LabelsConfig
from src.analytics.models import RelationState
from src.analytics.person_bag import PersonBagAssociationManager
from tests.helpers import entity, motion


class PersonBagAssociationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = PersonBagAssociationManager(AssociationConfig(), LabelsConfig())

    def _update(self, timestamp_ms: float, bag_x: float, second_person: bool = False):
        observations = [
            entity(1, "person", (100.0, 100.0, 300.0, 500.0), timestamp_ms),
            entity(3, "bag", (bag_x, 300.0, bag_x + 50.0, 380.0), timestamp_ms),
        ]
        if second_person:
            observations.append(
                entity(2, "person", (bag_x - 40.0, 100.0, bag_x + 100.0, 500.0), timestamp_ms)
            )
        motions = {item.entity_id: motion(item.entity_id) for item in observations}
        return self.manager.update(observations, motions, timestamp_ms)[0]

    def test_attachment_requires_persistence_and_sets_baseline(self) -> None:
        first = self._update(0.0, 220.0)
        self.assertEqual(first.state, RelationState.SWITCH_PENDING)
        attached = self._update(400.0, 220.0)
        self.assertEqual(attached.holder_person_id, 1)
        stable = self._update(900.0, 220.0)
        self.assertEqual(stable.baseline_holder_id, 1)

    def test_switch_requires_persistence(self) -> None:
        self._update(0.0, 220.0)
        self._update(400.0, 220.0)
        self._update(900.0, 220.0)

        pending = self._update(1000.0, 700.0, second_person=True)
        self.assertEqual(pending.holder_person_id, 1)
        self.assertEqual(pending.candidate_holder_id, 2)
        switched = self._update(1400.0, 700.0, second_person=True)
        self.assertEqual(switched.holder_person_id, 2)
        self.assertEqual(switched.previous_holder_id, 1)
        self.assertEqual(switched.baseline_holder_id, 1)


if __name__ == "__main__":
    unittest.main()
