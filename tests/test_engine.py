import unittest

from src.core.analytics.config import RuleEngineConfig
from src.core.analytics.types import PersonRole, RelationState
from src.services.snatch_analytics import SnatchAnalyticsEngine
from tests.helpers import tracked


class AnalyticsEngineTests(unittest.TestCase):
    def test_engine_emits_canonical_entities_relation_and_normal_role(self) -> None:
        engine = SnatchAnalyticsEngine(RuleEngineConfig())
        frame_shape = (1000, 1000, 3)
        result = None
        for frame_idx in range(10):
            timestamp_ms = frame_idx * 100.0
            result = engine.update(
                [
                    tracked(
                        10,
                        "person",
                        (100.0, 100.0, 300.0, 500.0),
                        timestamp_ms,
                        frame_idx,
                    ),
                    tracked(
                        20,
                        "bag",
                        (220.0, 300.0, 270.0, 380.0),
                        timestamp_ms,
                        frame_idx,
                    ),
                ],
                frame_idx,
                timestamp_ms,
                frame_shape,
            )

        self.assertIsNotNone(result)
        assert result is not None
        person = next(item for item in result.entities if item.label == "person")
        bag = next(item for item in result.entities if item.label == "bag")
        self.assertEqual(result.relations[0].holder_person_id, person.entity_id)
        self.assertEqual(result.relations[0].bag_id, bag.entity_id)
        self.assertEqual(result.relations[0].state, RelationState.ATTACHED)
        self.assertEqual(result.roles[person.entity_id].role, PersonRole.NORMAL)


if __name__ == "__main__":
    unittest.main()
