import tempfile
import unittest
from pathlib import Path

from src.analytics.config import load_rule_engine_config
from src.analytics.geometry import (
    bbox_from_relative_offset,
    normalized_point_to_person_distance,
    relative_bbox_offset,
)


class ConfigGeometryTests(unittest.TestCase):
    def test_default_yaml_loads(self) -> None:
        config = load_rule_engine_config("configs/snatch_rules.yaml")
        self.assertEqual(config.labels.person, "person")
        self.assertGreater(config.snatch.suspected_score_threshold, 0.0)

    def test_unknown_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.yaml"
            path.write_text("motion:\n  typo_alpha: 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "typo_alpha"):
                load_rule_engine_config(path)

    def test_relative_bbox_round_trip(self) -> None:
        person = (100.0, 100.0, 300.0, 500.0)
        bag = (250.0, 300.0, 310.0, 390.0)
        offset = relative_bbox_offset(bag, person)
        reconstructed = bbox_from_relative_offset(offset, person)
        for actual, expected in zip(reconstructed, bag):
            self.assertAlmostEqual(actual, expected)

    def test_point_inside_person_has_zero_distance(self) -> None:
        self.assertEqual(
            normalized_point_to_person_distance((150.0, 200.0), (100.0, 100.0, 300.0, 500.0)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
