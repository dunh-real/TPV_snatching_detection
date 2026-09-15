import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.core.analytics.types import (
    AnalyticsResult,
    BagPersonRelation,
    EventState,
    PersonRole,
    RelationState,
    RoleAssignment,
    SnatchEvent,
)
from src.database.operations import DetectionDB
from tests.helpers import entity, motion


class AnalyticsDatabaseTests(unittest.TestCase):
    def test_analytics_result_is_persisted_and_event_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "analytics.db"
            database = DetectionDB(str(db_path))
            video_id = database.create_video("input.mp4", 10.0, 20)
            result = self._result(EventState.ESCAPING, frame_idx=3)
            database.insert_analytics(video_id, result, "test-rules")
            database.insert_analytics(
                video_id,
                self._result(EventState.SUSPECTED, frame_idx=4),
                "test-rules",
            )
            events = database.get_snatch_events(video_id, state="suspected")
            relations = database.get_bag_person_relations(video_id)
            self.assertEqual(events[0]["evidence"], {"holder_switch": True})
            self.assertIn("candidate_scores", relations[0]["evidence"])
            database.close()

            connection = sqlite3.connect(db_path)
            try:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM entity_observations").fetchone()[0],
                    6,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM bag_person_relations").fetchone()[0],
                    2,
                )
                event = connection.execute(
                    "SELECT state, first_frame, last_frame FROM snatch_events"
                ).fetchone()
                self.assertEqual(event, ("suspected", 3, 4))
            finally:
                connection.close()

    @staticmethod
    def _result(state: EventState, frame_idx: int) -> AnalyticsResult:
        timestamp_ms = frame_idx * 100.0
        entities = [
            entity(1, "person", (100.0, 100.0, 300.0, 500.0), timestamp_ms),
            entity(2, "person", (500.0, 100.0, 700.0, 500.0), timestamp_ms),
            entity(3, "bag", (550.0, 300.0, 600.0, 380.0), timestamp_ms),
        ]
        return AnalyticsResult(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            entities=entities,
            motions={item.entity_id: motion(item.entity_id) for item in entities},
            relations=[
                BagPersonRelation(
                    bag_id=3,
                    holder_person_id=2,
                    baseline_holder_id=1,
                    previous_holder_id=1,
                    candidate_holder_id=None,
                    state=RelationState.ATTACHED,
                    confidence=0.9,
                    holder_duration_seconds=0.5,
                    previous_holder_duration_seconds=2.0,
                )
            ],
            events=[
                SnatchEvent(
                    event_id=1,
                    bag_id=3,
                    victim_person_id=1,
                    suspect_person_id=2,
                    state=state,
                    score=8.0,
                    confidence=0.9,
                    start_ms=100.0,
                    updated_ms=timestamp_ms,
                    end_ms=timestamp_ms if state == EventState.SUSPECTED else None,
                    evidence={"holder_switch": True},
                )
            ],
            roles={
                1: RoleAssignment(1, PersonRole.VICTIM, 0.9, 1),
                2: RoleAssignment(2, PersonRole.SUSPECT, 0.9, 1),
            },
        )


if __name__ == "__main__":
    unittest.main()
