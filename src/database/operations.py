"""Facade tương thích ngược cho tầng lưu trữ SQLite của dự án."""

from src.core.analytics.types import AnalyticsResult
from src.core.types import TrackedObject
from src.database.analytics_repository import AnalyticsRepository
from src.database.connection import open_connection
from src.database.detection_repository import DetectionRepository
from src.database.schema import initialize_schema


class DetectionDB:
    """Keep the original database API while delegating by responsibility."""

    def __init__(self, db_path: str = "data/detections.db"):
        self.db_path = db_path
        self._con = open_connection(db_path)
        self._init_db()
        self._detections = DetectionRepository(self._con)
        self._analytics = AnalyticsRepository(self._con)

    def _init_db(self) -> None:
        initialize_schema(self._con)

    def create_video(self, source_path: str, fps: float, total_frames: int) -> int:
        """Register a new video, return its video_id."""
        return self._detections.create_video(source_path, fps, total_frames)

    def insert_detections(
        self,
        video_id: int,
        objects: list[TrackedObject],
        *,
        commit: bool = True,
    ) -> None:
        """Batch-insert tracked detections for one frame."""
        self._detections.insert_detections(video_id, objects, commit=commit)

    def insert_analytics(
        self,
        video_id: int,
        result: AnalyticsResult,
        rules_version: str,
        *,
        commit: bool = True,
    ) -> None:
        """Persist one frame of explainable analytics output."""
        self._analytics.insert_analytics(
            video_id, result, rules_version, commit=commit
        )

    def commit(self) -> None:
        """Commit the current frame batch as one SQLite transaction."""
        self._con.commit()

    def get_detections(self, video_id: int) -> list[dict]:
        """Return all detections for a video, ordered by frame."""
        return self._detections.get_detections(video_id)

    def get_videos(self) -> list[dict]:
        """Return all registered videos."""
        return self._detections.get_videos()

    def get_entity_observations(self, video_id: int) -> list[dict]:
        """Return canonical entity observations ordered by frame."""
        return self._analytics.get_entity_observations(video_id)

    def get_bag_person_relations(self, video_id: int) -> list[dict]:
        """Return holder relation history with decoded evidence."""
        return self._analytics.get_bag_person_relations(video_id)

    def get_snatch_events(
        self,
        video_id: int,
        state: str | None = None,
    ) -> list[dict]:
        """Return rule-engine events, optionally filtered by final state."""
        return self._analytics.get_snatch_events(video_id, state)

    def get_person_roles(self, video_id: int) -> list[dict]:
        """Return per-frame person role assignments."""
        return self._analytics.get_person_roles(video_id)

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()
