"""Các thao tác lưu trữ cho kết quả analytics cướp giật có thể giải thích."""

import json
import sqlite3

from src.core.analytics.types import AnalyticsResult


class AnalyticsRepository:
    """Read and write analytics results for a video."""

    def __init__(self, connection: sqlite3.Connection):
        self._con = connection

    def insert_analytics(
        self,
        video_id: int,
        result: AnalyticsResult,
        rules_version: str,
    ) -> None:
        entity_rows = [
            (
                video_id,
                result.frame_idx,
                item.entity_id,
                item.raw_track_id,
                item.label,
                *item.bbox,
                item.confidence,
                int(item.observed),
                int(item.reliable),
                item.identity_confidence,
                result.timestamp_ms,
            )
            for item in result.entities
        ]
        if entity_rows:
            self._con.executemany(
                """INSERT INTO entity_observations
                   (video_id, frame_idx, entity_id, raw_track_id, label,
                    x1, y1, x2, y2, confidence, observed, reliable,
                    identity_confidence, timestamp_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                entity_rows,
            )

        relation_rows = [
            (
                video_id,
                result.frame_idx,
                item.bag_id,
                item.holder_person_id,
                item.baseline_holder_id,
                item.previous_holder_id,
                item.candidate_holder_id,
                item.state.value,
                item.confidence,
                json.dumps(
                    {**item.evidence, "candidate_scores": item.candidate_scores},
                    sort_keys=True,
                ),
                result.timestamp_ms,
            )
            for item in result.relations
        ]
        if relation_rows:
            self._con.executemany(
                """INSERT INTO bag_person_relations
                   (video_id, frame_idx, bag_entity_id, holder_person_id,
                    baseline_holder_id, previous_holder_id, candidate_holder_id,
                    state, confidence, evidence_json, timestamp_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                relation_rows,
            )

        for event in result.events:
            self._con.execute(
                """INSERT INTO snatch_events
                   (video_id, engine_event_id, bag_entity_id, victim_person_id,
                    suspect_person_id, state, score, confidence, start_ms,
                    updated_ms, end_ms, first_frame, last_frame, evidence_json,
                    rules_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(video_id, engine_event_id) DO UPDATE SET
                    state=excluded.state,
                    score=excluded.score,
                    confidence=excluded.confidence,
                    updated_ms=excluded.updated_ms,
                    end_ms=excluded.end_ms,
                    last_frame=excluded.last_frame,
                    evidence_json=excluded.evidence_json""",
                (
                    video_id,
                    event.event_id,
                    event.bag_id,
                    event.victim_person_id,
                    event.suspect_person_id,
                    event.state.value,
                    event.score,
                    event.confidence,
                    event.start_ms,
                    event.updated_ms,
                    event.end_ms,
                    result.frame_idx,
                    result.frame_idx,
                    json.dumps(event.evidence, sort_keys=True),
                    rules_version,
                ),
            )

        role_rows = [
            (
                video_id,
                result.frame_idx,
                role.person_id,
                role.role.value,
                role.confidence,
                role.event_id,
                result.timestamp_ms,
            )
            for role in result.roles.values()
        ]
        if role_rows:
            self._con.executemany(
                """INSERT INTO person_roles
                   (video_id, frame_idx, person_id, role, confidence, event_id,
                    timestamp_ms)
                   VALUES (?,?,?,?,?,?,?)""",
                role_rows,
            )
        self._con.commit()

    def get_entity_observations(self, video_id: int) -> list[dict]:
        return self._fetch_rows(
            """SELECT * FROM entity_observations
               WHERE video_id=? ORDER BY frame_idx, entity_id""",
            (video_id,),
        )

    def get_bag_person_relations(self, video_id: int) -> list[dict]:
        rows = self._fetch_rows(
            """SELECT * FROM bag_person_relations
               WHERE video_id=? ORDER BY frame_idx, bag_entity_id""",
            (video_id,),
        )
        for row in rows:
            row["evidence"] = json.loads(row.pop("evidence_json"))
        return rows

    def get_snatch_events(
        self,
        video_id: int,
        state: str | None = None,
    ) -> list[dict]:
        if state is None:
            rows = self._fetch_rows(
                """SELECT * FROM snatch_events
                   WHERE video_id=? ORDER BY start_ms, engine_event_id""",
                (video_id,),
            )
        else:
            rows = self._fetch_rows(
                """SELECT * FROM snatch_events
                   WHERE video_id=? AND state=?
                   ORDER BY start_ms, engine_event_id""",
                (video_id, state),
            )
        for row in rows:
            row["evidence"] = json.loads(row.pop("evidence_json"))
        return rows

    def get_person_roles(self, video_id: int) -> list[dict]:
        return self._fetch_rows(
            """SELECT * FROM person_roles
               WHERE video_id=? ORDER BY frame_idx, person_id""",
            (video_id,),
        )

    def _fetch_rows(self, query: str, parameters: tuple[object, ...]) -> list[dict]:
        self._con.row_factory = sqlite3.Row
        try:
            rows = self._con.execute(query, parameters).fetchall()
            return [dict(row) for row in rows]
        finally:
            self._con.row_factory = None
