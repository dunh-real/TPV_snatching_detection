import json
import sqlite3
from pathlib import Path

from src.analytics.models import AnalyticsResult
from src.core.models import TrackedObject


class DetectionDB:
    """Manages SQLite database for videos and detections."""

    def __init__(self, db_path: str = "data/detections.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.db_path)
        self._con.execute("PRAGMA foreign_keys = ON;")
        self._con.execute("PRAGMA journal_mode = WAL;")
        self._init_db()

    # ── Schema ────────────────────────────────────────────────

    def _init_db(self):
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path  TEXT NOT NULL,
                fps          REAL,
                total_frames INTEGER,
                created_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS detections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id     INTEGER NOT NULL,
                frame_idx    INTEGER NOT NULL,
                track_id     INTEGER NOT NULL,
                label        TEXT NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                confidence   REAL,
                timestamp_ms REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            );
            CREATE INDEX IF NOT EXISTS idx_det_video
                ON detections(video_id);
            CREATE INDEX IF NOT EXISTS idx_det_track
                ON detections(video_id, track_id);

            CREATE TABLE IF NOT EXISTS entity_observations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id            INTEGER NOT NULL,
                frame_idx           INTEGER NOT NULL,
                entity_id           INTEGER NOT NULL,
                raw_track_id        INTEGER,
                label               TEXT NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                confidence          REAL NOT NULL,
                observed            INTEGER NOT NULL,
                reliable            INTEGER NOT NULL,
                identity_confidence REAL NOT NULL,
                timestamp_ms        REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            );
            CREATE INDEX IF NOT EXISTS idx_entity_obs_video_frame
                ON entity_observations(video_id, frame_idx);
            CREATE INDEX IF NOT EXISTS idx_entity_obs_entity
                ON entity_observations(video_id, entity_id);

            CREATE TABLE IF NOT EXISTS bag_person_relations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id            INTEGER NOT NULL,
                frame_idx           INTEGER NOT NULL,
                bag_entity_id       INTEGER NOT NULL,
                holder_person_id    INTEGER,
                baseline_holder_id  INTEGER,
                previous_holder_id  INTEGER,
                candidate_holder_id INTEGER,
                state               TEXT NOT NULL,
                confidence          REAL NOT NULL,
                evidence_json       TEXT NOT NULL,
                timestamp_ms        REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            );
            CREATE INDEX IF NOT EXISTS idx_relation_video_bag
                ON bag_person_relations(video_id, bag_entity_id, frame_idx);

            CREATE TABLE IF NOT EXISTS snatch_events (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id            INTEGER NOT NULL,
                engine_event_id     INTEGER NOT NULL,
                bag_entity_id       INTEGER NOT NULL,
                victim_person_id    INTEGER NOT NULL,
                suspect_person_id   INTEGER NOT NULL,
                state               TEXT NOT NULL,
                score               REAL NOT NULL,
                confidence          REAL NOT NULL,
                start_ms            REAL NOT NULL,
                updated_ms          REAL NOT NULL,
                end_ms              REAL,
                first_frame         INTEGER NOT NULL,
                last_frame          INTEGER NOT NULL,
                evidence_json       TEXT NOT NULL,
                rules_version       TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                UNIQUE(video_id, engine_event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_event_video_state
                ON snatch_events(video_id, state);

            CREATE TABLE IF NOT EXISTS person_roles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id     INTEGER NOT NULL,
                frame_idx    INTEGER NOT NULL,
                person_id    INTEGER NOT NULL,
                role         TEXT NOT NULL,
                confidence   REAL NOT NULL,
                event_id     INTEGER,
                timestamp_ms REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            );
            CREATE INDEX IF NOT EXISTS idx_role_video_person
                ON person_roles(video_id, person_id, frame_idx);
        """)

    # ── Write ─────────────────────────────────────────────────

    def create_video(self, source_path: str, fps: float, total_frames: int) -> int:
        """Register a new video, return its video_id."""
        cur = self._con.execute(
            "INSERT INTO videos (source_path, fps, total_frames) VALUES (?,?,?)",
            (source_path, fps, total_frames),
        )
        self._con.commit()
        return cur.lastrowid

    def insert_detections(
        self,
        video_id: int,
        objects: list[TrackedObject],
        *,
        commit: bool = True,
    ) -> None:
        """Batch-insert tracked detections for one frame."""
        if not objects:
            return
        rows = [
            (video_id, o.frame_idx, o.track_id, o.label,
             *o.bbox, o.confidence, o.timestamp_ms)
            for o in objects
        ]
        self._con.executemany(
            """INSERT INTO detections
               (video_id, frame_idx, track_id, label,
                x1, y1, x2, y2, confidence, timestamp_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        if commit:
            self._con.commit()

    def insert_analytics(
        self,
        video_id: int,
        result: AnalyticsResult,
        rules_version: str,
    ) -> None:
        """Persist one frame of explainable analytics output."""
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

    # ── Read ──────────────────────────────────────────────────

    def get_detections(self, video_id: int) -> list[dict]:
        """Return all detections for a video, ordered by frame."""
        self._con.row_factory = sqlite3.Row
        rows = self._con.execute(
            "SELECT * FROM detections WHERE video_id=? ORDER BY frame_idx, track_id",
            (video_id,),
        ).fetchall()
        self._con.row_factory = None
        return [dict(r) for r in rows]

    def get_videos(self) -> list[dict]:
        """Return all registered videos."""
        self._con.row_factory = sqlite3.Row
        rows = self._con.execute(
            "SELECT * FROM videos ORDER BY created_at DESC"
        ).fetchall()
        self._con.row_factory = None
        return [dict(r) for r in rows]

    def get_entity_observations(self, video_id: int) -> list[dict]:
        """Return canonical entity observations ordered by frame."""
        return self._fetch_rows(
            """SELECT * FROM entity_observations
               WHERE video_id=? ORDER BY frame_idx, entity_id""",
            (video_id,),
        )

    def get_bag_person_relations(self, video_id: int) -> list[dict]:
        """Return holder relation history with decoded evidence."""
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
        """Return rule-engine events, optionally filtered by final state."""
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
        """Return per-frame person role assignments."""
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

    def close(self):
        """Close the database connection."""
        self._con.close()
