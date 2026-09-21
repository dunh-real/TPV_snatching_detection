"""Schema SQLite cho đầu ra phát hiện và analytics cướp giật."""

import sqlite3


SCHEMA_SQL = """
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
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the existing tables and indexes if they are missing."""
    connection.executescript(SCHEMA_SQL)
