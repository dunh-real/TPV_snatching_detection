import sqlite3
from pathlib import Path

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

    def insert_detections(self, video_id: int, objects: list[TrackedObject]):
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

    def close(self):
        """Close the database connection."""
        self._con.close()
