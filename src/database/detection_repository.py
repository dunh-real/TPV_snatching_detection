"""Các thao tác lưu trữ cho video và kết quả phát hiện đã bám vết."""

import sqlite3

from src.core.types import TrackedObject


class DetectionRepository:
    """Read and write video metadata and tracked detections."""

    def __init__(self, connection: sqlite3.Connection):
        self._con = connection

    def create_video(self, source_path: str, fps: float, total_frames: int) -> int:
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

    def get_detections(self, video_id: int) -> list[dict]:
        self._con.row_factory = sqlite3.Row
        rows = self._con.execute(
            "SELECT * FROM detections WHERE video_id=? ORDER BY frame_idx, track_id",
            (video_id,),
        ).fetchall()
        self._con.row_factory = None
        return [dict(row) for row in rows]

    def get_videos(self) -> list[dict]:
        self._con.row_factory = sqlite3.Row
        rows = self._con.execute(
            "SELECT * FROM videos ORDER BY created_at DESC"
        ).fetchall()
        self._con.row_factory = None
        return [dict(row) for row in rows]
