"""Factory tạo kết nối SQLite cho các repository lưu trữ."""

import sqlite3
from pathlib import Path


def open_connection(db_path: str) -> sqlite3.Connection:
    """Open the project SQLite database with its existing connection settings."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA journal_mode = WAL;")
    return connection
