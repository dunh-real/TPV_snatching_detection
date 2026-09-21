"""Server-owned paths used by the analysis API."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).resolve()


INPUT_ROOT = _path_from_env("TROMCAP_INPUT_ROOT", PROJECT_ROOT / "data" / "videos")
MODEL_PATH = _path_from_env(
    "TROMCAP_MODEL_PATH", PROJECT_ROOT / "models" / "detection" / "best.pt"
)
DB_PATH = _path_from_env(
    "TROMCAP_DB_PATH", PROJECT_ROOT / "data" / "database" / "detections.db"
)
TRACKER_CONFIG = _path_from_env(
    "TROMCAP_TRACKER_CONFIG", PROJECT_ROOT / "configs" / "custom_tracker.yaml"
)
RULES_CONFIG = _path_from_env(
    "TROMCAP_RULES_CONFIG", PROJECT_ROOT / "configs" / "snatch_rules.yaml"
)
OUTPUT_DIR = _path_from_env(
    "TROMCAP_OUTPUT_DIR", PROJECT_ROOT / "data" / "output_video"
)
