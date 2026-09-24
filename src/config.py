"""
Paths and season range, shared by all scripts.
Set NBA_DATA_DIR as an env var to keep data outside the project (e.g. outside OneDrive).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# start year of the season, 2010 = 2010-11, 2025 = 2025-26
SEASONS = range(2010, 2026)


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"
