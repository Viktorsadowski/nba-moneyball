#!/usr/bin/env python3
"""
Step 1: get the raw data.

  pbp     v3 play-by-play per season from github.com/shufinskiy/nba_data (no nba_api calls)
  roster  one LeagueGameLog call per season: who played in each game + full names + minutes.
          Used to match sub-in names to ids, and to check our minutes against the official ones.

Saves to data/raw/pbp_{year}.parquet and data/raw/roster_{year}.parquet, skips what already exists.

  python src/ingest.py                    # all seasons
  python src/ingest.py --seasons 2010 2025
  python src/ingest.py --no-rosters       # pbp only, no nba_api
"""

import argparse
import io
import tarfile
import time
import urllib.request

import pandas as pd

from config import RAW_DIR, SEASONS, season_label

PBP_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/nbastatsv3_{year}.tar.xz"


# ── play-by-play ──────────────────────────────────────────────────────────────

def download_pbp(year: int) -> None:
    out = RAW_DIR / f"pbp_{year}.parquet"
    if out.exists():
        print(f"  pbp {season_label(year)}: already there")
        return

    with urllib.request.urlopen(PBP_URL.format(year=year), timeout=120) as r:
        blob = r.read()

    # archive holds a single csv, read it straight out without unpacking to disk
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
        df = pd.read_csv(tf.extractfile(member), dtype={"gameId": str}, low_memory=False)

    # csv drops the leading zeros, nba_api uses the 10 char version
    df["gameId"] = df["gameId"].str.zfill(10)
    df.to_parquet(out, index=False)
    print(f"  pbp {season_label(year)}: {df['gameId'].nunique()} games, {len(df):,} rows")


# ── rosters from the player game log ──────────────────────────────────────────

def fetch_roster(year: int) -> None:
    out = RAW_DIR / f"roster_{year}.parquet"
    if out.exists():
        print(f"  roster {season_label(year)}: already there")
        return

    from nba_api.stats.endpoints import leaguegamelog  # only needed here

    df = leaguegamelog.LeagueGameLog(
        season=season_label(year),
        player_or_team_abbreviation="P",   # P = one row per player per game
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]

    df = df[["GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "MIN"]]
    df.to_parquet(out, index=False)
    print(f"  roster {season_label(year)}: {df['PLAYER_ID'].nunique()} players, {len(df):,} player-games")
    time.sleep(1.5)  # be nice to stats.nba.com


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    ap.add_argument("--no-rosters", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for y in args.seasons:
        print(season_label(y))
        download_pbp(y)
        if not args.no_rosters:
            fetch_roster(y)
