#!/usr/bin/env python3
"""
Step 5a: player info per season from stats.nba.com, two calls per season.

  players_{year}.parquet     age + box score totals per player (LeagueDashPlayerStats)
                             age is for the aging curve, box score for the box-prior RAPM later
  team_games_{year}.parquet  every team's schedule with dates (LeagueGameLog, team level)
                             needed to count games missed during an injury

  python src/players.py
  python src/players.py --seasons 2025
"""

import argparse
import time

import pandas as pd

from config import RAW_DIR, SEASONS, season_label


def fetch_players(year: int) -> None:
    out = RAW_DIR / f"players_{year}.parquet"
    if out.exists():
        print(f"  players {season_label(year)}: already there")
        return
    from nba_api.stats.endpoints import leaguedashplayerstats

    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season_label(year),
        per_mode_detailed="Totals",
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    df = df[[c for c in df.columns if not c.endswith("_RANK") and c not in ("CFID", "CFPARAMS")]]
    df.insert(0, "SEASON", year)
    df.to_parquet(out, index=False)
    print(f"  players {season_label(year)}: {len(df)} players, median age {df['AGE'].median():.0f}")
    time.sleep(1.5)


def fetch_team_games(year: int) -> None:
    out = RAW_DIR / f"team_games_{year}.parquet"
    if out.exists():
        print(f"  team games {season_label(year)}: already there")
        return
    from nba_api.stats.endpoints import leaguegamelog

    df = leaguegamelog.LeagueGameLog(
        season=season_label(year),
        player_or_team_abbreviation="T",
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    df = df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.to_parquet(out, index=False)
    print(f"  team games {season_label(year)}: {df['GAME_ID'].nunique()} games")
    time.sleep(1.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for y in args.seasons:
        print(season_label(y))
        fetch_players(y)
        fetch_team_games(y)
