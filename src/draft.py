#!/usr/bin/env python3
"""
Draft history for the rookie model: every pick from 2010 on (incl. the 2026 draft), one nba_api call.

Output: data/raw/draft.parquet  (draft_year, round, pick = overall pick, player_id, name, team, organization)

stats.nba.com blocks cloud servers, run it on your own machine:
  python src/draft.py
"""

import pandas as pd

from config import RAW_DIR

FIRST_YEAR = 2010


if __name__ == "__main__":
    from nba_api.stats.endpoints import drafthistory

    d = drafthistory.DraftHistory(league_id="00", timeout=60).get_data_frames()[0]
    d = d[d["SEASON"].astype(int) >= FIRST_YEAR]
    out = pd.DataFrame({
        "draft_year": d["SEASON"].astype(int),
        "round": d["ROUND_NUMBER"].astype(int),
        "pick": d["OVERALL_PICK"].astype(int),
        "player_id": d["PERSON_ID"].astype(int),
        "name": d["PLAYER_NAME"],
        "team": d["TEAM_ABBREVIATION"],
        "organization": d["ORGANIZATION"],
    })
    out.to_parquet(RAW_DIR / "draft.parquet", index=False)
    print(f"{len(out)} picks, drafts {out['draft_year'].min()}-{out['draft_year'].max()}")
    print(out[out["draft_year"] == out["draft_year"].max()].head(10).to_string(index=False))
