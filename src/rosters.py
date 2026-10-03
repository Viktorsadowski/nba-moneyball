#!/usr/bin/env python3
"""
Who is on which team right now, from nba.com (CommonTeamRoster, 30 calls).

The contracts page (bbref) lists a bought-out or waived player once per team he's paid by, with his total
salary on every row (KCP: MEM and PHI, both $20.2M, PHI really pays $2.4M). So the contracts alone can't
tell who plays where. This can: a contract row whose player isn't on that team's roster = dead money.

Output: data/raw/rosters.parquet  (team, player_id, name, how_acquired, position, season)

stats.nba.com blocks cloud servers, run it on your own machine:
  python src/rosters.py
"""

import time

import pandas as pd

from config import RAW_DIR, SEASONS, season_label

if __name__ == "__main__":
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams

    season = season_label(max(SEASONS) + 1)      # next season, "2026-27"
    rows = []
    for t in teams.get_teams():
        r = commonteamroster.CommonTeamRoster(team_id=t["id"], season=season, timeout=60).get_data_frames()[0]
        rows.append(pd.DataFrame({"team": t["abbreviation"], "player_id": r["PLAYER_ID"].astype(int),
                                  "name": r["PLAYER"], "how_acquired": r.get("HOW_ACQUIRED", pd.NA),
                                  # listed position (G, G-F, F, F-C, C), for rookies without stats (season_sim)
                                  "position": r.get("POSITION", pd.NA),
                                  "season": max(SEASONS) + 1}))
        time.sleep(1.5)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RAW_DIR / "rosters.parquet", index=False)
    print(f"{season}: {len(out)} players on 30 rosters ({out.groupby('team').size().min()}-"
          f"{out.groupby('team').size().max()} per team)")
    print(out[out["team"] == "PHI"][["name", "how_acquired"]].to_string(index=False))
