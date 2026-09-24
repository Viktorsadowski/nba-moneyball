#!/usr/bin/env python3
"""
Freeze the projections for next season, before it starts. The forward test (forward_test.py) checks them
against what actually happens, and that only counts if nobody could have touched them after tip-off.

Writes forecasts/{season}/ (in the repo, not in data/, so git tracks it):

  players.csv   every player with a projection: value O/D (raw + calibrated), expected games, minutes,
                WAR, WAR if healthy, team (from the current contracts page), salary
  teams.csv     projected net rating per 100 and wins per team: each roster's projected minutes
                (scaled to the 5 x 48 x 82 minutes a team has, leftovers played at replacement level,
                rookies at what first-year players usually post), same calibration as war.py
  meta.json     when, which git commit, model settings, sha256 of both csv's

Commit + push right after, the commit time is the proof it came before the season.
Won't overwrite an existing freeze (--force if you really mean it).

  python src/freeze.py
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ROOT, SEASONS, season_label
from surplus import contracts_with_ids

TEAM_MIN = 5 * 48 * 82      # player-minutes a team has to fill in a season
GAMES = 82


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    as_of = max(SEASONS)
    target = as_of + 1
    out = ROOT / "forecasts" / season_label(target)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists, that's the frozen one. --force to overwrite (and say why)")
    out.mkdir(parents=True, exist_ok=True)

    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    war = war[war["season"] == as_of].copy()
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    a, c, repl, ppw = prm["a"], prm["c"], prm["repl"], prm["ppw"]
    pace = prm["pace"][str(as_of)]           # possessions per player-minute

    # rookies: same number war.py uses, what first-year players posted (raw value units)
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    first = rapm.sort_values("season").drop_duplicates("player_id")
    first = first[first["season"] > min(SEASONS)]
    rookie = np.average(first["rapm"], weights=first["minutes"])
    repl_raw = (repl - c) / a                # replacement level back in raw units

    # who plays where next season: current contracts
    con = contracts_with_ids()
    con = con[con["season"] == target].sort_values("salary").drop_duplicates("player_id", keep="last")
    con = con[["player_id", "team", "salary"]]

    players = war.merge(con, on="player_id", how="left")
    cols = ["player_id", "name", "team", "age_next", "o_val", "d_val", "val", "val_cal", "exp_games", "exp_inj_games",
            "proj_min", "war", "war_healthy", "salary"]
    players = players[cols].sort_values("war", ascending=False)
    players.to_csv(out / "players.csv", index=False, float_format="%.4f")

    # ── teams ──
    roster = con.merge(war[["player_id", "val", "proj_min"]], on="player_id", how="left")
    roster["val"] = roster["val"].fillna(rookie)
    # rookies / no minutes history: a bench role until they show otherwise
    roster["proj_min"] = roster["proj_min"].fillna(800)
    rows = []
    for team, r in roster.groupby("team"):
        mins = r["proj_min"].sum()
        scale = min(1.0, TEAM_MIN / mins)
        used = r["proj_min"] * scale
        filler = TEAM_MIN - used.sum()
        poss = (used.sum() + filler) * pace
        # same form as the calibration in war.py: diff = a * sum(val * poss / 100) + c * poss / 100
        net_raw = (r["val"] * used).sum() * pace / 100 + repl_raw * filler * pace / 100
        diff = a * net_raw + c * poss / 100
        rows.append(dict(team=team, players=len(r), roster_minutes_share=used.sum() / TEAM_MIN, diff=diff))
    teams = pd.DataFrame(rows)
    # the league has to add up to .500: center the differentials
    teams["diff"] -= teams["diff"].mean()
    teams["net_per100"] = teams["diff"] / (TEAM_MIN / 5 * pace) * 100
    teams["proj_wins"] = GAMES / 2 + teams["diff"] / ppw
    teams = teams.sort_values("proj_wins", ascending=False)
    teams[["team", "proj_wins", "net_per100", "players", "roster_minutes_share"]].to_csv(
        out / "teams.csv", index=False, float_format="%.3f")

    meta = {
        "season": season_label(target),
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": season_label(as_of),
        "git_commit_at_freeze": git_commit(),
        "model": {"rapm": "ridge with box-score prior (box_prior.py)", "value": "50/30/20 x minutes, 250 ghost min, aged",
                  "calibration_a": a, "calibration_c": c, "replacement": repl, "points_per_win": ppw,
                  "rookie_raw_value": rookie},
        "sha256": {"players.csv": sha256(out / "players.csv"), "teams.csv": sha256(out / "teams.csv")},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"frozen {season_label(target)} -> {out}")
    print(f"  {len(players)} players, {players['team'].notna().sum()} with a {season_label(target)} contract")
    print("\nprojected standings:")
    print(teams[["team", "proj_wins", "net_per100", "players", "roster_minutes_share"]].round(1).to_string(index=False))
    print("\nnow commit + push, the commit time is the timestamp:")
    print(f'  git add forecasts && git commit -m "freeze {season_label(target)} forecast"')
