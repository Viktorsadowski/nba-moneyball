#!/usr/bin/env python3
"""
Forward test: the frozen forecast (freeze.py) against what actually happened. Runs any time during or
after the season, on whatever games are played so far.

Three checks, each against a simple baseline, so "good" means something:

  teams         projected wins vs actual win pace (wins per 82 so far)
                baseline: last season's wins, pulled a third of the way back to 41
  stints        frozen player values -> points per 100 in every stint played so far
                (same test as box_prior.py), vs last season's plain RAPM and prior RAPM
  availability  projected games vs games actually played (scaled to the team games so far)
                baseline: last season's games played, same scaling

First checks the sha256 of the frozen files against meta.json, so an edited forecast shows up.

Data for the new season (no change to config.py needed). The raw files get skipped if they exist, so for
a mid-season update delete them first:

  Remove-Item data\\raw\\pbp_2026.parquet, data\\raw\\roster_2026.parquet, data\\raw\\players_2026.parquet, data\\raw\\team_games_2026.parquet
  python src/ingest.py --seasons 2026
  python src/players.py --seasons 2026
  python src/lineups.py --seasons 2026
  python src/forward_test.py
"""

import argparse
import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from box_prior import forward_error
from config import PROCESSED_DIR, RAW_DIR, ROOT

GAMES = 82


def check_hashes(folder) -> None:
    meta = json.loads((folder / "meta.json").read_text())
    for name, h in meta["sha256"].items():
        now = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        status = "ok" if now == h else "CHANGED SINCE THE FREEZE"
        print(f"  {name}: {status}")
    print(f"  frozen {meta['frozen_utc']}, git commit {meta['git_commit_at_freeze'][:10]}")


def team_results(year: int) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED_DIR / f"stints_{year}.parquet")
    g = st.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"),
                                  ph=("pts_h", "sum"), pa=("pts_a", "sum"))
    t = pd.concat([pd.DataFrame({"team_id": g["home"], "diff": g["ph"] - g["pa"]}),
                   pd.DataFrame({"team_id": g["away"], "diff": g["pa"] - g["ph"]})])
    t["win"] = (t["diff"] > 0).astype(int)
    r = t.groupby("team_id").agg(games=("win", "size"), wins=("win", "sum"))
    r["wins82"] = r["wins"] / r["games"] * GAMES
    abbr = pd.read_parquet(RAW_DIR / f"team_games_{year}.parquet").drop_duplicates("TEAM_ID") \
        .set_index("TEAM_ID")["TEAM_ABBREVIATION"]
    r["team"] = r.index.map(abbr)
    return r.reset_index()


def scores(pred, actual) -> str:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return f"MAE {np.mean(np.abs(pred - actual)):5.2f}   r {np.corrcoef(pred, actual)[0, 1]:.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="folder name in forecasts/, e.g. 2026-27 (default: newest)")
    args = ap.parse_args()
    folder = ROOT / "forecasts" / (args.season or sorted(p.name for p in (ROOT / "forecasts").iterdir())[-1])
    year = int(folder.name[:4])
    print(f"forward test {folder.name}")
    check_hashes(folder)
    # nothing to test before the season has games (players.py even writes empty files before tip-off)
    need = [PROCESSED_DIR / f"stints_{year}.parquet", RAW_DIR / f"players_{year}.parquet",
            RAW_DIR / f"team_games_{year}.parquet"]
    missing = [f.name for f in need if not f.exists() or pd.read_parquet(f).empty]
    if missing:
        raise SystemExit(f"no {folder.name} games yet ({', '.join(missing)} missing or empty). run this once the "
                         f"season has started, after ingest/players/lineups --seasons {year}")

    players = pd.read_csv(folder / "players.csv")
    teams = pd.read_csv(folder / "teams.csv")
    res = {"season": folder.name, "run": date.today().isoformat()}

    # ── teams ──
    now = team_results(year)
    prev = team_results(year - 1)
    base = prev.set_index("team")["wins82"].map(lambda w: GAMES / 2 + (w - GAMES / 2) * 2 / 3)
    m = now.merge(teams, on="team", how="inner")
    m["baseline"] = m["team"].map(base)
    games_so_far = int(now["games"].median())
    print(f"\nTEAMS after ~{games_so_far} games each, projected wins vs actual pace (wins per 82)")
    print(f"  model     {scores(m['proj_wins'], m['wins82'])}")
    print(f"  baseline  {scores(m['baseline'], m['wins82'])}   (last season, a third back to 41)")
    res["teams"] = {"games": games_so_far, "mae_model": float(np.mean(np.abs(m["proj_wins"] - m["wins82"]))),
                    "mae_baseline": float(np.mean(np.abs(m["baseline"] - m["wins82"])))}
    show = m.assign(diff=m["wins82"] - m["proj_wins"]).sort_values("diff")
    print("  biggest misses (actual pace - projection):")
    print(pd.concat([show.head(3), show.tail(3)])[["team", "proj_wins", "wins", "games", "wins82", "diff"]]
          .round(1).to_string(index=False))

    # ── stints ──
    print("\nSTINTS: player values -> points per 100 in every stint played so far")
    vals = {"frozen forecast": (players.assign(season=year - 1), "o_val", "d_val")}
    for f, name in (("rapm_plain.parquet", "last season plain RAPM"), ("rapm.parquet", "last season prior RAPM")):
        if (PROCESSED_DIR / f).exists():
            vals[name] = (pd.read_parquet(PROCESSED_DIR / f), "o_rapm", "d_rapm")
    res["stints"] = {}
    for name, (v, co, cd) in vals.items():
        e = forward_error(v, co, cd, seasons=[year - 1])
        res["stints"][name] = e["gain"]
        print(f"  {name:<24} explains {e['gain']:.3%} of the variance (mse {e['mse']:.1f}, level-only {e['base']:.1f})")

    # ── availability ──
    box = pd.read_parquet(RAW_DIR / f"players_{year}.parquet")[["PLAYER_ID", "GP"]]
    box_prev = pd.read_parquet(RAW_DIR / f"players_{year - 1}.parquet")[["PLAYER_ID", "GP"]]
    share = games_so_far / GAMES
    # rotation players, the ones anyone plans around (a few have no games projection, left out)
    p = players[(players["proj_min"] >= 1000) & players["exp_games"].notna()].copy()
    p["actual"] = p["player_id"].map(box.set_index("PLAYER_ID")["GP"]).fillna(0)
    p["model"] = p["exp_games"] * share
    p["baseline"] = p["player_id"].map(box_prev.set_index("PLAYER_ID")["GP"]).fillna(0) * share
    print(f"\nAVAILABILITY, {len(p)} rotation players, games played so far (of ~{games_so_far})")
    print(f"  model     {scores(p['model'], p['actual'])}")
    print(f"  baseline  {scores(p['baseline'], p['actual'])}   (last season's games played)")
    res["availability"] = {"mae_model": float(np.mean(np.abs(p["model"] - p["actual"]))),
                           "mae_baseline": float(np.mean(np.abs(p["baseline"] - p["actual"])))}

    out = folder / f"results_{res['run']}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out}")
