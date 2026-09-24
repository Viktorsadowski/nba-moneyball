#!/usr/bin/env python3
"""
Step 4: blend single-season RAPM into one "current value" per player.

  value = sum(w_i * min_i * rapm_i) / (sum(w_i * min_i) + ghost)

  w      recency weights, most recent season first (default 50/30/20)
  min_i  minutes that season, RAPM noise shrinks with playing time so more minutes = more trust
  ghost  "ghost minutes" at 0 (league average). Barely moves a guy with 7000 min over 3 years,
         pulls a one-good-season guy towards the middle. Same trick as Marcel in baseball.

Done separately for O and D, total = O + D.

Age adjustment (needs aging.py first): every past season gets moved along the aging curve to the
age he'll be NEXT season, before blending. A 21-year-old's rookie year counts as better than it was,
a 34-year-old's good year from 3 seasons ago counts as worse. So the value is a projection of next season.

Backtest: blend seasons t-3..t-1, predict season t RAPM, error weighted by season t minutes.
Tells us whether 50/30/20, the ghost minutes and the age adjustment actually help.
(small leak: the aging curve is fit on all seasons incl. the ones being predicted, the curve is
a smooth quadratic over ~8k player-seasons so it barely matters, but worth knowing)

  python src/value.py                               # backtest + value table with 50/30/20, ghost 1000
  python src/value.py --weights 0.6 0.25 0.15 --ghost 1500
  python src/value.py --no-age                      # without the age adjustment
"""

import argparse
from itertools import product

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label

WEIGHT_GRID = [
    (1.0, 0.0, 0.0),     # last season only
    (0.5, 0.5, 0.0),
    (0.7, 0.2, 0.1),
    (0.6, 0.25, 0.15),
    (0.5, 0.3, 0.2),     # the default
    (0.4, 0.35, 0.25),
    (1 / 3, 1 / 3, 1 / 3),
]
GHOST_GRID = [0, 250, 500, 1000, 1500, 2000, 3000]
TARGET_MIN = 500          # only score players with real minutes in the target season
FIRST_TARGET = 2013       # first season with 3 full seasons behind it

# set in main: (o_fn, d_fn), each age -> points vs peak. None = no age adjustment
AGING = None


def load_aging():
    # curve is a quadratic per side, refit the table so it also works past 38 (LeBron)
    c = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    return tuple(np.poly1d(np.polyfit(c["age"], c[k], 2)) for k in ("o_curve", "d_curve"))


def add_ages(rapm: pd.DataFrame) -> pd.DataFrame:
    ages = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["SEASON", "PLAYER_ID", "AGE"]]
                      for y in SEASONS if (RAW_DIR / f"players_{y}.parquet").exists()])
    ages.columns = ["season", "player_id", "age"]
    return rapm.merge(ages, on=["season", "player_id"], how="left")


# ── the blend ─────────────────────────────────────────────────────────────────

def blend(rapm: pd.DataFrame, as_of: int, weights, ghost: float) -> pd.DataFrame:
    """Value per player using seasons as_of, as_of-1, as_of-2 (as many as weights)."""
    parts = []
    for lag, w in enumerate(weights):
        s = rapm[rapm["season"] == as_of - lag][["player_id", "minutes", "o_rapm", "d_rapm", "age"]].copy()
        s["wm"] = w * s["minutes"]
        if AGING is not None:
            # move that season to the age he'll be in the season we're predicting (as_of + 1)
            o_fn, d_fn = AGING
            now, then = s["age"] + lag + 1, s["age"]
            ok = s["age"].notna()
            s.loc[ok, "o_rapm"] += o_fn(now[ok]) - o_fn(then[ok])
            s.loc[ok, "d_rapm"] += d_fn(now[ok]) - d_fn(then[ok])
        parts.append(s)
    d = pd.concat(parts)

    g = d.assign(wo=d["wm"] * d["o_rapm"], wd=d["wm"] * d["d_rapm"]).groupby("player_id")
    out = g[["wm", "wo", "wd"]].sum()
    denom = out["wm"] + ghost
    out["o_val"] = out["wo"] / denom
    out["d_val"] = out["wd"] / denom
    out["val"] = out["o_val"] + out["d_val"]
    return out[["o_val", "d_val", "val", "wm"]].reset_index()


# ── backtest ──────────────────────────────────────────────────────────────────

def backtest(rapm: pd.DataFrame, weights, ghost: float) -> float:
    last = int(rapm["season"].max())
    errs, wsum = 0.0, 0.0
    for t in range(FIRST_TARGET, last + 1):
        pred = blend(rapm, t - 1, weights, ghost)
        tgt = rapm[(rapm["season"] == t) & (rapm["minutes"] >= TARGET_MIN)][["player_id", "minutes", "rapm"]]
        # no history at all = rookie, predicted at 0 (league average)
        m = tgt.merge(pred[["player_id", "val"]], on="player_id", how="left").fillna({"val": 0.0})
        errs += np.sum(m["minutes"] * (m["rapm"] - m["val"]) ** 2)
        wsum += m["minutes"].sum()
    return np.sqrt(errs / wsum)


def run_grid(rapm: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(weights="/".join(f"{w:.2f}" for w in ws), ghost=g, rmse=backtest(rapm, ws, g))
            for ws, g in product(WEIGHT_GRID, GHOST_GRID)]
    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.3, 0.2])
    ap.add_argument("--ghost", type=float, default=1000)
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-age", action="store_true")
    args = ap.parse_args()

    rapm = add_ages(pd.read_parquet(PROCESSED_DIR / "rapm.parquet"))
    use_age = not args.no_age and (PROCESSED_DIR / "aging_curve.parquet").exists()

    if not args.no_backtest:
        # reference points: everyone = 0, and the raw RAPM of last season with no blending at all
        zero = backtest(rapm, (0.0,), 1e12)
        print(f"backtest {FIRST_TARGET}-{rapm['season'].max()}, predicting next season RAPM "
              f"(minutes-weighted RMSE, players {TARGET_MIN}+ min)")
        print(f"  everyone = 0:         {zero:.3f}")
        if use_age:
            no_age = backtest(rapm, args.weights, args.ghost)
            AGING = load_aging()
            print(f"  your settings, no age adjustment: {no_age:.3f}")
            print("  grid below is WITH the age adjustment")
        grid = run_grid(rapm)
        table = grid.pivot(index="weights", columns="ghost", values="rmse")
        print(table.round(3).to_string())
        best = grid.loc[grid["rmse"].idxmin()]
        mine = backtest(rapm, args.weights, args.ghost)
        print(f"\n  best: weights {best['weights']}, ghost {best['ghost']:.0f} -> {best['rmse']:.3f}")
        print(f"  yours: weights {'/'.join(map(str, args.weights))}, ghost {args.ghost:.0f} -> {mine:.3f}")

    AGING = load_aging() if use_age else None

    # value table "as of" the end of every season, with the chosen settings
    names = rapm.groupby("player_id")["name"].last()
    out = []
    for s in sorted(rapm["season"].unique()):
        v = blend(rapm, s, args.weights, args.ghost)
        cur = rapm[rapm["season"] == s][["player_id", "minutes", "team_id"]]
        v = v.merge(cur, on="player_id", how="left")  # minutes/team this season, NaN if he sat out
        v["season"] = s
        out.append(v)
    val = pd.concat(out, ignore_index=True)
    val["name"] = val["player_id"].map(names)
    val.to_parquet(PROCESSED_DIR / "value.parquet", index=False)

    last = val["season"].max()
    show = val[(val["season"] == last) & (val["minutes"] >= 1000)].sort_values("val", ascending=False)
    tag = "age-adjusted projection for next season" if use_age else "no age adjustment"
    print(f"\nvalue as of end of {season_label(last)} ({tag}), top 15 (1000+ min this season):")
    print(show[["name", "minutes", "o_val", "d_val", "val"]].head(15).round(2).to_string(index=False))
