#!/usr/bin/env python3
"""
Step 9d: who is hurt right now, and what that does to next season.

injury.py projects games missed from injury history + age. It doesn't know that a player ended the season
still out with a torn ACL. This adds that, for the last season in the data:

  still out     listed out with an injury at the end of the season and no game since. from all the older
                absences of the same type that had already lasted as long, how much longer did they last?
                -> expected games of next season still missed. (a hamstring strain in April is long healed by
                October, a January ACL often isn't.) types with too few long cases use all types together
  rust          ACL etc: the first 20 games back are a bit worse (injury_windows.py, shrunk estimate per type,
                only the negative ones). for players still out or back fewer than 20 games at season end

injury_risk.parquet (last season's rows) gets updated:
  exp_inj_games    current absence + the history-based rate on the games after it
  exp_games        lowered by the same amount
  exp_games_later  the history-only number, for the seasons after next (surplus.py)
  val_adj          rust spread over his expected games next season (raw value units, per 100)
the originals stay in *_hist columns, so a rerun starts from them and doesn't stack.

Output: data/processed/current_injuries.parquet + updated injury_risk.parquet

  python src/current_injuries.py      # after injury.py and injury_windows.py
"""

import re

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, SEASONS, season_label
from injury_types import MIN_GAMES, injury_type
from injury_windows import RUST, all_spells, load_games

FULL = 82
MIN_CASES = 8           # fewer older cases than this for a type -> all types together
OUT_TAG = 10            # listed out within the last 10 days of the season = still out
# a fresh injury that's obviously long even if it happened in the last week ("tear", "surgery"...).
# injury.SERIOUS is too wide here: it matches any "achilles", also tendinitis and post-repair load management
FRESH_SERIOUS = re.compile(r"tear|torn|ruptur|fractur|broken|surgery|reconstruct", re.I)
NOT_FRESH = re.compile(r"management|recovery|tendin|sore", re.I)


def season_bounds(sched: pd.DataFrame) -> tuple[float, int]:
    # typical regular season length in days + opening day month/day, from the schedules we have
    y = sched["GAME_DATE"].dt.year.where(sched["GAME_DATE"].dt.month >= 8, sched["GAME_DATE"].dt.year - 1)
    b = sched.groupby(y)["GAME_DATE"].agg(["min", "max"])
    length = float((b["max"] - b["min"]).dt.days.median())
    opening_doy = int(b["min"].dt.dayofyear.median())
    return length, opening_doy


if __name__ == "__main__":
    as_of = max(SEASONS)
    rosters, scheds, apps, sched = load_games()
    end = apps["date"].max()
    length, opening_doy = season_bounds(sched)
    t0 = pd.Timestamp(f"{as_of + 1}-01-01") + pd.Timedelta(days=opening_doy - 1)   # next opening day, roughly
    print(f"data through {end.date()}, next season assumed to open {t0.date()}, {length:.0f} days long")

    sp = all_spells(rosters, scheds, apps, sched)
    sp["type"] = sp["note"].map(injury_type)

    # history: finished absences, or ones that never ended long enough ago to call it (career over = inf)
    hist = sp[sp["missed"] >= MIN_GAMES].copy()
    hist["days"] = (hist["ret"] - hist["start"]).dt.days.astype(float)
    old_open = hist["ret"].isna() & (hist["start"] < end - pd.Timedelta(days=400))
    hist.loc[old_open, "days"] = np.inf
    hist = hist[hist["days"].notna()]
    # absences over the 2011 lockout or the 2020 covid break had an offseason months longer than normal,
    # measured in days they'd look like extra-long recoveries. left out
    odd = [(pd.Timestamp("2011-06-15"), pd.Timestamp("2011-12-24")), (pd.Timestamp("2020-03-12"), pd.Timestamp("2020-12-21"))]
    stop = hist["ret"].fillna(end)
    for lo, hi in odd:
        hist = hist[~((hist["start"] <= hi) & (stop.loc[hist.index] >= lo))]
    # only players who were really in the league before (800+ min in the year before): fringe guys who never
    # come back after an injury are mostly guys who didn't get another contract, that says nothing about recovery
    mins = {p: (g["date"].to_numpy(), g["min"].to_numpy()) for p, g in apps.groupby("pid")}

    def min_before(pid, start):
        d, m = mins.get(pid, (np.array([], dtype="datetime64[ns]"), np.array([])))
        ok = (d < np.datetime64(start)) & (d >= np.datetime64(start - pd.Timedelta(days=365)))
        return float(m[ok].sum())
    hist = hist[[min_before(p, st) >= 800 for p, st in zip(hist["pid"], hist["start"])]]

    def remaining_games(typ: str, start: pd.Timestamp) -> tuple[float, float, int]:
        """expected games of next season missed, chance he's back by opening day, cases used."""
        e_obs = (end - start).days
        e0 = (t0 - start).days
        pool = hist[(hist["type"] == typ) & (hist["days"] > e_obs)]["days"]
        if len(pool) < MIN_CASES:
            pool = hist[hist["days"] > e_obs]["days"]
        if pool.empty:
            return float(FULL), 0.0, 0
        missed_days = np.clip(pool.to_numpy() - e0 - 7, 0, length)
        # "back by opening day" with a week of slack: opening day moves a few days from year to year
        return float(missed_days.mean() / length * FULL), float((pool <= e0 + 7).mean()), len(pool)

    rust = pd.read_parquet(PROCESSED_DIR / "injury_windows.parquet").set_index("group")["shrunk_rust"].clip(upper=0)

    rows = []
    # still out at the end of the season
    # a real absence already (10+ games) or a serious note (a tear on the last day of the season counts too).
    # a one-game "Achilles tendinitis" listing in April does not
    fresh_serious = sp["note"].str.contains(FRESH_SERIOUS) & ~sp["note"].str.contains(NOT_FRESH)
    cur = sp[sp["ret"].isna() & (sp["last"] >= end - pd.Timedelta(days=OUT_TAG))
             & ((sp["missed"] >= MIN_GAMES) | fresh_serious)]
    for r in cur.itertuples(index=False):
        g, p_back, n = remaining_games(r.type, r.start)
        # back during next season -> his first 20 games back carry the rust
        p_return = 1.0 if g < FULL - 0.5 else 0.0
        rows.append(dict(player_id=int(r.pid), status="out", type=r.type, out_since=r.start, note=r.note,
                         cur_games=g, p_back_opening=p_back, cases=n, rust_games=RUST * p_return))
    # back already, but fewer than 20 games before the season ended
    played = {p: g["date"].to_numpy() for p, g in apps.groupby("pid")}
    back = sp[sp["ret"].notna() & (sp["ret"] >= end - pd.Timedelta(days=90)) & (sp["missed"] >= MIN_GAMES)]
    for r in back.itertuples(index=False):
        since = int((played[r.pid] >= np.datetime64(r.ret)).sum())
        if since < RUST:
            rows.append(dict(player_id=int(r.pid), status="back", type=r.type, out_since=r.start, note=r.note,
                             cur_games=0.0, p_back_opening=1.0, cases=0, rust_games=float(RUST - since)))
    ci = pd.DataFrame(rows).sort_values("cur_games", ascending=False).drop_duplicates("player_id")
    ci["rust"] = ci["type"].map(rust).fillna(0)
    ci["season"] = as_of

    # ── into injury_risk ──
    risk = pd.read_parquet(PROCESSED_DIR / "injury_risk.parquet")
    for c in ("exp_inj_games", "exp_games", "val_adj"):
        if f"{c}_hist" not in risk.columns:
            risk[f"{c}_hist"] = risk[c]
    for c in ("exp_inj_games", "exp_games", "val_adj"):
        risk[c] = risk[f"{c}_hist"].astype(float)   # val_adj comes in as int 0s
    risk["exp_games_later"] = risk["exp_games_hist"]

    m = (risk["season"] == as_of) & risk["player_id"].isin(ci["player_id"])
    c = ci.set_index("player_id").reindex(risk.loc[m, "player_id"])
    base_inj = risk.loc[m, "exp_inj_games_hist"].to_numpy()
    cur_g = c["cur_games"].to_numpy()
    # the absence he's in + the usual rate on whatever is left of the season
    new_inj = np.minimum(FULL, cur_g + base_inj * (FULL - cur_g) / FULL)
    new_games = np.clip(risk.loc[m, "exp_games_hist"].to_numpy() - (new_inj - base_inj), 0, None)
    risk.loc[m, "exp_inj_games"] = new_inj
    risk.loc[m, "exp_games"] = new_games
    rg = np.minimum(c["rust_games"].to_numpy(), new_games)
    risk.loc[m, "val_adj"] = risk.loc[m, "val_adj_hist"].to_numpy() + \
        c["rust"].to_numpy() * rg / np.maximum(new_games, 1)
    risk.to_parquet(PROCESSED_DIR / "injury_risk.parquet", index=False)
    ci.to_parquet(PROCESSED_DIR / "current_injuries.parquet", index=False)

    names = pd.concat(rosters.values()).drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["PLAYER_NAME"]
    ci["name"] = ci["player_id"].map(names)
    ci["hist_inj"] = ci["player_id"].map(risk[risk["season"] == as_of].set_index("player_id")["exp_inj_games_hist"])
    print(f"\n{(ci['status'] == 'out').sum()} players still out at the end of {season_label(as_of)}, "
          f"{(ci['status'] == 'back').sum()} back for fewer than {RUST} games")
    show = ci[(ci["cur_games"] >= 5) | (ci["rust"] < 0) & (ci["rust_games"] > 0)].head(25)
    print(show[["name", "type", "out_since", "cur_games", "p_back_opening", "cases", "rust_games", "rust", "hist_inj"]]
          .assign(out_since=show["out_since"].dt.date).round(2).to_string(index=False))
