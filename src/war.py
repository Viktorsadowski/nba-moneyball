#!/usr/bin/env python3
"""
Step 8: wins above replacement, projected one season ahead.

  WAR = (value - replacement) * possessions / 100 / points_per_win

  value            age-adjusted blended RAPM from value.py (points per 100, O + D), calibrated on
                   team results: the ridge + blend shrink everything towards 0, so it gets stretched back
                   by however much it takes to predict next season's team point differentials
  replacement      what a near-minimum contract gets you: minutes-weighted value of guys paid in the
                   bottom 20% of the league the season after (so, what teams actually got for that money)
  possessions      projected minutes * league pace per minute
  points_per_win   fitted on our own games: team wins ~ point differential, per team-season

Two versions:
  war          minutes = expected games (injury.py: 82 - expected injury games - expected other
               absences) * his blended minutes per game. without injury.py's output it falls back
               to a 50/30/20 blend of his last 3 seasons' minutes
  war_healthy  same but with zero injury games, only the usual rest / coach's decision absences
  inj_cost     war_healthy - war = wins his injury risk is expected to cost

Sanity check: predict every team's wins with last season's values and this season's actual
possessions, compare with the real win totals. Same table gives the calibration.

Output: data/processed/war.parquet (one row per player per "as of" season, projecting the next one)

  python src/war.py
"""

import json
import re

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from lineups import norm, strip_suffix
from scrape_salaries import unmojibake

WEIGHTS = (0.5, 0.3, 0.2)
MIN_SALARY_Q = 0.20        # bottom 20% of salaries = "near minimum"
FULL_SEASON = 82
H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]


# ── points per win ────────────────────────────────────────────────────────────

def team_seasons() -> pd.DataFrame:
    """wins, games and point differential per team-season, from the stints."""
    rows = []
    for y in SEASONS:
        st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet")
        g = st.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"),
                                      ph=("pts_h", "sum"), pa=("pts_a", "sum"))
        home = pd.DataFrame({"team": g["home"], "diff": g["ph"] - g["pa"]})
        away = pd.DataFrame({"team": g["away"], "diff": g["pa"] - g["ph"]})
        t = pd.concat([home, away])
        t["win"] = (t["diff"] > 0).astype(int)
        s = t.groupby("team").agg(games=("win", "size"), wins=("win", "sum"), diff=("diff", "mean"))
        s["season"] = y
        rows.append(s.reset_index())
    return pd.concat(rows, ignore_index=True)


def points_per_win(ts: pd.DataFrame) -> float:
    # win% = 0.5 + b * diff per game, so one extra point per game over 82 games = 82 * b wins
    b = np.polyfit(ts["diff"], ts["wins"] / ts["games"], 1, w=ts["games"])[0]
    return 1 / b  # points of season differential per win


# ── salaries -> player ids ────────────────────────────────────────────────────

def compact(n: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm(n))


def unsuffix(c: str) -> str:
    return re.sub(r"(jr|sr|ii|iii|iv)$", "", c)


def last_name(n: str) -> str:
    return compact(strip_suffix(norm(n)).split(" ")[-1])


def salaries_with_ids() -> pd.DataFrame:
    # full name first, then last name + team, then last name + first initial ("Lou" vs "Louis" Williams)
    out = []
    for y in SEASONS:
        f = RAW_DIR / f"salaries_{y}.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f).dropna(subset=["salary"])
        s["player"] = s["player"].map(unmojibake)  # older scrapes have "JokiÄ" for "Jokić"
        p = pd.read_parquet(RAW_DIR / f"players_{y}.parquet")
        full, by_team, by_init = {}, {}, {}
        for pid, n, t in zip(p["PLAYER_ID"], p["PLAYER_NAME"], p["TEAM_ABBREVIATION"]):
            for k in {compact(n), unsuffix(compact(n))}:
                full.setdefault(k, set()).add(pid)
            by_team.setdefault((last_name(n), t), set()).add(pid)
            by_init.setdefault((last_name(n), compact(n)[:1]), set()).add(pid)

        def find(n, t):
            for hit in (full.get(compact(n)), full.get(unsuffix(compact(n))),
                        by_team.get((last_name(n), t)), by_init.get((last_name(n), compact(n)[:1]))):
                if hit and len(hit) == 1:
                    return next(iter(hit))
            return None

        s["player_id"] = [find(n, t) for n, t in zip(s["player"], s["team"])]
        s = s.dropna(subset=["player_id"])
        # traded players show up on two team pages with the same salary, keep one
        s = s.groupby("player_id", as_index=False)["salary"].max()
        s["season"] = y
        out.append(s)
    return pd.concat(out, ignore_index=True)


# ── playing time ──────────────────────────────────────────────────────────────

def minutes_table(ts: pd.DataFrame) -> pd.DataFrame:
    p = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["SEASON", "PLAYER_ID", "GP", "MIN", "AGE"]]
                   for y in SEASONS])
    p.columns = ["season", "player_id", "gp", "min", "age"]
    # shortened seasons (66 games in 2011-12, bubble, 72 in 2020-21) scaled up to 82
    season_len = ts.groupby("season")["games"].median()
    p["min82"] = p["min"] * FULL_SEASON / p["season"].map(season_len)
    p["mpg"] = p["min"] / p["gp"].clip(lower=1)
    return p


def projected_minutes(mins: pd.DataFrame, as_of: int) -> pd.DataFrame:
    # blend over the seasons he was in the league, weights renormalised (a rookie = his one season)
    parts = []
    for lag, w in enumerate(WEIGHTS):
        s = mins[mins["season"] == as_of - lag][["player_id", "min82", "mpg", "gp"]].copy()
        s["w"] = w
        parts.append(s)
    d = pd.concat(parts)
    g = d.assign(wm=d["w"] * d["min82"], wp=d["w"] * d["mpg"] * d["gp"], wg=d["w"] * d["gp"]).groupby("player_id")
    out = pd.DataFrame({
        "proj_min": g["wm"].sum() / g["w"].sum(),
        # minutes per game weighted by games played, so a 10-game season doesn't set his role
        "proj_mpg": g["wp"].sum() / g["wg"].sum().clip(lower=1),
    })
    out["healthy_min"] = out["proj_mpg"] * FULL_SEASON
    return out.reset_index()


# ── team check ────────────────────────────────────────────────────────────────

def team_check(val: pd.DataFrame, ts: pd.DataFrame, rookie: float) -> pd.DataFrame:
    """per team-season: predicted point differential from last season's values, and the real one."""
    rows = []
    for t in list(SEASONS)[3:]:
        st = pd.read_parquet(PROCESSED_DIR / f"stints_{t}.parquet")
        long = pd.concat([
            st.melt(id_vars=["home_team", "poss"], value_vars=H, value_name="player_id")
              .rename(columns={"home_team": "team"}),
            st.melt(id_vars=["away_team", "poss"], value_vars=A, value_name="player_id")
              .rename(columns={"away_team": "team"}),
        ])
        tp = long.groupby(["team", "player_id"])["poss"].sum().reset_index()
        prev = val[val["season"] == t - 1].set_index("player_id")["val"]
        tp["val"] = tp["player_id"].map(prev).fillna(rookie)
        g = tp.assign(pts=tp["val"] * tp["poss"] / 100).groupby("team")
        real = ts[ts["season"] == t].set_index("team")
        rows.append(pd.DataFrame({"season": t, "net": g["pts"].sum(), "pp100": g["poss"].sum() / 100,
                                  "diff": real["diff"] * real["games"], "wins": real["wins"],
                                  "games": real["games"]}))
    return pd.concat(rows).dropna()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ts = team_seasons()
    ppw = points_per_win(ts)
    print(f"points per win: {ppw:.1f} (fitted on {len(ts)} team-seasons)")

    val = pd.read_parquet(PROCESSED_DIR / "value.parquet")
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    mins = minutes_table(ts)
    sal = salaries_with_ids()

    # league pace: possessions per player-minute on court (same for everyone, ~2 per minute)
    pace = (rapm.groupby("season")["poss"].sum() / rapm.groupby("season")["minutes"].sum())

    # rookies have no value yet, give them what first-year players actually posted
    first = rapm.sort_values("season").drop_duplicates("player_id")
    first = first[first["season"] > min(SEASONS)]
    rookie = np.average(first["rapm"], weights=first["minutes"])

    # ── calibration, on team results ──
    # RAPM is ridge-shrunk and the blend shrinks again, so the values are squeezed towards 0.
    # predict every team's season t point differential from its players' season t-1 values
    # (possessions from season t), then regress the real differential on it:
    #   real = a * predicted + c*possessions   ->   val_cal = a * val + c
    # out of sample by construction (t-1 values predicting t), a > 1 means the values were too timid
    chk = team_check(val, ts, rookie)
    X = np.column_stack([chk["net"], chk["pp100"]])
    a, c = np.linalg.lstsq(X, chk["diff"], rcond=None)[0]
    r_raw = np.corrcoef(chk["net"], chk["diff"])[0, 1]
    print(f"calibration: val_cal = {a:.2f} * val {c:+.2f}   (team differential r = {r_raw:.2f}, "
          f"{len(chk)} team-seasons)")
    val["val_cal"] = a * val["val"] + c
    rookie_cal = a * rookie + c

    # ── replacement level ──
    # value as of season s, for guys who then played season s+1 on a near-minimum deal
    cut = sal.groupby("season")["salary"].quantile(MIN_SALARY_Q)
    cheap = sal[sal["salary"] <= sal["season"].map(cut)][["season", "player_id"]]
    nxt = rapm[["season", "player_id", "minutes"]]
    pool = cheap.merge(nxt, on=["season", "player_id"]).assign(season=lambda d: d["season"] - 1)
    pool = pool.merge(val[["season", "player_id", "val_cal"]], on=["season", "player_id"])
    repl = np.average(pool["val_cal"], weights=pool["minutes"])
    by_season = pool.groupby("season").apply(lambda g: np.average(g["val_cal"], weights=g["minutes"]),
                                             include_groups=False)
    league_war = -repl * (rapm.groupby("season")["poss"].sum().mean() / 100) / ppw
    print(f"replacement level: {repl:+.2f} per 100 (from {len(pool):,} near-minimum player-seasons, "
          f"by season {by_season.min():+.2f} to {by_season.max():+.2f})")
    print(f"  -> about {league_war:.0f} WAR in the league per season, a replacement-level team wins "
          f"~{41 - league_war / 30:.0f} of 82")

    # team check again, now in wins and calibrated
    pred82 = 41 + (a * chk["net"] + c * chk["pp100"]) / ppw * 82 / chk["games"]
    real82 = chk["wins"] / chk["games"] * 82
    print(f"team wins check: MAE {(pred82 - real82).abs().mean():.1f} wins per 82, "
          f"r = {np.corrcoef(pred82, real82)[0, 1]:.2f}")

    # ── projections, as of the end of each season, for the next one ──
    rpath = PROCESSED_DIR / "injury_risk.parquet"
    risk = pd.read_parquet(rpath) if rpath.exists() else None
    print("minutes from:", "injury model (injury.py)" if risk is not None else "past minutes blend (run injury.py)")
    out = []
    for s in sorted(val["season"].unique()):
        v = val[val["season"] == s][["player_id", "name", "o_val", "d_val", "val", "val_cal"]]
        m = projected_minutes(mins, s)
        d = v.merge(m, on="player_id", how="inner")
        if risk is not None:
            r = risk[risk["season"] == s][["player_id", "exp_games", "exp_inj_games", "exp_other_games", "val_adj"]]
            d = d.merge(r, on="player_id", how="left")
            has = d["exp_games"].notna()
            d.loc[has, "proj_min"] = d.loc[has, "exp_games"] * d.loc[has, "proj_mpg"]
            d.loc[has, "healthy_min"] = (FULL_SEASON - d.loc[has, "exp_other_games"]) * d.loc[has, "proj_mpg"]
            # injury discount on the value itself, 0 unless injury.py found a real effect
            d["val_cal"] = d["val_cal"] + a * d["val_adj"].fillna(0)
        # last known age, moved forward if he sat the whole season out (Kyrie, Haliburton in 2025-26)
        known = mins[mins["season"] <= s].sort_values("season").drop_duplicates("player_id", keep="last")
        age = known.set_index("player_id")["age"] + (s - known.set_index("player_id")["season"])
        d["age_next"] = d["player_id"].map(age) + 1
        k = pace.get(s, pace.iloc[-1]) / 100 / ppw
        d["war"] = (d["val_cal"] - repl) * d["proj_min"] * k
        d["war_healthy"] = (d["val_cal"] - repl) * d["healthy_min"] * k
        d["inj_cost"] = d["war_healthy"] - d["war"]
        d["season"] = s
        out.append(d)
    war = pd.concat(out, ignore_index=True)
    war.to_parquet(PROCESSED_DIR / "war.parquet", index=False)
    # constants surplus.py needs to project further out
    params = dict(repl=repl, a=a, c=c, ppw=ppw, pace={int(k): v for k, v in pace.items()})
    (PROCESSED_DIR / "war_params.json").write_text(json.dumps(params, indent=1))

    last = war["season"].max()
    show = war[war["season"] == last].sort_values("war", ascending=False)
    print(f"\nprojected {season_label(last + 1)} WAR, top 20:")
    cols = ["name", "age_next", "val_cal", "proj_min", "war", "war_healthy", "inj_cost"]
    if "exp_inj_games" in show:
        cols.insert(4, "exp_inj_games")
    print(show[cols].head(20).round(1).to_string(index=False))
    # biggest share of their healthy value lost to injury risk, among players that matter
    top = show[show["war_healthy"] > 2].assign(pct_lost=lambda x: 100 * x["inj_cost"] / x["war_healthy"])
    top = top.sort_values("pct_lost", ascending=False)
    print(f"\nbiggest share of value lost to injury risk, {season_label(last + 1)} (2+ WAR healthy):")
    print(top[cols + ["pct_lost"]].head(10).round(1).to_string(index=False))
