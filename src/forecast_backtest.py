#!/usr/bin/env python3
"""
How good is the preseason forecast? The simulator run as it would have been every October, against what
happened and against the Vegas win totals.

war.py's team check (MAE 6.1) uses the minutes players really played that season, so it knows who got hurt.
This one doesn't:

  rosters     every player's team in his first game of the season (from the stints). players with a projection
              from the season before get it (value, expected minutes), that year's draftees their draft slot
              (rookies.py), anyone else is filler at replacement level. someone who never played that season
              isn't on a roster, which is mostly what the market knew too (Rose 2012, Durant 2019, Kawhi 2021)
  forecast    season_sim: team strength best players first, lineup shape, 10,000 seasons -> wins, playoff odds
  benchmarks  everyone wins 41, same as last season, the Vegas win total (data/win_totals_history.csv, from
              basketball-reference's preseason odds pages, typed over by hand). all per 82 games, the 2011,
              2019 and 2020 seasons were shorter
  stretch     built this way the forecasts sat too close to 41, so season_sim now stretches team strength
              (STRETCH). here: the forecast without it, with a stretch learned on the other seasons (the honest
              number), and the simulator as it is now (its factor has seen all these seasons)
  over/under  which side of the line were we on, and was that side right

Not fully out of sample: the aging curve, the calibration, the injury model and the rookie model are fitted
on all seasons. The box prior and the young-player bump only use earlier (or other) seasons.

Output: data/processed/forecast_backtest.parquet, figures/forecast_backtest.png

  python src/forecast_backtest.py
"""

import json

import numpy as np
import pandas as pd

import season_sim as S
from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label

H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]
FIRST = 2013            # three seasons of RAPM behind every projection from here on


def team_abbr() -> dict:
    from nba_api.stats.static import teams
    return {t["id"]: t["abbreviation"] for t in teams.get_teams()}


def first_team(y: int, abbr: dict) -> pd.Series:
    """player -> the team he played his first game of season y for"""
    st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet", columns=["game_id", "home_team", "away_team", *H, *A])
    long = pd.concat([
        st.melt(id_vars=["game_id", "home_team"], value_vars=H, value_name="pid").rename(columns={"home_team": "team"}),
        st.melt(id_vars=["game_id", "away_team"], value_vars=A, value_name="pid").rename(columns={"away_team": "team"}),
    ])
    f = long.sort_values("game_id").drop_duplicates("pid")
    return f.set_index("pid")["team"].map(abbr)


def rosters_for(y: int, war, roles, draft, model, abbr) -> pd.DataFrame:
    w = war[war["season"] == y - 1].set_index("player_id")
    ft = first_team(y, abbr)
    r = pd.DataFrame({"player_id": ft.index, "team": ft.to_numpy()})
    r["val"] = r["player_id"].map(w["val"] + w["val_adj"].fillna(0))
    r["min"] = r["player_id"].map(w["proj_min"])
    r["avail"] = r["player_id"].map(w["proj_min"] / w["healthy_min"]).clip(0, 1).fillna(1.0)
    # that year's draft class: value and minutes of the slot
    d = draft[draft["draft_year"] == y].drop_duplicates("player_id").set_index("player_id")["pick"]
    rook = r["val"].isna() & r["player_id"].isin(d.index)
    pk = r.loc[rook, "player_id"].map(d)
    r.loc[rook, "val"], r.loc[rook, "min"] = pk.map(model["val"]), pk.map(model["min"])
    r = r.dropna(subset=["val", "min"])           # no projection, not a draftee: replacement-level filler
    rl = roles[roles["season"] <= y - 1].sort_values("season").drop_duplicates("player_id", keep="last")
    r["role"] = r["player_id"].map(rl.set_index("player_id")["role"]).fillna("wing")
    return r


def forecasts(seasons, n_sims: int = S.N_SIMS) -> pd.DataFrame:
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    prm["repl_raw"] = (prm["repl"] - prm["c"]) / prm["a"]
    prm["pos"] = json.loads((PROCESSED_DIR / "position_effects.json").read_text())
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    roles = pd.read_parquet(PROCESSED_DIR / "roles.parquet")
    draft = pd.read_parquet(RAW_DIR / "draft.parquet")
    model = pd.read_parquet(PROCESSED_DIR / "rookie_model.parquet").set_index("pick")
    abbr = team_abbr()
    gm = S.game_model(prm)
    sd = S.strength_sd(gm, prm)
    out = []
    for y in seasons:
        r = rosters_for(y, war, roles, draft, model, abbr)
        nets, off = S.league_nets(r, prm)
        res = S.simulate(nets, gm, sd, n_sims=n_sims, po_shift=S.playoff_shift(r, prm, off))
        out.append(res.assign(season=y))
    return pd.concat(out, ignore_index=True)


def made_playoffs(abbr: dict) -> pd.DataFrame:
    pg = pd.read_parquet(RAW_DIR / "playoff_games.parquet")
    r1 = pg[pg["round"] == 1]
    m = pd.concat([r1[["season", "home"]].rename(columns={"home": "t"}),
                   r1[["season", "away"]].rename(columns={"away": "t"})]).drop_duplicates()
    return m.assign(team=m["t"].map(abbr), made=1)[["season", "team", "made"]]


def left_out(x: pd.DataFrame, cols: list) -> np.ndarray:
    """actual - 41 on (cols - 41), no intercept, weights from the other seasons"""
    pred = np.zeros(len(x))
    for y in x["season"].unique():
        tr, te = (x["season"] != y).to_numpy(), (x["season"] == y).to_numpy()
        b = np.linalg.lstsq(x.loc[tr, cols].to_numpy() - 41, x.loc[tr, "wins82"].to_numpy() - 41, rcond=None)[0]
        pred[te] = 41 + (x.loc[te, cols].to_numpy() - 41) @ b
    return pred


if __name__ == "__main__":
    abbr = team_abbr()
    v = pd.read_csv(ROOT / "data" / "win_totals_history.csv")
    v["line"] = v["win_total"] * 82 / v["sched"]
    v["wins82"] = v["wins"] / (v["wins"] + v["losses"]) * 82
    last = v[["season", "team", "wins82"]].assign(season=lambda d: d["season"] + 1).rename(columns={"wins82": "last"})
    # once without season_sim's stretch (to measure it), once as the simulator runs today
    k_now, S.STRETCH = S.STRETCH, 1.0
    raw = forecasts(range(FIRST, max(SEASONS) + 1))
    S.STRETCH = k_now
    f = forecasts(range(FIRST, max(SEASONS) + 1))
    x = (v.merge(raw[["season", "team", "wins"]].rename(columns={"wins": "ours"}), on=["season", "team"])
          .merge(f[["season", "team", "wins", "playoffs"]].rename(columns={"wins": "now"}), on=["season", "team"])
          .merge(last, on=["season", "team"]).merge(made_playoffs(abbr), on=["season", "team"], how="left"))
    x["made"] = x["made"].fillna(0)
    x["ours_s"] = left_out(x, ["ours"])
    x["blend"] = left_out(x, ["ours", "line"])
    n_seasons = x["season"].nunique()
    print(f"{len(x)} team-seasons, {season_label(FIRST)} to {season_label(max(SEASONS))}")
    print(f"lines add up to {v[v['sched'] == 82].groupby('season')['win_total'].sum().mean():.0f} wins a season "
          "on average, 1,230 are there to win")

    err = lambda c: float((x["wins82"] - c).abs().mean())
    res = {"everyone wins 41": err(41), "same as last season": err(x["last"]),
           "ours before the stretch": err(x["ours"]), "our forecast": err(x["ours_s"]),
           "Vegas win total": err(x["line"]), "ours + Vegas": err(x["blend"])}
    print("\nwins per 82, mean absolute error:")
    for k, e in res.items():
        print(f"  {k:26s} {e:.2f}")
    per = x.groupby("season").apply(lambda g: pd.Series({
        "ours": (g["wins82"] - g["ours_s"]).abs().mean(), "vegas": (g["wins82"] - g["line"]).abs().mean()}),
        include_groups=False)
    print(f"  our forecast better than Vegas in {(per['ours'] < per['vegas']).sum()} of {n_seasons} seasons")
    print("  (our forecast = stretch learned on the other seasons)")

    # how far from 41 do teams really end up, per win we / the market put them away from it
    k_us = np.sum((x["ours"] - 41) * (x["wins82"] - 41)) / np.sum((x["ours"] - 41) ** 2)
    k_v = np.sum((x["line"] - 41) * (x["wins82"] - 41)) / np.sum((x["line"] - 41) ** 2)
    k_now_w = np.sum((x["now"] - 41) * (x["wins82"] - 41)) / np.sum((x["now"] - 41) ** 2)
    print(f"\nspread: without the stretch teams end up {k_us:.2f}x as far from 41 as we say, {k_v:.2f}x as far as "
          f"the line says (sd of forecasts: ours {x['ours'].std():.1f}, Vegas {x['line'].std():.1f}, "
          f"actual {x['wins82'].std():.1f})")
    print(f"the simulator as it is (STRETCH {S.STRETCH}): {k_now_w:.2f}x, MAE {err(x['now']):.2f} "
          "(its factor was picked on these seasons) -> MAE_WINS and STRETCH in season_sim.py")
    # both in one regression, bootstrap over seasons
    X = np.column_stack([x["ours"] - 41, x["line"] - 41])
    b = np.linalg.lstsq(X, x["wins82"] - 41, rcond=None)[0]
    rng = np.random.default_rng(1)
    S_ = x["season"].unique()
    bs, dm, db = [], [], []
    for _ in range(2000):
        g = pd.concat([x[x["season"] == s] for s in rng.choice(S_, len(S_))])
        bs.append(np.linalg.lstsq(np.column_stack([g["ours"] - 41, g["line"] - 41]), g["wins82"] - 41, rcond=None)[0])
        dm.append((g["wins82"] - g["ours_s"]).abs().mean() - (g["wins82"] - g["line"]).abs().mean())
        db.append((g["wins82"] - g["blend"]).abs().mean() - (g["wins82"] - g["line"]).abs().mean())
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"both together: ours {b[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}], Vegas {b[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}]")
    print(f"error vs Vegas: stretched {np.mean(dm):+.2f} [{np.percentile(dm, 2.5):+.2f}, {np.percentile(dm, 97.5):+.2f}], "
          f"blend {np.mean(db):+.2f} [{np.percentile(db, 2.5):+.2f}, {np.percentile(db, 97.5):+.2f}]")

    # over / under: the side of the line the stretched forecast was on
    push = x["wins82"] == x["line"]
    side, gap = np.sign(x["ours_s"] - x["line"]), (x["ours_s"] - x["line"]).abs()
    right = np.sign(x["wins82"] - x["line"]) == side
    ou = {}
    print("\nover / under, our side of the line:")
    for lab, m in (("all", ~push), ("4+ wins apart", ~push & (gap >= 4)), ("6+ wins apart", ~push & (gap >= 6))):
        ou[lab] = (float(right[m].mean()), int(m.sum()))
        print(f"  {lab:14s} right {right[m].mean():.1%} of {m.sum()}")
    print(f"  always the under: {(np.sign(x['wins82'] - x['line'])[~push] == -1).mean():.1%}")

    # playoff odds: said vs happened
    x["bin"] = pd.cut(x["playoffs"], [0, .1, .25, .4, .6, .75, .9, 1.0], include_lowest=True).astype(str)
    cal = x.groupby("bin").agg(n=("made", "size"), said=("playoffs", "mean"), happened=("made", "mean")).sort_values("said")
    print("\nplayoff odds, said vs happened:")
    print(cal.round(2).to_string())
    print(f"Brier score {((x['playoffs'] - x['made']) ** 2).mean():.3f} (everyone at 16 in 30: "
          f"{((16 / 30 - x['made']) ** 2).mean():.3f})")
    x.to_parquet(PROCESSED_DIR / "forecast_backtest.parquet", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid, blue, grey = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#9a9893"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(surface)
    names = list(res)[::-1]
    # ours in blue, the benchmarks grey
    a1.barh(names, [res[n] for n in names], color=[blue if n in ("our forecast", "ours + Vegas") else grey
                                                   for n in names], height=0.62)
    for i, n in enumerate(names):
        a1.text(res[n] + 0.12, i, f"{res[n]:.2f}", va="center", fontsize=8, color=ink)
    a1.set_xlim(0, max(res.values()) + 1.2)
    a1.set_xlabel("miss per team, wins per 82", color=muted, fontsize=8.5)
    a1.set_title("Preseason forecasts against the final standings", loc="left", fontsize=9.5, color=ink)
    a2.plot([0, 100], [0, 100], color=grid, lw=1.2)
    a2.scatter(cal["said"] * 100, cal["happened"] * 100, s=cal["n"] * 0.9, color=blue, zorder=3)
    a2.set_xlabel("playoff chance we gave, %", color=muted, fontsize=8.5)
    a2.set_ylabel("made the playoffs, %", color=muted, fontsize=8.5)
    a2.set_title("Playoff odds: said vs happened", loc="left", fontsize=9.5, color=ink)
    a2.set_xlim(-3, 103)
    a2.set_ylim(-3, 103)
    a2.grid(color=grid, lw=0.6)
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8, length=0)
    a1.tick_params(axis="y", labelcolor=ink)
    a1.grid(axis="x", color=grid, lw=0.6)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(ROOT / "figures" / "forecast_backtest.png", facecolor=surface)
    print(f"\nplot -> {ROOT / 'figures' / 'forecast_backtest.png'}")
