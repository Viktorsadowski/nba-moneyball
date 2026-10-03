#!/usr/bin/env python3
"""
Who's right when our model and the market disagree? The PHI plan is a bet that they are wrong about Embiid
and Brown and we are right, so test that bet on history, with the one judge neither side controls: wins.

  teams       every team-season 2013-14 to 2025-26. its players (who actually played for it that season) get
              valued three ways before the season:
                payroll   what the team pays them (the real market)
                market    what the market model says their stats are worth (market_value.py, stats up to the
                          season before, out of sample by season)
                ours      our projected WAR (war.py, as of the season before)
              each scaled by how much he really played vs how much we expected (same for all three, so
              injuries and trades count the same way for everyone)
              -> which one predicts the team's wins, and does ours still add something once the market's
              price is known? if yes, the gap between them is real money
  players     the players the two disagree most on: did he end up closer to what we said or to what the market
              said? realized WAR in the next 2 seasons (from RAPM, so this one leans our way) vs each forecast,
              all as a price: the minimum + WAR x $ per WAR

Output: data/processed/arbitrage.parquet, figures/arbitrage.png

  python src/arbitrage.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ROOT, SEASONS
from market_value import CAP, LO, features, predict, salary_series, stats_table

H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]
FIRST = 2013


def team_minutes(y: int) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet",
                         columns=["game_id", "home_team", "away_team", "dur", "pts_h", "pts_a", *H, *A])
    long = pd.concat([
        st.melt(id_vars=["home_team", "dur"], value_vars=H, value_name="pid").rename(columns={"home_team": "team"}),
        st.melt(id_vars=["away_team", "dur"], value_vars=A, value_name="pid").rename(columns={"away_team": "team"}),
    ])
    m = long.groupby(["team", "pid"])["dur"].sum().div(60).rename("min").reset_index()
    g = st.groupby("game_id").agg(h=("home_team", "first"), a=("away_team", "first"),
                                  ph=("pts_h", "sum"), pa=("pts_a", "sum"))
    res = pd.concat([pd.DataFrame({"team": g["h"], "w": (g["ph"] > g["pa"]).astype(int)}),
                     pd.DataFrame({"team": g["a"], "w": (g["pa"] > g["ph"]).astype(int)})])
    wins = res.groupby("team")["w"].agg(["sum", "size"])
    return m, (wins["sum"] / wins["size"] * 82).rename("wins82")


def cv_mae(df: pd.DataFrame, cols: list) -> float:
    err = []
    for y in df["season"].unique():
        tr, te = df["season"] != y, df["season"] == y
        X = np.column_stack([np.ones(tr.sum())] + [df.loc[tr, c] for c in cols])
        b = np.linalg.lstsq(X, df.loc[tr, "wins82"], rcond=None)[0]
        Xt = np.column_stack([np.ones(te.sum())] + [df.loc[te, c] for c in cols])
        err.append(np.abs(df.loc[te, "wins82"] - Xt @ b))
    return float(np.concatenate(err).mean())


if __name__ == "__main__":
    m_model = json.loads((PROCESSED_DIR / "market_model.json").read_text())
    m_model = dict(cols=m_model["cols"], mu=np.array(m_model["mu"]), sd=np.array(m_model["sd"]),
                   beta=np.array(m_model["beta"]), sigma=m_model["sigma"])
    stats = stats_table()
    sal = salary_series().set_index(["player_id", "season"])["salary"]
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    war = war.assign(season=war["season"] + 1).set_index(["player_id", "season"])   # as of s-1 for s
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    dpw_share = 7.7e6 / CAP[max(SEASONS)]

    teams, players = [], []
    for y in range(FIRST, max(SEASONS) + 1):
        mins, wins = team_minutes(y)
        f = features(stats, pd.DataFrame({"player_id": mins["pid"].unique(), "season": y}), skip_missed=True)
        price = pd.Series(np.nan_to_num(predict(m_model, f), nan=0.03), index=f["player_id"].to_numpy())
        mins["price"] = mins["pid"].map(price)
        mins["salary"] = [sal.get((p, y), np.nan) for p in mins["pid"]]
        mins["salary"] = mins["salary"] / CAP[y]
        w = war.reindex(pd.MultiIndex.from_arrays([mins["pid"], np.full(len(mins), y)]))
        mins["proj_min"] = w["proj_min"].to_numpy()
        mins["war_proj"] = w["war"].to_numpy()
        # only players all three can price: history in our model, a salary that season
        ok = mins["proj_min"].notna() & mins["salary"].notna()
        mins["scale"] = (mins["min"] / mins["proj_min"]).clip(0, 1.5)
        mins["ours"] = mins["war_proj"] * dpw_share            # $ per WAR, as a share of the cap
        x = mins[ok]
        for c in ("salary", "price", "ours"):
            x = x.assign(**{f"{c}_s": x[c] * x["scale"]})
        t = x.groupby("team")[["salary_s", "price_s", "ours_s", "min"]].sum()
        t["cover"] = t["min"] / mins.groupby("team")["min"].sum()
        t["wins82"] = wins
        t["season"] = y
        teams.append(t.reset_index())
        players.append(x.assign(season=y))
    T = pd.concat(teams).dropna()
    T = T[T["cover"] > 0.6]
    print(f"{len(T)} team-seasons, {FIRST}-{max(SEASONS)}; players priced cover {T['cover'].mean():.0%} of the minutes")

    print("\nteam wins (per 82) from what the roster was worth before the season, leave one season out:")
    res = {}
    for name, cols in (("payroll", ["salary_s"]), ("market model", ["price_s"]), ("our model", ["ours_s"]),
                       ("payroll + ours", ["salary_s", "ours_s"]), ("market + ours", ["price_s", "ours_s"]),
                       ("all three", ["salary_s", "price_s", "ours_s"])):
        res[name] = cv_mae(T, cols)
        r = np.corrcoef(T[cols[-1]], T["wins82"])[0, 1] if len(cols) == 1 else np.nan
        print(f"  {name:16s} MAE {res[name]:.2f} wins" + (f"   r {r:.2f}" if len(cols) == 1 else ""))
    # joint fit on everything, standardized: how much each one counts once the others are known
    Z = T[["salary_s", "price_s", "ours_s"]]
    Z = (Z - Z.mean()) / Z.std()
    X = np.column_stack([np.ones(len(T)), Z])
    b = np.linalg.lstsq(X, T["wins82"], rcond=None)[0]
    rng = np.random.default_rng(3)
    bs = np.array([np.linalg.lstsq(X[i], T["wins82"].to_numpy()[i], rcond=None)[0]
                   for i in (rng.integers(0, len(T), len(T)) for _ in range(1000))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print("\nall three together, wins per 1 sd (95% interval):")
    for k, name in enumerate(("payroll", "market model", "our model"), start=1):
        print(f"  {name:13s} {b[k]:+.2f}  [{lo[k]:+.2f}, {hi[k]:+.2f}]")

    # ── players: the big disagreements ──
    P = pd.concat(players)
    # same currency as the market price: a replacement-level guy still costs the minimum
    P["ours_p"] = LO + P["war_proj"].clip(lower=0) * dpw_share
    P["gap"] = P["ours_p"] - P["price"]                   # + = we like him more than the market does
    # realized WAR next 2 seasons (from that season's RAPM), as a share of the cap like the two forecasts
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet").set_index(["player_id", "season"])
    k = {int(s): v / 100 / prm["ppw"] for s, v in prm["pace"].items()}

    def realized(pid, y):
        out = []
        for s in (y, y + 1):
            if (pid, s) in rapm.index:
                r = rapm.loc[(pid, s)]
                out.append(max(0.0, (prm["a"] * r["rapm"] + prm["c"] - prm["repl"]) * r["minutes"] * k.get(s, 0.02)))
            else:
                out.append(0.0)
        return LO + np.mean(out) * dpw_share
    P = P[P["season"] <= max(SEASONS) - 1]
    big = P[P["gap"].abs() >= P["gap"].abs().quantile(0.9)].copy()
    big["real"] = [realized(p, y) for p, y in zip(big["pid"], big["season"])]
    big["side"] = np.where(big["gap"] > 0, "we liked more", "market liked more")
    g = big.groupby("side").apply(lambda d: pd.Series({
        "n": len(d), "ours": d["ours_p"].mean() * 100, "market": d["price"].mean() * 100, "real": d["real"].mean() * 100,
        "closer_to_us": float((np.abs(d["real"] - d["ours_p"]) < np.abs(d["real"] - d["price"])).mean())}),
        include_groups=False)
    print("\nthe 10% biggest disagreements (value in % of cap per season, realized = the next 2 seasons):")
    print(g.round(2).to_string())
    T.to_parquet(PROCESSED_DIR / "arbitrage.parquet", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid, blue, orange, grey = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834", "#9a9893"
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    names = ["payroll", "market model", "our model", "market + ours"]
    BLUE_ = blue
    vals = [res[n] for n in names]
    # ours in blue, the rest grey
    ax.barh(names[::-1], vals[::-1], color=[BLUE_ if n.startswith("our") else grey for n in names][::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8, color=ink)
    ax.set_xlim(min(vals) - 1, max(vals) + 0.6)
    ax.set_xlabel("error predicting team wins, per 82 (season left out)", color=muted, fontsize=8.5)
    ax.set_title("What the roster is worth before the season vs how many games it wins", loc="left", fontsize=10, color=ink)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "arbitrage.png", facecolor=surface)
