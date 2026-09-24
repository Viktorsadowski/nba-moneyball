#!/usr/bin/env python3
"""
Sub-question: does defense win championships?

Read as: for two teams with the same regular-season net rating, does the one that gets there with
defense do better in the playoffs than the one that gets there with offense?

  team ratings   regular season, from our stints: points scored / allowed per 100 possessions vs the
                 league average that season. o = offense (+ good), d = defense (+ good), net = o + d
  games          margin (home - away) = a + b_o * (o_home - o_away) + b_d * (d_home - d_away)
                 fitted on regular season games (ratings leave that game out) and on playoff games

If defense wins championships, b_d > b_o in the playoffs. The regular season fit is the baseline where
both should be about the same. Playoff CI from a bootstrap over series (games in a series aren't
independent). Plus a plain look at the champions: where did they rank on O and D.

Output
  data/processed/defense.parquet
  figures/defense.png

  python src/playoffs.py     # playoff results first
  python src/defense.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label

RNG = np.random.default_rng(9)


def ratings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """team-season o/d/net + per game totals (for the leave-one-out in the regular season fit)."""
    teams, games = [], []
    for y in SEASONS:
        st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet",
                             columns=["game_id", "home_team", "away_team", "pts_h", "pts_a", "poss"])
        g = st.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"),
                                      pts_h=("pts_h", "sum"), pts_a=("pts_a", "sum"), poss=("poss", "sum"))
        g["season"] = y
        games.append(g.reset_index())
        # per team: points for, points against, possessions (both sides play about the same number)
        h = g[["home", "pts_h", "pts_a", "poss"]].set_axis(["team", "pf", "pa", "poss"], axis=1)
        a = g[["away", "pts_a", "pts_h", "poss"]].set_axis(["team", "pf", "pa", "poss"], axis=1)
        t = pd.concat([h, a]).groupby("team").sum()
        t["season"] = y
        teams.append(t.reset_index())
    return pd.concat(teams, ignore_index=True), pd.concat(games, ignore_index=True)


def od(pf, pa, poss, lg):
    return 100 * pf / poss - lg, lg - 100 * pa / poss


def design(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"margin": g["pts_h"] - g["pts_a"], "do": g["o_h"] - g["o_a"], "dd": g["d_h"] - g["d_a"]})


def ols(x: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x["do"], x["dd"]])
    return np.linalg.lstsq(X, x["margin"].to_numpy(), rcond=None)[0]


def logit(x: pd.DataFrame, iters=25) -> np.ndarray:
    # plain newton, 3 coefficients, no need for statsmodels
    X = np.column_stack([np.ones(len(x)), x["do"], x["dd"]])
    y = (x["margin"] > 0).to_numpy(float)
    b = np.zeros(3)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H + 1e-9 * np.eye(3), X.T @ (y - p))
    return b


def boot(x: pd.DataFrame, groups: np.ndarray, fn, n=1000) -> np.ndarray:
    idx = pd.Series(np.arange(len(x))).groupby(groups).apply(np.array).to_list()
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(idx), len(idx))
        out.append(fn(x.iloc[np.concatenate([idx[i] for i in pick])]))
    return np.array(out)


def plot(tab: pd.DataFrame, playoff_teams: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor(surface)
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8.5)

    # left: points of margin per point of O vs D edge
    x = np.arange(2)
    for k, (term, color, lab) in enumerate((("b_o", blue, "offense edge"), ("b_d", orange, "defense edge"))):
        t = tab.set_index("sample")
        v, lo, hi = t[term], t[f"{term}_lo"], t[f"{term}_hi"]
        xs = x + (k - 0.5) * 0.22
        a1.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", color=color, ms=7, capsize=0, lw=1.6, label=lab)
    a1.set_xticks(x, tab["sample"])
    a1.set_xlim(-0.6, 1.6)
    a1.set_ylabel("points of margin per point / 100 edge", color=muted, fontsize=8.5)
    a1.set_title("What a rating edge is worth in a game", loc="left", fontsize=10, color=ink)
    a1.grid(axis="y", color=grid, lw=0.8)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="lower left")

    # right: every playoff team, O vs D, champions highlighted
    p = playoff_teams
    a2.scatter(p["o"], p["d"], s=16, color=muted, alpha=0.35, lw=0, label="playoff teams")
    c = p[p["champion"]]
    a2.scatter(c["o"], c["d"], s=46, color=orange, zorder=3, label="champions")
    # labels: first spot (right/left, above/below) that doesn't hit an earlier label. boxes in data units,
    # rough size of "GSW 18" at this scale
    wbox, hbox, placed = 0.8, 0.3, []
    for r in c.sort_values("o", ascending=False).itertuples():
        for dx, dy in ((0.15, 0.08), (0.15, -0.38), (-0.95, 0.08), (-0.95, -0.38), (0.15, 0.4), (-0.95, 0.4)):
            x0, y0 = r.o + dx, r.d + dy
            if all(abs(x0 - px) > wbox or abs(y0 - py) > hbox for px, py in placed):
                break
        placed.append((x0, y0))
        a2.text(x0, y0, f"{r.abbr} {r.season % 100 + 1:02d}", fontsize=7, color=ink)
    x_lo, x_hi = p["o"].min() - 0.8, p["o"].max() + 1.2
    y_lo, y_hi = p["d"].min() - 0.8, p["d"].max() + 0.8
    for k in range(-15, 19, 3):
        a2.plot([-20, 20], [k + 20, k - 20], color=grid, lw=0.6, zorder=0)  # same-net lines
    a2.axhline(0, color=grid, lw=0.8)
    a2.axvline(0, color=grid, lw=0.8)
    a2.set_xlim(x_lo, x_hi)
    a2.set_ylim(y_lo, y_hi)
    a2.set_xlabel("offense vs league, per 100", color=muted, fontsize=8.5)
    a2.set_ylabel("defense vs league, per 100", color=muted, fontsize=8.5)
    a2.set_title("Regular season O and D of every playoff team (diagonals = same net)", loc="left", fontsize=10,
                 color=ink)
    a2.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")
    fig.suptitle("Does defense win championships?", x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "defense.png", facecolor=surface)
    print(f"\nplot -> {out / 'defense.png'}")


if __name__ == "__main__":
    teams, games = ratings()
    lg = teams.groupby("season").apply(lambda t: 100 * t["pf"].sum() / t["poss"].sum())
    teams["o"], teams["d"] = od(teams["pf"], teams["pa"], teams["poss"], teams["season"].map(lg))
    teams["net"] = teams["o"] + teams["d"]
    T = teams.set_index(["season", "team"])

    # regular season games, each game's own points taken out of both teams' ratings
    g = games.copy()
    lgs = g["season"].map(lg)
    for side, pf, pa in (("h", "pts_h", "pts_a"), ("a", "pts_a", "pts_h")):
        tm = g["home"] if side == "h" else g["away"]
        tt = T.reindex(pd.MultiIndex.from_arrays([g["season"], tm]))
        o, d = od(tt["pf"].to_numpy() - g[pf].to_numpy(), tt["pa"].to_numpy() - g[pa].to_numpy(),
                  tt["poss"].to_numpy() - g["poss"].to_numpy(), lgs.to_numpy())
        g[f"o_{side}"], g[f"d_{side}"] = o, d
    reg = design(g)

    po = pd.read_parquet(RAW_DIR / "playoff_games.parquet")
    for side, tm in (("h", "home"), ("a", "away")):
        tt = T.reindex(pd.MultiIndex.from_arrays([po["season"], po[tm]]))
        po[f"o_{side}"], po[f"d_{side}"] = tt["o"].to_numpy(), tt["d"].to_numpy()
    po = po.dropna(subset=["o_h", "o_a"])
    pox = design(po)
    series = (po["season"].astype(str) + "_" + po["game_id"].str[7:9]).to_numpy()
    print(f"{len(reg):,} regular season games, {len(pox):,} playoff games in {len(set(series))} series")

    rows = []
    for name, x, grp in (("regular season", reg, g["season"].astype(str) + "_" + g["home"].astype(str)),
                         ("playoffs", pox, series)):
        b = ols(x)
        bb = boot(x, np.asarray(grp), ols, n=300 if name == "regular season" else 1000)
        lb = logit(x)
        lbb = boot(x, np.asarray(grp), logit, n=200 if name == "regular season" else 500)
        diff = bb[:, 2] - bb[:, 1]
        rows.append(dict(sample=name, n=len(x), home=b[0], b_o=b[1], b_d=b[2],
                         b_o_lo=np.percentile(bb[:, 1], 2.5), b_o_hi=np.percentile(bb[:, 1], 97.5),
                         b_d_lo=np.percentile(bb[:, 2], 2.5), b_d_hi=np.percentile(bb[:, 2], 97.5),
                         d_minus_o=b[2] - b[1], dmo_lo=np.percentile(diff, 2.5), dmo_hi=np.percentile(diff, 97.5),
                         p_d_bigger=(diff > 0).mean(),
                         logit_o=lb[1], logit_d=lb[2], logit_dmo_lo=np.percentile(lbb[:, 2] - lbb[:, 1], 2.5),
                         logit_dmo_hi=np.percentile(lbb[:, 2] - lbb[:, 1], 97.5)))
    tab = pd.DataFrame(rows)
    print("\nmargin = a + b_o * O edge + b_d * D edge   (edges in points per 100, regular season ratings)")
    for r in tab.itertuples():
        print(f"  {r.sample:<15} home {r.home:+.2f}   b_o {r.b_o:.3f} [{r.b_o_lo:.3f}, {r.b_o_hi:.3f}]   "
              f"b_d {r.b_d:.3f} [{r.b_d_lo:.3f}, {r.b_d_hi:.3f}]   d - o {r.d_minus_o:+.3f} "
              f"[{r.dmo_lo:+.3f}, {r.dmo_hi:+.3f}]  (share of bootstraps with D > O: {r.p_d_bigger:.0%})")
    print("\nsame thing on wins (logit), d - o:")
    for r in tab.itertuples():
        print(f"  {r.sample:<15} b_o {r.logit_o:.3f}   b_d {r.logit_d:.3f}   d - o {r.logit_d - r.logit_o:+.3f} "
              f"[{r.logit_dmo_lo:+.3f}, {r.logit_dmo_hi:+.3f}]")

    # champions: won the finals series
    fin = po[po["round"] == 4].copy()
    fin["win_h"] = fin["pts_h"] > fin["pts_a"]
    champs = {}
    for s, f in fin.groupby("season"):
        w = pd.concat([f.loc[f["win_h"], "home"], f.loc[~f["win_h"], "away"]]).value_counts()
        champs[s] = int(w.index[0])
    abbr = pd.concat([pd.read_parquet(RAW_DIR / f"team_games_{y}.parquet")[["TEAM_ID", "TEAM_ABBREVIATION"]]
                      .assign(season=y) for y in SEASONS]).drop_duplicates(["season", "TEAM_ID"])
    abbr = abbr.set_index(["season", "TEAM_ID"])["TEAM_ABBREVIATION"]
    teams["rank_o"] = teams.groupby("season")["o"].rank(ascending=False).astype(int)
    teams["rank_d"] = teams.groupby("season")["d"].rank(ascending=False).astype(int)
    teams["rank_net"] = teams.groupby("season")["net"].rank(ascending=False).astype(int)
    teams["abbr"] = [abbr.get((s, t), "?") for s, t in zip(teams["season"], teams["team"])]
    teams["champion"] = [champs.get(s) == t for s, t in zip(teams["season"], teams["team"])]
    in_po = set(zip(po["season"], po["home"])) | set(zip(po["season"], po["away"]))
    pt = teams[[(s, t) in in_po for s, t in zip(teams["season"], teams["team"])]].copy()
    c = teams[teams["champion"]].sort_values("season")
    print("\nchampions, regular season rank (of 30):")
    print(c.assign(season=c["season"].map(season_label))[["season", "abbr", "o", "d", "rank_o", "rank_d", "rank_net"]]
          .round(1).to_string(index=False))
    print(f"  top-5 offense: {(c['rank_o'] <= 5).sum()} of {len(c)},  top-5 defense: {(c['rank_d'] <= 5).sum()} "
          f"of {len(c)},  top-3 net: {(c['rank_net'] <= 3).sum()} of {len(c)}")

    # among playoff teams: does D share of net predict rounds won, given net?
    wins = pd.concat([po.loc[po["pts_h"] > po["pts_a"], ["season", "home"]].set_axis(["season", "team"], axis=1),
                      po.loc[po["pts_h"] < po["pts_a"], ["season", "away"]].set_axis(["season", "team"], axis=1)])
    pt["po_wins"] = pt.set_index(["season", "team"]).index.map(wins.groupby(["season", "team"]).size()).fillna(0).to_numpy()
    X = np.column_stack([np.ones(len(pt)), pt["o"], pt["d"]])
    b = np.linalg.lstsq(X, pt["po_wins"].to_numpy(), rcond=None)[0]
    print(f"\nplayoff teams (n={len(pt)}): playoff games won = {b[0]:.1f} + {b[1]:.2f} * O + {b[2]:.2f} * D")

    tab.to_parquet(PROCESSED_DIR / "defense.parquet", index=False)
    plot(tab, pt)
