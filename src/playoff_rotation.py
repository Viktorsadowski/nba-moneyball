#!/usr/bin/env python3
"""
Playoff rotations: in the playoffs the best players play more and the bench plays less, so depth is worth
less and stars more than in the regular season. How much, and does it predict playoff games better?

  shares      every playoff team since 2010-11: rank its players by regular-season minutes per game (with
              that team), then each rank's share of the team's minutes, regular season vs playoffs
              (stints_po_*, playoff_stints.py). 5 x 48 = 240 minutes a game to hand out
  strength    a team's net per 100 = sum of value x minute share, with the regular-season shares or the
              playoff shares, players ranked by value (the coach plays his best guys). value = our blended
              RAPM as of that season (value.py)
  test        playoff game margins ~ home court + (home - away) strength, with each set of shares: which one
              explains the playoff games better

Output: data/processed/playoff_shares.parquet (minute share by rank, regular vs playoffs)  (season_sim.py)
        figures/playoff_rotation.png

  python src/playoff_rotation.py      # after playoff_stints.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS

RANKS = 13
H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]


def minutes(st: pd.DataFrame) -> pd.DataFrame:
    """minutes per player per team per game from stints"""
    long = pd.concat([
        st.melt(id_vars=["game_id", "home_team", "dur"], value_vars=H, value_name="pid")
          .rename(columns={"home_team": "team"}),
        st.melt(id_vars=["game_id", "away_team", "dur"], value_vars=A, value_name="pid")
          .rename(columns={"away_team": "team"}),
    ])
    return long.groupby(["team", "pid", "game_id"])["dur"].sum().div(60).rename("min").reset_index()


def season_shares(y: int) -> pd.DataFrame | None:
    po_path = PROCESSED_DIR / f"stints_po_{y}.parquet"
    if not po_path.exists():
        return None
    reg = minutes(pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet", columns=["game_id", "home_team",
                                                                                 "away_team", "dur", *H, *A]))
    po = minutes(pd.read_parquet(po_path, columns=["game_id", "home_team", "away_team", "dur", *H, *A]))
    rows = []
    for team, p in po.groupby("team"):
        r = reg[reg["team"] == team]
        n_reg, n_po = r["game_id"].nunique(), p["game_id"].nunique()
        mpg_reg = r.groupby("pid")["min"].sum() / n_reg
        mpg_po = p.groupby("pid")["min"].sum() / n_po
        m = pd.DataFrame({"reg": mpg_reg, "po": mpg_po}).fillna(0)
        m = m.sort_values("reg", ascending=False).head(RANKS)
        m["rank"] = np.arange(1, len(m) + 1)
        m["season"], m["team"], m["games_po"] = y, team, n_po
        rows.append(m.reset_index(names="pid"))
    return pd.concat(rows)


def team_strength(vals: np.ndarray, shares: np.ndarray, repl: float) -> float:
    """value-ranked players get the rank shares, whatever the shares don't cover goes to replacement"""
    v = np.sort(vals)[::-1][:len(shares)]
    s = shares[:len(v)]
    return float(np.sum(v * s) + repl * (1 - s.sum())) * 5


def plot(sh: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    x = sh["rank"].to_numpy()
    ax.bar(x - 0.2, sh["reg_mpg"], width=0.4, color=blue, label="regular season")
    ax.bar(x + 0.2, sh["po_mpg"], width=0.4, color=orange, label="playoffs")
    ax.set_xticks(x)
    ax.set_xlabel("player's rank on his team by regular-season minutes", color=muted, fontsize=8.5)
    ax.set_ylabel("minutes per game", color=muted, fontsize=8.5)
    ax.set_title("Playoff rotations get shorter", loc="left", fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, lw=0.6)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "playoff_rotation.png", facecolor=surface)


if __name__ == "__main__":
    import json
    d = pd.concat([x for x in (season_shares(y) for y in SEASONS) if x is not None])
    sh = d.groupby("rank").agg(reg_mpg=("reg", "mean"), po_mpg=("po", "mean")).reset_index()
    sh["reg_share"], sh["po_share"] = sh["reg_mpg"] / 240, sh["po_mpg"] / 240
    print(f"{d.groupby(['season', 'team']).ngroups} playoff teams. minutes per game by rank on the team:")
    print(sh.round(3).to_string(index=False))
    top5 = sh.loc[sh["rank"] <= 5]
    print(f"top 5 play {top5['reg_share'].sum():.0%} of the minutes in the regular season, "
          f"{top5['po_share'].sum():.0%} in the playoffs. ranks 9+: {sh.loc[sh['rank'] >= 9, 'reg_share'].sum():.0%} "
          f"-> {sh.loc[sh['rank'] >= 9, 'po_share'].sum():.0%}")
    sh.to_parquet(PROCESSED_DIR / "playoff_shares.parquet", index=False)

    # ── test on playoff games ──
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    repl_raw = (prm["repl"] - prm["c"]) / prm["a"]
    val = pd.read_parquet(PROCESSED_DIR / "value.parquet")
    games = pd.read_parquet(RAW_DIR / "playoff_games.parquet")
    rs, ps = sh["reg_share"].to_numpy(), sh["po_share"].to_numpy()
    rows = []
    for y in SEASONS:
        # a team's players = who played for it in the regular season, value as of the end of that season
        reg = minutes(pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet",
                                      columns=["game_id", "home_team", "away_team", "dur", *H, *A]))
        tot = reg.groupby(["team", "pid"])["min"].sum().reset_index()
        tot = tot.sort_values("min").drop_duplicates("pid", keep="last")        # his last / main team
        v = val[val["season"] == y].set_index("player_id")["val"]
        tot["val"] = tot["pid"].map(v).fillna(repl_raw)
        strength = {}
        for team, g in tot.groupby("team"):
            strength[team] = (team_strength(g["val"].to_numpy(), rs, repl_raw),
                              team_strength(g["val"].to_numpy(), ps, repl_raw))
        for r in games[games["season"] == y].itertuples():
            if r.home in strength and r.away in strength:
                rows.append(dict(season=y, m=r.pts_h - r.pts_a,
                                 d_reg=strength[r.home][0] - strength[r.away][0],
                                 d_po=strength[r.home][1] - strength[r.away][1]))
    g = pd.DataFrame(rows).dropna()
    print(f"\n{len(g)} playoff games: margin ~ home court + strength gap")
    res = {}
    for k in ("d_reg", "d_po"):
        X = np.column_stack([np.ones(len(g)), g[k]])
        b, *_ = np.linalg.lstsq(X, g["m"], rcond=None)
        e = g["m"] - X @ b
        # leave one season out, so the two versions are compared out of sample
        cv = []
        for y in g["season"].unique():
            tr, te = g["season"] != y, g["season"] == y
            bb, *_ = np.linalg.lstsq(X[tr.to_numpy()], g["m"][tr], rcond=None)
            cv.append(((g["m"][te] - X[te.to_numpy()] @ bb) ** 2).to_numpy())
        res[k] = float(np.sqrt(np.concatenate(cv).mean()))
        print(f"  {'regular-season shares' if k == 'd_reg' else 'playoff shares':22s} slope {b[1]:.2f}, "
              f"home {b[0]:+.2f}, R2 {1 - e.var() / g['m'].var():.3f}, out-of-sample RMSE {res[k]:.3f}")
    diff = res["d_reg"] - res["d_po"]
    print(f"  playoff shares {'better' if diff > 0 else 'worse'} by {abs(diff):.3f} points of RMSE")
    plot(sh)
