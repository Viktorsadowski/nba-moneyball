#!/usr/bin/env python3
"""
Positions: does it matter who plays where, on top of what the players are worth?

Team strength everywhere else is just the sum of player values, so a lineup of five centers counts the same as
a normal one. This checks if that's wrong, and by how much.

  roles       from the box score, not the listed position (a "forward" can be a big or a wing): per 36 minutes
              rebounds, blocks, assists and 3-point attempts, k-means into 3 groups (guard, wing, big), fitted on
              all player-seasons with 500+ minutes weighted by minutes. a player's role = his last 3 seasons'
              stats together. no stats (rookies): nba.com's listed position if rosters.py saved it, else wing
  test        every stint 2013-14 on: margin per 100 - what the ten players' RAPM says it should be, explained by
              how many bigs and guards each side has on the floor. RAPM of that same season, so the players' usual
              lineups are already in their numbers, what's left is the lineup shape itself
  result      effect per 100 of 0 / 1 / 3+ bigs and 0 / 1 / 3+ guards vs the usual 2 of each, with bootstrap
              intervals over games. season_sim + trade_search turn it into a penalty for rosters that can't
              field normal lineups

Output: data/processed/roles.parquet (player_id, season, role)
        data/processed/position_effects.json

  python src/positions.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS

H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]
FEATS = ["reb36", "blk36", "ast36", "fg3a36"]
FIRST_TEST = 2013


def box() -> pd.DataFrame:
    p = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet") for y in SEASONS])
    p = p.groupby(["SEASON", "PLAYER_ID"])[["MIN", "REB", "BLK", "AST", "FG3A"]].sum().reset_index()
    p.columns = ["season", "player_id", "min", "reb", "blk", "ast", "fg3a"]
    return p


def kmeans(X: np.ndarray, w: np.ndarray, k: int = 3, iters: int = 100, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([np.average(X[lab == j], axis=0, weights=w[lab == j]) for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def roles(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """role per player per season, from that season + the 2 before (so it's stable and known before the next)"""
    rows = []
    for s in sorted(p["season"].unique()):
        w = p[p["season"].between(s - 2, s)].groupby("player_id")[["min", "reb", "blk", "ast", "fg3a"]].sum()
        w = w[w["min"] >= 300]
        for f, col in zip(FEATS, ["reb", "blk", "ast", "fg3a"]):
            w[f] = w[col] / w["min"] * 36
        w["season"] = s
        rows.append(w.reset_index())
    r = pd.concat(rows, ignore_index=True)
    fit = r[r["min"] >= 500]
    mu, sd = fit[FEATS].mean(), fit[FEATS].std()
    C = kmeans(((fit[FEATS] - mu) / sd).to_numpy(), fit["min"].to_numpy())
    # name the clusters: most rebounds+blocks = big, most assists = guard, the other one = wing
    big = int(np.argmax(C[:, 0] + C[:, 1]))
    guard = int(np.argmax(np.where(np.arange(3) == big, -np.inf, C[:, 2])))
    names = {big: "big", guard: "guard", 3 - big - guard: "wing"}
    lab = np.argmin(((((r[FEATS] - mu) / sd).to_numpy()[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    r["role"] = [names[x] for x in lab]
    cents = {names[j]: dict(zip(FEATS, (C[j] * sd.to_numpy() + mu.to_numpy()).round(1))) for j in range(3)}
    return r[["player_id", "season", "role"]], cents


def stint_table(y: int, role: pd.Series, rapm: pd.Series) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet",
                         columns=["game_id", "pts_h", "pts_a", "poss", *H, *A])
    st = st[st["poss"] > 0]
    out = pd.DataFrame({"game_id": st["game_id"], "poss": st["poss"],
                        "m": (st["pts_h"] - st["pts_a"]) / st["poss"] * 100})
    for side, cols in (("h", H), ("a", A)):
        ids = st[cols].to_numpy()
        rl = np.vectorize(lambda p: role.get(p, "wing"))(ids)
        out[f"big_{side}"] = (rl == "big").sum(1)
        out[f"grd_{side}"] = (rl == "guard").sum(1)
        out[f"rapm_{side}"] = np.vectorize(lambda p: rapm.get(p, 0.0))(ids).sum(1)
    out["season"] = y
    return out


def dummies(n: np.ndarray, levels=(0, 1, 3)) -> np.ndarray:
    # 2 = the usual, the baseline. 3 means 3 or more
    return np.column_stack([(n == 0), (n == 1), (n >= 3)]).astype(float)


if __name__ == "__main__":
    p = box()
    rl, cents = roles(p)
    rl.to_parquet(PROCESSED_DIR / "roles.parquet", index=False)
    print("roles (per 36 minutes, cluster centers):")
    for k, v in cents.items():
        print(f"  {k:5s} {v}")
    last = rl[rl["season"] == max(SEASONS)]["role"].value_counts()
    print("  last season:", last.to_dict())

    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    tabs = []
    for y in range(FIRST_TEST, max(SEASONS) + 1):
        role = rl[rl["season"] == y].set_index("player_id")["role"]
        rp = rapm[rapm["season"] == y].set_index("player_id")["rapm"]
        tabs.append(stint_table(y, role, rp))
    t = pd.concat(tabs, ignore_index=True)
    # what the ten players' RAPM doesn't explain, home court taken out with the mean
    t["resid"] = t["m"] - (t["rapm_h"] - t["rapm_a"])
    t["resid"] -= np.average(t["resid"], weights=t["poss"])
    X = np.column_stack([dummies(t["big_h"].to_numpy()) - dummies(t["big_a"].to_numpy()),
                         dummies(t["grd_h"].to_numpy()) - dummies(t["grd_a"].to_numpy())])
    names = ["0 bigs", "1 big", "3+ bigs", "0 guards", "1 guard", "3+ guards"]
    w = t["poss"].to_numpy()
    sw = np.sqrt(w)
    b = np.linalg.lstsq(X * sw[:, None], t["resid"].to_numpy() * sw, rcond=None)[0]
    # bootstrap over games (stints in a game aren't independent)
    games = t["game_id"].to_numpy()
    uniq, inv = np.unique(games, return_inverse=True)
    rng = np.random.default_rng(5)
    XtX = [None]
    bs = []
    # per-game sums make the bootstrap cheap: X'WX and X'Wy per game
    G = len(uniq)
    xwx = np.zeros((G, X.shape[1], X.shape[1]))
    xwy = np.zeros((G, X.shape[1]))
    Xw = X * w[:, None]
    np.add.at(xwx, inv, Xw[:, :, None] * X[:, None, :])
    np.add.at(xwy, inv, Xw * t["resid"].to_numpy()[:, None])
    for _ in range(300):
        c = np.bincount(rng.integers(0, G, G), minlength=G)
        bs.append(np.linalg.solve(np.tensordot(c, xwx, 1) + 1e-9 * np.eye(X.shape[1]), c @ xwy))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    share = {}
    for side in ("h",):
        for k, col in (("big", "big_h"), ("grd", "grd_h")):
            vc = t.groupby(col)["poss"].sum() / t["poss"].sum()
            share[k] = {int(i): float(v) for i, v in vc.items()}
    print(f"\n{len(t):,} stints {FIRST_TEST}-{max(SEASONS)}, points per 100 vs the usual 2 bigs / 2 guards, "
          "on top of the players' RAPM:")
    for n, bb, l, h in zip(names, b, lo, hi):
        print(f"  {n:10s} {bb:+.2f}  [{l:+.2f}, {h:+.2f}]")
    print("share of possessions by number of bigs on the floor:",
          {k: f"{v:.1%}" for k, v in share["big"].items()})
    print("by number of guards:", {k: f"{v:.1%}" for k, v in share["grd"].items()})
    # ── from a roster to lineup shapes ──
    # a team's average number of bigs (guards) on the floor -> share of its possessions with 0 bigs, 3+ bigs,
    # 0 guards, 1 guard. quadratic, fitted over team-seasons. then a roster's lineup effect = effects x shares,
    # minus the league average, so it only says "worse / better shaped than a normal team"
    t["team_h"] = np.nan
    tt = []
    for y in range(FIRST_TEST, max(SEASONS) + 1):
        st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet", columns=["home_team", "away_team", "poss"])
        st = st[st["poss"] > 0]
        tt.append(pd.DataFrame({"home": st["home_team"].to_numpy(), "away": st["away_team"].to_numpy()}))
    teams = pd.concat(tt, ignore_index=True)
    per = []
    for side, tcol in (("h", "home"), ("a", "away")):
        per.append(pd.DataFrame({"season": t["season"].to_numpy(), "team": teams[tcol].to_numpy(),
                                 "poss": t["poss"].to_numpy(), "nb": t[f"big_{side}"].to_numpy(),
                                 "ng": t[f"grd_{side}"].to_numpy()}))
    per = pd.concat(per)
    g = per.groupby(["season", "team"]).apply(lambda x: pd.Series({
        "Eb": np.average(x.nb, weights=x.poss), "Eg": np.average(x.ng, weights=x.poss),
        "p0b": np.average(x.nb == 0, weights=x.poss), "p3b": np.average(x.nb >= 3, weights=x.poss),
        "p0g": np.average(x.ng == 0, weights=x.poss), "p1g": np.average(x.ng == 1, weights=x.poss)}),
        include_groups=False)
    polys = {k: list(np.polyfit(g[x], g[k], 2)) for k, x in (("p0b", "Eb"), ("p3b", "Eb"), ("p0g", "Eg"),
                                                              ("p1g", "Eg"))}
    eff = dict(zip(names, map(float, b)))
    avg = float(eff["0 bigs"] * g["p0b"].mean() + eff["3+ bigs"] * g["p3b"].mean()
                + eff["0 guards"] * g["p0g"].mean() + eff["1 guard"] * g["p1g"].mean())
    print(f"\nteam-seasons: average bigs on the floor {g['Eb'].mean():.2f} (sd {g['Eb'].std():.2f}), "
          f"guards {g['Eg'].mean():.2f} (sd {g['Eg'].std():.2f}). league-average lineup effect {avg:+.2f} per 100")
    (PROCESSED_DIR / "position_effects.json").write_text(json.dumps(
        dict(effect=eff, lo=dict(zip(names, map(float, lo))), hi=dict(zip(names, map(float, hi))), share=share,
             centers=cents, polys=polys, avg=avg, Eb_range=[float(g["Eb"].min()), float(g["Eb"].max())],
             Eg_range=[float(g["Eg"].min()), float(g["Eg"].max())]), indent=1))
