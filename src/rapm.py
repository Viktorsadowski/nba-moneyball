#!/usr/bin/env python3
"""
Step 3: RAPM per season, split into offense and defense.

Every stint becomes two rows, one per team on offense:
  y      = points scored per 100 possessions in that stint
  x      = +1 for the 5 offensive players (offense columns), +1 for the 5 defenders (defense columns),
           plus a home-offense flag
  weight = possessions in the stint

Ridge regression, lambda picked by cross-validation with whole games held out.
Everyone stays in the regression (low-minute guys still share the floor with the good ones),
filtering on minutes only happens when looking at the output.

Sign convention in the output: positive is good for both O and D.
  o_rapm  points per 100 added on offense
  d_rapm  points per 100 saved on defense
  rapm    o_rapm + d_rapm

  python src/rapm.py
  python src/rapm.py --seasons 2025 --alpha 2000     # skip the CV, use a fixed lambda
"""

import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label

ALPHAS = [300, 1000, 2000, 3000, 5000, 10000]
H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]


# ── design matrix ─────────────────────────────────────────────────────────────

def build_rows(st: pd.DataFrame):
    # possessions alternate, so use the average of both sides' estimate as weight for both rows
    st = st[st["poss"] > 0]
    players = np.unique(st[H + A].to_numpy())
    col = {p: i for i, p in enumerate(players)}
    n_p = len(players)

    def side(off_cols, def_cols, pts, is_home):
        off = st[off_cols].to_numpy()
        dfn = st[def_cols].to_numpy()
        n = len(st)
        rows = np.repeat(np.arange(n), 5)
        o_idx = np.vectorize(col.get)(off).ravel()
        d_idx = np.vectorize(col.get)(dfn).ravel() + n_p
        r = np.concatenate([rows, rows, np.arange(n)])
        c = np.concatenate([o_idx, d_idx, np.full(n, 2 * n_p)])
        v = np.concatenate([np.ones(10 * n), np.full(n, float(is_home))])
        X = sparse.csr_matrix((v, (r, c)), shape=(n, 2 * n_p + 1))
        y = 100 * st[pts].to_numpy() / st["poss"].to_numpy()
        return X, y

    Xh, yh = side(H, A, "pts_h", True)
    Xa, ya = side(A, H, "pts_a", False)
    X = sparse.vstack([Xh, Xa]).tocsr()
    y = np.concatenate([yh, ya])
    w = np.concatenate([st["poss"].to_numpy()] * 2)
    groups = np.concatenate([st["game_id"].to_numpy()] * 2)  # both rows of a stint land in the same fold
    return X, y, w, groups, players


# ── lambda by cross-validation ────────────────────────────────────────────────

def pick_alpha(X, y, w, groups) -> tuple[float, dict]:
    scores = {}
    for a in ALPHAS:
        errs, wsum = 0.0, 0.0
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr])
            errs += np.sum(w[te] * (y[te] - m.predict(X[te])) ** 2)
            wsum += w[te].sum()
        scores[a] = errs / wsum
    best = min(scores, key=scores.get)
    return best, scores


# ── one season ────────────────────────────────────────────────────────────────

def fit_season(year: int, alpha: float | None) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED_DIR / f"stints_{year}.parquet")
    X, y, w, groups, players = build_rows(st)

    if alpha is None:
        alpha, scores = pick_alpha(X, y, w, groups)
        # baseline = predict league average for every row, to see how much the model actually adds
        base = np.sum(w * (y - np.average(y, weights=w)) ** 2) / w.sum()
        print(f"\n{season_label(year)}: lambda {alpha}  cv mse {scores[alpha]:.1f} (baseline {base:.1f})")
    else:
        print(f"\n{season_label(year)}: lambda {alpha} (fixed)")

    m = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    n_p = len(players)
    coef = m.coef_
    print(f"  home court: {coef[-1]:+.2f} pts/100   league avg ortg: {m.intercept_:.1f}")

    out = pd.DataFrame({
        "season": year,
        "player_id": players,
        "o_rapm": coef[:n_p],
        "d_rapm": -coef[n_p:2 * n_p],  # flipped: allowing fewer points = positive
    })
    out["rapm"] = out["o_rapm"] + out["d_rapm"]
    out["lambda"] = alpha

    # minutes and possessions from our own stints, name + team from the roster if we have it
    long = st.melt(id_vars=["dur", "poss"], value_vars=H + A, value_name="player_id")
    usage = long.groupby("player_id").agg(minutes=("dur", "sum"), poss=("poss", "sum"))
    usage["minutes"] /= 60
    out = out.merge(usage, on="player_id", how="left")

    rpath = RAW_DIR / f"roster_{year}.parquet"
    if rpath.exists():
        r = pd.read_parquet(rpath)
        names = r.groupby("PLAYER_ID")["PLAYER_NAME"].first()
        # most games played for a team = his main team that season
        team = r.groupby(["PLAYER_ID", "TEAM_ID"]).size().reset_index().sort_values(0).drop_duplicates("PLAYER_ID", keep="last")
        out["name"] = out["player_id"].map(names)
        out["team_id"] = out["player_id"].map(team.set_index("PLAYER_ID")["TEAM_ID"])

    show = out[out["minutes"] >= 1000].sort_values("rapm", ascending=False)
    cols = [c for c in ["name", "minutes", "o_rapm", "d_rapm", "rapm"] if c in show]
    print("  top 10 (1000+ min):")
    print(show[cols].head(10).round(2).to_string(index=False))
    return out


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    res = pd.concat([fit_season(y, args.alpha) for y in args.seasons], ignore_index=True)
    out = PROCESSED_DIR / "rapm.parquet"
    # rerunning a few seasons only replaces those, keeps the rest of the file
    if out.exists():
        old = pd.read_parquet(out)
        res = pd.concat([old[~old["season"].isin(args.seasons)], res], ignore_index=True)
    res.sort_values(["season", "rapm"], ascending=[True, False]).to_parquet(out, index=False)
    print(f"\nsaved {len(res):,} player-seasons -> {out}")
