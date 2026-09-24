#!/usr/bin/env python3
"""
Step 3b: RAPM with a box-score prior.

Plain RAPM shrinks every player towards 0 (= league average). Players who always share the floor, or
play few minutes, end up close to 0 whatever they did. With a prior the ridge shrinks towards what his
box score says instead:

  1. box prior     box stats per 100 possessions (points, shots, free throws, threes, assists, turnovers,
                   rebounds, steals, blocks, fouls, minutes per game) -> O-RAPM and D-RAPM, weighted linear
                   fit on plain RAPM. for season s the weights are fitted on all OTHER seasons except s and
                   s+1, so the prior never saw the season it's used for, or the one the forward test predicts
  2. priored ridge  minimize  |y - X b|^2 + lambda |b - b_prior|^2. same as plain ridge on y - X b_prior,
                   then add b_prior back. lambda picked by CV again (the best one changes with a prior)
  3. forward test  does it predict better? player values from season s -> points per 100 in every stint
                   of season s+1 (possession-weighted error). plain vs prior vs box only

The plain version is kept as rapm_plain.parquet. rapm.parquet becomes the priored one (same columns
+ o_prior, d_prior, prior=True), so everything downstream picks it up without changes.

  python src/rapm.py          # plain first
  python src/box_prior.py     # then this, then the rest of the pipeline from aging.py on
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from rapm import A, H, build_rows, pick_alpha

STATS = ["PTS", "FGA", "FTA", "FG3A", "FG3M", "AST", "TOV", "OREB", "DREB", "STL", "BLK", "PF"]
GHOST_POSS = 300        # rates of low-possession guys pulled towards the league rate, like the ghost minutes
MIN_FIT_POSS = 500      # only players with real possessions teach the prior


# ── 1. box prior ──────────────────────────────────────────────────────────────

def box_features(rapm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in SEASONS:
        p = pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["PLAYER_ID", "GP", "MIN", *STATS]]
        r = rapm[rapm["season"] == y][["player_id", "poss", "o_rapm", "d_rapm", "minutes"]]
        d = r.merge(p, left_on="player_id", right_on="PLAYER_ID", how="left").drop(columns="PLAYER_ID")
        d[STATS] = d[STATS].fillna(0)
        poss = d["poss"].clip(lower=1)
        for s in STATS:
            lg = d[s].sum() / poss.sum()
            d[f"{s.lower()}100"] = 100 * (d[s] + lg * GHOST_POSS) / (poss + GHOST_POSS)
        d["mpg"] = (d["MIN"] / d["GP"].clip(lower=1)).fillna(0)
        d["season"] = y
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


FEATS = [f"{s.lower()}100" for s in STATS] + ["mpg"]


def fit_prior(f: pd.DataFrame) -> pd.DataFrame:
    """o_prior, d_prior per player-season, weights never fitted on s or s+1."""
    out = []
    for s in SEASONS:
        tr = f[~f["season"].isin([s, s + 1]) & (f["poss"] >= MIN_FIT_POSS)]
        mu, sd = tr[FEATS].mean(), tr[FEATS].std()
        Xtr = (tr[FEATS] - mu) / sd
        w = tr["poss"].clip(upper=6000)
        te = f[f["season"] == s]
        Xte = (te[FEATS] - mu) / sd
        res = te[["season", "player_id"]].copy()
        for side in ("o", "d"):
            m = Ridge(alpha=1.0).fit(Xtr, tr[f"{side}_rapm"], sample_weight=w)
            res[f"{side}_prior"] = m.predict(Xte)
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ── 2. priored ridge ──────────────────────────────────────────────────────────

def fit_season_prior(year: int, prior: pd.DataFrame, plain: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED_DIR / f"stints_{year}.parquet")
    X, y, w, groups, players = build_rows(st)
    n_p = len(players)
    pr = prior[prior["season"] == year].set_index("player_id").reindex(players).fillna(0)
    # defense columns are points allowed in the regression, so the prior goes in with a minus
    b0 = np.concatenate([pr["o_prior"].to_numpy(), -pr["d_prior"].to_numpy(), [0.0]])
    y_off = y - X @ b0
    alpha, scores = pick_alpha(X, y_off, w, groups)
    m = Ridge(alpha=alpha).fit(X, y_off, sample_weight=w)
    coef = m.coef_ + b0
    out = pd.DataFrame({"season": year, "player_id": players, "o_rapm": coef[:n_p], "d_rapm": -coef[n_p:2 * n_p]})
    out["rapm"] = out["o_rapm"] + out["d_rapm"]
    out["lambda"] = alpha
    out["o_prior"], out["d_prior"] = pr["o_prior"].to_numpy(), pr["d_prior"].to_numpy()
    # minutes, possessions, name, team: same as the plain run
    keep = plain[plain["season"] == year].set_index("player_id")[["minutes", "poss", "name", "team_id"]]
    out = out.join(keep, on="player_id")
    print(f"  {season_label(year)}: lambda {alpha:g}  cv mse {scores[alpha]:.1f}")
    return out


# ── 3. forward test ───────────────────────────────────────────────────────────

def forward_error(values: pd.DataFrame, col_o: str, col_d: str) -> dict:
    """season s values -> season s+1 stints. unknown players = 0. league level + home court from s+1 itself,
    same for every method, so only the player values make the difference."""
    num, den, base_num, per = 0.0, 0.0, 0.0, {}
    for s in list(SEASONS)[:-1]:
        v = values[values["season"] == s].set_index("player_id")
        st = pd.read_parquet(PROCESSED_DIR / f"stints_{s + 1}.parquet")
        st = st[st["poss"] > 0]
        o = v[col_o].to_dict()
        d = v[col_d].to_dict()
        P = st[H + A].to_numpy()
        vo = np.vectorize(lambda p: o.get(p, 0.0))(P)
        vd = np.vectorize(lambda p: d.get(p, 0.0))(P)
        # home on offense: + home O - away D, away on offense: + away O - home D
        pred_h = vo[:, :5].sum(1) - vd[:, 5:].sum(1)
        pred_a = vo[:, 5:].sum(1) - vd[:, :5].sum(1)
        yh = 100 * st["pts_h"].to_numpy() / st["poss"].to_numpy()
        ya = 100 * st["pts_a"].to_numpy() / st["poss"].to_numpy()
        w = st["poss"].to_numpy()
        yy = np.concatenate([yh, ya])
        pp = np.concatenate([pred_h, pred_a])
        ww = np.concatenate([w, w])
        home = np.concatenate([np.ones(len(yh)), np.zeros(len(ya))])
        # level + home fitted on the residual, identical treatment for every method
        Z = np.column_stack([np.ones(len(yy)), home])
        b = np.linalg.lstsq(Z * np.sqrt(ww)[:, None], (yy - pp) * np.sqrt(ww), rcond=None)[0]
        err = yy - pp - Z @ b
        b0 = np.linalg.lstsq(Z * np.sqrt(ww)[:, None], yy * np.sqrt(ww), rcond=None)[0]
        per[s] = np.sum(ww * err ** 2) / ww.sum()
        num += np.sum(ww * err ** 2)
        base_num += np.sum(ww * (yy - Z @ b0) ** 2)
        den += ww.sum()
    return {"mse": num / den, "base": base_num / den, "gain": 1 - num / base_num, "per": per}


if __name__ == "__main__":
    cur = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    plain_path = PROCESSED_DIR / "rapm_plain.parquet"
    # rapm.py always writes the plain version into rapm.parquet. if that's what's there, it's the fresh one
    if "prior" not in cur.columns or not cur["prior"].any():
        cur.to_parquet(plain_path, index=False)
    plain = pd.read_parquet(plain_path)

    f = box_features(plain)
    prior = fit_prior(f)
    fit = f.merge(prior, on=["season", "player_id"])
    big = fit[fit["poss"] >= MIN_FIT_POSS]
    for side in ("o", "d"):
        r = np.corrcoef(big[f"{side}_prior"], big[f"{side}_rapm"])[0, 1]
        print(f"box prior vs plain {side.upper()}-RAPM (out of season, {MIN_FIT_POSS}+ poss): r = {r:.2f}")

    print("\npriored RAPM per season:")
    res = pd.concat([fit_season_prior(y, prior, plain) for y in SEASONS], ignore_index=True)
    res["prior"] = True

    print("\nforward test: season s values -> points per 100 in every stint of season s+1")
    tests = {"plain RAPM": (plain, "o_rapm", "d_rapm"),
             "box prior only": (res, "o_prior", "d_prior"),
             "RAPM with box prior": (res, "o_rapm", "d_rapm")}
    errs = {}
    for name, (v, co, cd) in tests.items():
        e = errs[name] = forward_error(v, co, cd)
        print(f"  {name:<20} mse {e['mse']:.2f}   (level-only baseline {e['base']:.2f}, explains {e['gain']:.3%})")
    # stint outcomes are mostly noise, so the share explained is tiny. what counts: better than plain, every season?
    wins = sum(errs["RAPM with box prior"]["per"][s] < errs["plain RAPM"]["per"][s] for s in errs["plain RAPM"]["per"])
    print(f"  prior beats plain in {wins} of {len(errs['plain RAPM']['per'])} seasons, "
          f"{errs['RAPM with box prior']['gain'] / errs['plain RAPM']['gain'] - 1:+.0%} more of the variance explained")

    show = res[(res["season"] == max(SEASONS)) & (res["minutes"] >= 1000)].sort_values("rapm", ascending=False)
    print(f"\ntop 10 {season_label(max(SEASONS))} with the prior (1000+ min):")
    print(show[["name", "minutes", "o_prior", "d_prior", "o_rapm", "d_rapm", "rapm"]].head(10).round(2)
          .to_string(index=False))
    res.sort_values(["season", "rapm"], ascending=[True, False]).to_parquet(PROCESSED_DIR / "rapm.parquet", index=False)
    print(f"\nsaved -> {PROCESSED_DIR / 'rapm.parquet'} (plain kept in rapm_plain.parquet)")
