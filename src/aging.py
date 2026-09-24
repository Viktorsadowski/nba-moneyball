#!/usr/bin/env python3
"""
Step 7: aging curve. How much does a player's O and D RAPM change with age.

Two estimates, they should roughly agree:

  fixed effects  rapm_it = talent_i + age_effect(age_it), weighted by minutes.
                 each player's own level is soaked up by talent_i, so what's left is the age shape.
                 main estimate.
  delta method   average change from age a to a+1 over players with 500+ min in both seasons,
                 weighted by the smaller of the two minute totals, then added up. the classic
                 baseball way, biased by survivors (guys who fall off a cliff stop playing and
                 never show up as a big negative delta), so it's just the sanity check.

Both get smoothed with a weighted quadratic in age (enough for a hump shape) and set to 0 at the peak.

Output
  data/processed/aging_curve.parquet   age, o_curve, d_curve, curve (points per 100, relative to peak)
  figures/aging_curve.png

  python src/aging.py
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS

AGE_MIN, AGE_MAX = 19, 38   # few players outside this, they get clipped into the ends
DELTA_MIN = 500


def load() -> pd.DataFrame:
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    ages = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["SEASON", "PLAYER_ID", "AGE"]]
                      for y in SEASONS if (RAW_DIR / f"players_{y}.parquet").exists()])
    ages.columns = ["season", "player_id", "age"]
    d = rapm.merge(ages, on=["season", "player_id"], how="inner")
    d["age"] = d["age"].clip(AGE_MIN, AGE_MAX).astype(int)
    return d[d["minutes"] > 0]


# ── fixed effects ─────────────────────────────────────────────────────────────

def fixed_effects(d: pd.DataFrame, col: str) -> pd.Series:
    pids = {p: i for i, p in enumerate(d["player_id"].unique())}
    ages = list(range(AGE_MIN, AGE_MAX + 1))
    n, n_p = len(d), len(pids)
    rows = np.arange(n)
    p_idx = d["player_id"].map(pids).to_numpy()
    a_idx = n_p + (d["age"].to_numpy() - AGE_MIN)
    X = sparse.csr_matrix((np.ones(2 * n), (np.concatenate([rows, rows]), np.concatenate([p_idx, a_idx]))),
                          shape=(n, n_p + len(ages)))
    # light ridge so a 30-minute guy doesn't get a crazy talent term, the age terms have
    # thousands of rows each so the penalty barely touches them
    m = Ridge(alpha=1.0).fit(X, d[col].to_numpy(), sample_weight=d["minutes"].to_numpy() / 1000)
    return pd.Series(m.coef_[n_p:], index=ages)


# ── delta method ──────────────────────────────────────────────────────────────

def delta_method(d: pd.DataFrame, col: str) -> pd.Series:
    a = d[["player_id", "season", "age", "minutes", col]]
    b = a.assign(season=a["season"] - 1)  # next season, shifted back so it lines up
    p = a.merge(b, on=["player_id", "season"], suffixes=("", "_next"))
    p = p[(p["minutes"] >= DELTA_MIN) & (p["minutes_next"] >= DELTA_MIN)]
    p["w"] = np.minimum(p["minutes"], p["minutes_next"])
    p["delta"] = p[f"{col}_next"] - p[col]
    step = p.groupby("age").apply(lambda g: np.average(g["delta"], weights=g["w"]), include_groups=False)
    step = step.reindex(range(AGE_MIN, AGE_MAX)).fillna(0)
    # value at age a = sum of steps before a
    return pd.Series(np.concatenate([[0], step.cumsum().to_numpy()]), index=range(AGE_MIN, AGE_MAX + 1))


# ── smoothing ─────────────────────────────────────────────────────────────────

def smooth(raw: pd.Series, weight: pd.Series) -> pd.Series:
    ages = raw.index.to_numpy()
    coef = np.polyfit(ages, raw.to_numpy(), deg=2, w=np.sqrt(weight.reindex(ages).fillna(1).to_numpy()))
    fit = np.polyval(coef, ages)
    return pd.Series(fit - fit.max(), index=ages)


def plot(curve: pd.DataFrame, raw_total: pd.Series, delta_total: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    blue, orange = "#2a78d6", "#eb6834"

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    a = curve["age"]
    ax.plot(a, curve["o_curve"], color=blue, lw=2, label="Offense")
    ax.plot(a, curve["d_curve"], color=orange, lw=2, label="Defense")
    ax.plot(a, curve["curve"], color=ink, lw=2.4, label="Total")
    ax.scatter(raw_total.index, raw_total - raw_total.max(), s=22, color=ink, alpha=0.35, zorder=3,
               label="Total, unsmoothed")
    ax.plot(delta_total.index, delta_total - delta_total.max(), color=muted, lw=1.2, ls="--",
            label="Total, delta method")

    # direct labels at the right end
    for col, c, txt in (("o_curve", blue, "Offense"), ("d_curve", orange, "Defense"), ("curve", ink, "Total")):
        ax.annotate(txt, (a.iloc[-1], curve[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=ink)

    peak = int(a[curve["curve"].idxmax()])
    ax.axvline(peak, color=grid, lw=1, zorder=0)
    ax.annotate(f"peak {peak}", (peak, 0), xytext=(4, 6), textcoords="offset points", fontsize=9, color=muted)

    ax.set_xlabel("Age", color=muted)
    ax.set_ylabel("RAPM vs peak (points per 100 possessions)", color=muted)
    ax.set_title("NBA aging curve, 2010-11 to 2025-26", loc="left", color=ink, fontsize=12)
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.28))
    ax.set_xlim(AGE_MIN - 0.5, AGE_MAX + 2)
    ax.set_xticks(range(AGE_MIN + 1, AGE_MAX + 1, 2))
    fig.tight_layout()

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "aging_curve.png", facecolor=surface)
    print(f"\nplot -> {out / 'aging_curve.png'}")


if __name__ == "__main__":
    d = load()
    mins_by_age = d.groupby("age")["minutes"].sum()

    fe = {c: fixed_effects(d, c) for c in ("o_rapm", "d_rapm")}
    curve = pd.DataFrame({"age": range(AGE_MIN, AGE_MAX + 1)})
    curve["o_curve"] = smooth(fe["o_rapm"], mins_by_age).to_numpy()
    curve["d_curve"] = smooth(fe["d_rapm"], mins_by_age).to_numpy()
    # total = O + D, re-zeroed at its own peak (O and D don't peak at the same age)
    tot = curve["o_curve"] + curve["d_curve"]
    curve["curve"] = tot - tot.max()
    curve.to_parquet(PROCESSED_DIR / "aging_curve.parquet", index=False)

    raw_total = fe["o_rapm"] + fe["d_rapm"]
    delta_total = delta_method(d, "rapm")

    print(f"{len(d):,} player-seasons, {d['player_id'].nunique():,} players\n")
    show = curve.assign(minutes=curve["age"].map(mins_by_age).fillna(0).astype(int),
                        delta_method=(delta_total - delta_total.max()).to_numpy())
    print(show.round(2).to_string(index=False))
    for c, n in (("o_curve", "offense"), ("d_curve", "defense"), ("curve", "total")):
        print(f"  {n} peaks at {int(curve.loc[curve[c].idxmax(), 'age'])}")
    plot(curve, raw_total, delta_total)
