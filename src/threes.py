#!/usr/bin/env python3
"""
Sub-question: are three-point shooters overpaid?

Overpaid = paid more than the wins we expect from them. So per player-season:

  salary in WAR units   (salary - near-minimum salary) / $ per WAR that season. what the team paid,
                        counted in wins at that season's market price
  expected WAR          war.py's projection made the summer before (what a team could know when paying)
  shooting              from the season before, what teams saw: 3PA per 36 min and 3P% (shrunk towards
                        35% with 100 fake attempts, so 4/8 isn't a 50% shooter). both as z-scores within season

  salary_war = b0 + b1 * expected WAR + b2 * 3PA/36 + b3 * 3P% + rookie deal + age + age^2

b2 > 0: for the same expected wins, high-volume shooters get paid more = overpaid.
b2 < 0: underpaid. Per era (2011-14, 2015-19, 2020-25) since the league changed a lot, and per season
for the plot. Rookie-scale deals are set by the draft slot, not the market, so they get a flag.
Bootstrap over players for the CI.

Also a second version with points per 36 as a control: is it the shooting, or just scoring?

Output
  data/processed/threes.parquet    coefficients per era / season
  figures/threes.png

  python src/threes.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS
from surplus import dollars_per_war
from war import MIN_SALARY_Q, salaries_with_ids

ERAS = {"2011-15": range(2011, 2015), "2015-20": range(2015, 2020), "2020-26": range(2020, 2026)}
MIN_MIN = 500          # played real minutes the season before, otherwise there is no shooting profile
PRIOR_PCT, PRIOR_N = 0.35, 100
RNG = np.random.default_rng(3)


def panel() -> pd.DataFrame:
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    sal = salaries_with_ids()
    dpw = dollars_per_war(war, sal)
    box = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")
                     [["PLAYER_ID", "MIN", "FG3M", "FG3A", "FGA", "PTS", "AGE"]].assign(season=y) for y in SEASONS])
    box.columns = ["player_id", "min", "fg3m", "fg3a", "fga", "pts", "age", "season"]
    first = box.groupby("player_id")["season"].min()
    age_first = box.sort_values("season").drop_duplicates("player_id").set_index("player_id")["age"]

    # salary in season s <- projection + box score as of s-1
    proj = war[["player_id", "season", "war", "age_next"]].assign(season=war["season"] + 1)
    prev = box.assign(season=box["season"] + 1)
    d = sal.merge(proj, on=["player_id", "season"]).merge(prev, on=["player_id", "season"])
    d = d[(d["min"] >= MIN_MIN) & d["season"].isin(dpw.index)].copy()

    floor = d.groupby("season")["salary"].transform(lambda x: x.quantile(MIN_SALARY_Q))
    d["dpw"] = d["season"].map(dpw)
    d["salary_war"] = (d["salary"] - floor).clip(lower=0) / d["dpw"]
    d["war"] = d["war"].clip(lower=0)
    d["fg3a36"] = d["fg3a"] / d["min"] * 36
    d["fg3pct"] = (d["fg3m"] + PRIOR_PCT * PRIOR_N) / (d["fg3a"] + PRIOR_N)
    d["pts36"] = d["pts"] / d["min"] * 36
    for c in ("fg3a36", "fg3pct", "pts36"):
        d[f"z_{c}"] = d.groupby("season")[c].transform(lambda x: (x - x.mean()) / x.std())
    # rookie scale = first 4 seasons. guys already in the 2010-11 data only if they were 20 or younger then
    exp = d["season"] - d["player_id"].map(first)
    fresh = (d["player_id"].map(first) > min(SEASONS)) | (d["player_id"].map(age_first) <= 20)
    d["rookie"] = ((exp <= 3) & fresh).astype(float)
    d["age"] = d["age_next"]
    return d.reset_index(drop=True)


def ols(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    return np.linalg.lstsq(X, d["salary_war"].to_numpy(), rcond=None)[0][1:]


def fit(d: pd.DataFrame, cols: list[str], n_boot=400) -> pd.DataFrame:
    """coefficients in WAR units + $M per season (at that sample's average $/WAR), player bootstrap CI."""
    est = ols(d, cols)
    players = d["player_id"].unique()
    by = {p: i for p, i in d.groupby("player_id").indices.items()}
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(players, len(players))
        boots.append(ols(d.iloc[np.concatenate([by[p] for p in pick])], cols))
    boots = np.array(boots)
    m = d["dpw"].mean() / 1e6
    return pd.DataFrame({"term": cols, "war_units": est, "lo": np.percentile(boots, 2.5, 0),
                         "hi": np.percentile(boots, 97.5, 0), "musd": est * m,
                         "musd_lo": np.percentile(boots, 2.5, 0) * m, "musd_hi": np.percentile(boots, 97.5, 0) * m})


BASE = ["war", "z_fg3a36", "z_fg3pct", "rookie", "age", "age2"]


def plot(per: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for term, color, label in (("z_fg3a36", blue, "3PA per 36 (volume)"), ("z_fg3pct", orange, "3P% (accuracy)")):
        t = per[per["term"] == term]
        ax.fill_between(t["season"], t["musd_lo"], t["musd_hi"], color=color, alpha=0.15, lw=0)
        ax.plot(t["season"], t["musd"], color=color, lw=2, marker="o", ms=4, label=label)
    ax.axhline(0, color=ink, lw=0.8)
    ax.set_xticks(per["season"].unique(), [f"{s % 100:02d}-{(s + 1) % 100:02d}" for s in per["season"].unique()],
                  fontsize=8)
    ax.set_ylabel("$M a season for +1 SD, same expected wins", color=muted, fontsize=8.5)
    ax.set_title("Paid for shooting beyond the wins it brings? (above 0 = overpaid)", loc="left", fontsize=11,
                 color=ink)
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8.5, left=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "threes.png", facecolor=surface)
    print(f"\nplot -> {out / 'threes.png'}")


if __name__ == "__main__":
    d = panel()
    d["age2"] = (d["age"] - 27) ** 2
    print(f"{len(d):,} player-seasons with a salary, a projection and {MIN_MIN}+ minutes the season before, "
          f"{d['season'].min()}-{d['season'].max()}")
    print(f"$ per WAR {d['dpw'].min() / 1e6:.1f}M to {d['dpw'].max() / 1e6:.1f}M. salary in WAR units: "
          f"mean {d['salary_war'].mean():.2f}, expected WAR mean {d['war'].mean():.2f}")

    rows = []
    for era, yrs in ERAS.items():
        sub = d[d["season"].isin(yrs)]
        for name, cols in (("value + shooting", BASE), ("+ points per 36", BASE + ["z_pts36"])):
            r = fit(sub, cols).assign(era=era, model=name, n=len(sub))
            rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    for name in res["model"].unique():
        print(f"\nMODEL: salary (WAR units) ~ {name}.  shown: $M per season, 95% CI")
        t = res[(res["model"] == name) & res["term"].isin(["war", "z_fg3a36", "z_fg3pct", "z_pts36"])]
        print(t.pivot_table(index="term", columns="era", values="musd", sort=False).round(2).to_string())
        lo = t.pivot_table(index="term", columns="era", values="musd_lo", sort=False).round(1).astype(str)
        hi = t.pivot_table(index="term", columns="era", values="musd_hi", sort=False).round(1).astype(str)
        print("  CI:\n" + ("[" + lo + ", " + hi + "]").to_string())

    # per season for the plot, main model only
    per = pd.concat([fit(d[d["season"] == s], BASE, n_boot=200).assign(season=s) for s in sorted(d["season"].unique())])
    res = pd.concat([res, per.assign(model="value + shooting, per season")], ignore_index=True)

    # plain view: surplus by 3PA/36 quintile, per era (no controls)
    d["surplus_war"] = d["war"] - d["salary_war"]
    d["q"] = d.groupby("season")["fg3a36"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1)
    print("\nexpected WAR minus salary in WAR units, by 3PA/36 quintile (Q5 = most threes), no controls:")
    print(d.pivot_table(index="q", columns=pd.cut(d["season"], [2010, 2014, 2019, 2025], labels=list(ERAS)),
                        values="surplus_war", aggfunc="mean", observed=True).round(2).to_string())

    res.to_parquet(PROCESSED_DIR / "threes.parquet", index=False)
    plot(per)
