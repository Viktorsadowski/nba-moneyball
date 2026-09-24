#!/usr/bin/env python3
"""
Step 9: injury proneness, and what it does to a player's value.

Three questions:
  1. are some players injury prone?   does missing games to injury this season predict missing
     games next season, beyond chance? (year-to-year correlation, share of variance that belongs
     to the player, next-season games missed by quintile of past injuries)
  2. do injured players decline faster?   is next season's RAPM lower than our age-adjusted
     projection said, for guys with an injury history? if yes the projection gets a discount
  3. injury risk model   predict next season's games missed to injury. a simple linear model
     vs gradient boosting, rolling backtest (train on everything before, predict the season after)

Injury games are normalised within each season (share of team games missed / league average
that season), because the PST log (to 2019-20) catches fewer short absences than the official
reports (2021-22 on). 2020-21 has no injury data, it's skipped as a target and NaN as a feature.
Converted back to games with the official-report era average, the more complete count.

Output
  data/processed/injury_risk.parquet   per player per "as of" season, projecting the next one:
                                       exp_inj_games, exp_other_games, exp_games, val_adj
  figures/injury_proneness.png

  python src/injury.py
"""

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label

WEIGHTS = (0.5, 0.3, 0.2)
REPORT_ERA = range(2021, 2026)
ROTATION = 500            # minutes last season, for the evaluation + the analyses
FULL = 82
SERIOUS = re.compile(r"achilles|acl|cruciate|fractur|broken|surgery|torn|tear|meniscus|patell|reconstruct", re.I)
RNG = np.random.default_rng(7)


# ── panel ─────────────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame:
    av = pd.read_parquet(PROCESSED_DIR / "availability.parquet")
    pl = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["SEASON", "PLAYER_ID", "AGE", "MIN", "GP"]]
                    for y in SEASONS])
    pl.columns = ["season", "player_id", "age", "min", "gp_box"]
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")[["season", "player_id", "rapm", "name"]]
    val = pd.read_parquet(PROCESSED_DIR / "value.parquet")[["season", "player_id", "val"]]

    d = av.merge(pl, on=["season", "player_id"], how="left").merge(rapm, on=["season", "player_id"], how="left")
    d = d.merge(val, on=["season", "player_id"], how="left")
    d["min"] = d["min"].fillna(0)
    d["mpg"] = d["min"] / d["gp"].clip(lower=1)
    d["share"] = d["inj_games"] / d["team_gp"]
    d["other"] = (1 - d["gp"] / d["team_gp"] - d["share"].fillna(0)).clip(0, 1)
    d["serious"] = d["inj_note"].fillna("").str.contains(SERIOUS).astype(float)
    d.loc[d["inj_games"].isna(), "serious"] = np.nan

    # league average per season over guys who were actually on a roster most of the year
    pop = (d["gp"] + d["inj_games"].fillna(0)) >= 20
    mean_share = d[pop].groupby("season")["share"].mean()
    d["norm"] = d["share"] / d["season"].map(mean_share)
    d.attrs["mean_share"] = mean_share
    return d


def features(d: pd.DataFrame, s: int) -> pd.DataFrame:
    """what we know about everyone at the end of season s."""
    cur = d[d["season"] == s].set_index("player_id")
    f = pd.DataFrame(index=cur.index)
    num = {"norm": 0.0, "other": 0.0}
    den = {"norm": 0.0, "other": 0.0}
    for k in num:
        num[k] = pd.Series(0.0, index=f.index)
        den[k] = pd.Series(0.0, index=f.index)
    serious_3y = pd.Series(0.0, index=f.index)
    n_seasons = pd.Series(0, index=f.index)
    for lag, w in enumerate(WEIGHTS):
        past = d[d["season"] == s - lag].set_index("player_id").reindex(f.index)
        for k in num:
            ok = past[k].notna()
            num[k] = num[k].add(w * past[k].where(ok, 0), fill_value=0)
            den[k] = den[k].add(w * ok, fill_value=0)
        serious_3y = np.maximum(serious_3y, past["serious"].fillna(0))
        n_seasons += past["gp"].notna().astype(int)
    f["norm_blend"] = num["norm"] / den["norm"].replace(0, np.nan)   # NaN if no injury data at all
    f["other_blend"] = num["other"] / den["other"].replace(0, np.nan)
    f["norm_last"] = cur["norm"]
    f["spells_last"] = cur["inj_spells"]
    f["serious_last"] = cur["serious"]
    f["serious_3y"] = serious_3y
    f["n_seasons"] = n_seasons
    f["age_next"] = cur["age"] + 1
    f["mpg"] = cur["mpg"]
    f["min_last"] = cur["min"]
    f["gp_last"] = cur["gp"] / cur["team_gp"]
    f["val"] = cur["val"]
    f["season"] = s
    return f.reset_index()


def pairs(d: pd.DataFrame) -> pd.DataFrame:
    """features at s + what actually happened at s+1."""
    out = []
    for s in SEASONS[:-1]:
        f = features(d, s)
        nxt = d[d["season"] == s + 1][["player_id", "norm", "other", "share", "inj_games", "team_gp", "rapm", "min"]]
        nxt = nxt.rename(columns={c: f"{c}_next" for c in nxt.columns if c != "player_id"})
        out.append(f.merge(nxt, on="player_id", how="inner"))
    return pd.concat(out, ignore_index=True)


# ── 1. proneness ──────────────────────────────────────────────────────────────

def proneness(p: pd.DataFrame, d: pd.DataFrame, per_game: float) -> pd.DataFrame:
    r = p[(p["min_last"] >= ROTATION) & p["norm_next"].notna()]
    print("\n1. ARE SOME PLAYERS INJURY PRONE?")
    for col, lab in (("norm_last", "last season"), ("norm_blend", "last 3 seasons (50/30/20)")):
        ok = r[col].notna()
        rho = r.loc[ok, [col, "norm_next"]].corr(method="spearman").iloc[0, 1]
        # null: shuffle next season within season, how big does rho get by chance
        null = [r.loc[ok, col].corr(r.loc[ok].groupby("season")["norm_next"].transform(np.random.permutation),
                                    method="spearman") for _ in range(200)]
        print(f"  injuries {lab:26} vs next season: spearman {rho:.2f}  (by chance: +-{2 * np.std(null):.2f})")

    # share of the spread that sits with the player (one-way ANOVA ICC), players with 4+ seasons of data
    q = d[(d["min"] >= ROTATION) & d["norm"].notna()]
    q = q[q.groupby("player_id")["norm"].transform("size") >= 4]
    k = q.groupby("player_id").size().mean()
    msb = q.groupby("player_id")["norm"].mean().var() * k
    msw = q.groupby("player_id")["norm"].var().mean()
    icc = (msb - msw) / (msb + (k - 1) * msw)
    print(f"  share of injury variance that belongs to the player (ICC): {icc:.2f} "
          f"({q['player_id'].nunique()} players with 4+ rotation seasons)")

    # quintiles of the 3-season history
    r = r[r["norm_blend"].notna() & (r["n_seasons"] >= 2)].copy()
    r["q"] = pd.qcut(r["norm_blend"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["games_next"] = r["norm_next"] * per_game
    tab = r.groupby("q").agg(n=("games_next", "size"), hist=("norm_blend", "mean"), games_next=("games_next", "mean"))
    tab["lo"], tab["hi"] = zip(*[boot_ci(r.loc[r["q"] == i, "games_next"]) for i in tab.index])
    print("  games missed to injury next season (per 82, official-report scale), by past-injury quintile:")
    for i, row in tab.iterrows():
        print(f"    Q{i}  past {row['hist']:.2f}x league avg -> {row['games_next']:.1f} games  "
              f"[{row['lo']:.1f}, {row['hi']:.1f}]  n={int(row['n'])}")

    # the career list: most games lost per season relative to the league, 5+ seasons of data
    c = d[d["norm"].notna() & ((d["gp"] + d["inj_games"]) >= 20)]
    c = c.groupby("player_id").agg(name=("name", "last"), seasons=("norm", "size"), norm=("norm", "mean"))
    c = c[c["seasons"] >= 5].sort_values("norm", ascending=False)
    names = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet") for y in SEASONS]) \
        .groupby("PLAYER_ID")["PLAYER_NAME"].last()
    c["name"] = c.index.map(names)
    c["games_per_82"] = c["norm"] * per_game
    print("  most injury-prone careers (5+ seasons, games lost per 82 on the official-report scale):")
    print(c[["name", "seasons", "games_per_82"]].head(12).round(1).to_string(index=False))
    return tab


def boot_ci(x: pd.Series, w: pd.Series | None = None, n: int = 1000) -> tuple[float, float]:
    # same (weighted) mean as the table shows, resampled
    x = x.to_numpy()
    w = np.ones(len(x)) if w is None else w.to_numpy()
    means = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        means.append(np.average(x[i], weights=w[i]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── 2. decline ────────────────────────────────────────────────────────────────

def decline(p: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """projection error (next season RAPM - our age-adjusted value) vs injury history."""
    r = p[(p["min_last"] >= ROTATION) & (p["min_next"] >= ROTATION) & p["val"].notna()
          & p["rapm_next"].notna() & p["norm_blend"].notna()].copy()
    r["err"] = r["rapm_next"] - r["val"]
    r["w"] = np.minimum(r["min_last"], r["min_next"])
    xs = ["norm_blend", "serious_3y"]
    # val as a control too: the projection is shrunk, so good players beat it and bad ones miss it,
    # regardless of injuries. without it that level effect would leak into the injury terms

    def fit(df):
        X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in xs]
                            + [df["age_next"].to_numpy(), df["val"].to_numpy()])
        W = np.sqrt(df["w"].to_numpy())
        return np.linalg.lstsq(X * W[:, None], df["err"].to_numpy() * W, rcond=None)[0]

    b = fit(r)
    # bootstrap over players, a player's seasons stay together
    players = r["player_id"].unique()
    groups = {pid: g for pid, g in r.groupby("player_id")}
    boots = []
    for _ in range(300):
        pick = RNG.choice(players, len(players))
        boots.append(fit(pd.concat([groups[x] for x in pick])))
    boots = np.array(boots)
    print("\n2. DO PLAYERS WITH AN INJURY HISTORY DECLINE FASTER?")
    print("  next season RAPM minus our (age-adjusted) projection, regressed on injury history:")
    coef = {}
    for i, name in enumerate(["intercept"] + xs + ["age_next", "val"]):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        sig = "" if lo < 0 < hi else "  <- significant"
        print(f"    {name:12} {b[i]:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
        coef[name] = (b[i], lo, hi)

    r["q"] = pd.qcut(r["norm_blend"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    tab = r.groupby("q").apply(lambda g: pd.Series({"err": np.average(g["err"], weights=g["w"]),
                                                    "n": len(g)}), include_groups=False)
    tab["lo"], tab["hi"] = zip(*[boot_ci(r.loc[r["q"] == i, "err"], r.loc[r["q"] == i, "w"]) for i in tab.index])
    print("  projection error by past-injury quintile (negative = did worse than projected):")
    for i, row in tab.iterrows():
        print(f"    Q{i}  {row['err']:+.2f}  [{row['lo']:+.2f}, {row['hi']:+.2f}]")
    return coef, tab


# ── 3. risk model ─────────────────────────────────────────────────────────────

FEATS = ["norm_blend", "norm_last", "spells_last", "serious_last", "serious_3y", "n_seasons",
         "age_next", "mpg", "min_last", "gp_last"]


def linear_fit(tr: pd.DataFrame, col: str, target: str):
    # regression to the mean + age: pred = b0 + b1*history + b2*age
    x = tr[col].fillna(tr[col].mean() if col == "other_blend" else 1.0)
    X = np.column_stack([np.ones(len(tr)), x, tr["age_next"].fillna(27)])
    return np.linalg.lstsq(X, tr[target].to_numpy(), rcond=None)[0]


def linear_pred(b, df: pd.DataFrame, col: str) -> np.ndarray:
    x = df[col].fillna(1.0 if col == "norm_blend" else df[col].mean())
    return np.clip(b[0] + b[1] * x + b[2] * df["age_next"].fillna(27), 0, None)


def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def risk_model(p: pd.DataFrame, per_game: float) -> pd.DataFrame:
    print("\n3. INJURY RISK MODEL (rolling backtest, train on seasons before, predict the next)")
    have = p[p["norm_next"].notna()]
    res = []
    for T in SEASONS[3:]:
        te = have[(have["season"] == T - 1) & (have["min_last"] >= ROTATION)]
        tr = have[have["season"] < T - 1]
        if te.empty or len(tr) < 500:
            continue
        b = linear_fit(tr, "norm_blend", "norm_next")
        g = gbm().fit(tr[FEATS], tr["norm_next"])
        res.append(pd.DataFrame({
            "season": T, "actual": te["norm_next"] * per_game,
            "league_avg": per_game,
            "last_season": te["norm_last"].fillna(1.0) * per_game,
            "linear": linear_pred(b, te, "norm_blend") * per_game,
            "gbm": np.clip(g.predict(te[FEATS]), 0, None) * per_game,
        }))
    res = pd.concat(res)
    print(f"  mean absolute error, games missed to injury next season (rotation players, n={len(res)}):")
    maes = {}
    for m in ("league_avg", "last_season", "linear", "gbm"):
        maes[m] = (res[m] - res["actual"]).abs().mean()
        rmse = np.sqrt(((res[m] - res["actual"]) ** 2).mean())
        print(f"    {m:12} MAE {maes[m]:5.2f}   RMSE {rmse:5.2f}")
    best = min(("linear", "gbm"), key=maes.get)
    print(f"  -> using {best}")
    return best


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(t1: pd.DataFrame, t2: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    labs = ["Q1\nfewest", "Q2", "Q3", "Q4", "Q5\nmost"]
    # projections are shrunk, so everyone beats them a bit. show each quintile vs the overall average
    avg = np.average(t2["err"], weights=t2["n"])
    t2 = t2.assign(err_rel=t2["err"] - avg, lo=t2["lo"] - avg, hi=t2["hi"] - avg)
    for ax, tab, col, title, fmt in (
        (axes[0], t1, "games_next", "Games missed to injury next season", "{:.1f}"),
        (axes[1], t2, "err_rel", "Next season RAPM vs projection, relative to typical (per 100)", "{:+.2f}"),
    ):
        ax.set_facecolor(surface)
        x = np.arange(len(tab))
        y = tab[col].to_numpy()
        ax.bar(x, y, width=0.6, color=blue, zorder=2)
        ax.errorbar(x, y, yerr=[np.clip(y - tab["lo"], 0, None), np.clip(tab["hi"] - y, 0, None)], fmt="none", ecolor=ink, elinewidth=1,
                    capsize=3, zorder=3)
        for xi, yi, hi, lo in zip(x, y, tab["hi"], tab["lo"]):
            ax.annotate(fmt.format(yi), (xi, hi if yi >= 0 else lo), xytext=(0, 4 if yi >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8.5, color=ink)
        ax.axhline(0, color=muted, lw=0.8)
        ax.margins(y=0.12)  # room for the value labels under the negative bars
        ax.set_xticks(x, labs)
        ax.set_title(title, loc="left", fontsize=10.5, color=ink)
        ax.set_xlabel("injuries over the previous 3 seasons, quintile", color=muted, fontsize=9)
        ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8.5)
    fig.suptitle("Injury history predicts more injuries, but not a worse player when he's on the floor",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "injury_proneness.png", facecolor=surface)
    print(f"\nplot -> {out / 'injury_proneness.png'}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    d = load_panel()
    mean_share = d.attrs["mean_share"]
    # games per 82 that "1x league average" is worth, official-report era
    per_game = float(mean_share[[y for y in REPORT_ERA if y in mean_share.index]].mean()) * FULL
    print(f"league average: {per_game:.1f} games per 82 missed to injury (official-report era)")

    p = pairs(d)
    t1 = proneness(p, d, per_game)
    coef, t2 = decline(p)
    best = risk_model(p, per_game)
    plot(t1, t2)

    # value discount only if the injury history terms are clearly there
    use = {k: v[0] for k, v in coef.items() if k in ("norm_blend", "serious_3y") and not (v[1] < 0 < v[2])}
    center = p.loc[p["min_last"] >= ROTATION, "norm_blend"].mean()
    print(f"\nvalue adjustment applied from: {', '.join(use) or 'nothing (not significant)'}")

    # projections as of every season, each fit only on what was known by then
    have = p[p["norm_next"].notna()]
    out = []
    for s in SEASONS:
        f = features(d, s)
        tr = have[have["season"] < s]
        if len(tr) < 500:
            tr = have  # first seasons: not enough history, use everything (only affects old seasons)
        if best == "gbm":
            inj = np.clip(gbm().fit(tr[FEATS], tr["norm_next"]).predict(f[FEATS]), 0, None)
        else:
            inj = linear_pred(linear_fit(tr, "norm_blend", "norm_next"), f, "norm_blend")
        oth = linear_pred(linear_fit(tr.dropna(subset=["other_next"]), "other_blend", "other_next"), f, "other_blend")
        f["exp_inj_games"] = inj * per_game
        f["exp_other_games"] = oth * FULL
        f["exp_games"] = (FULL - f["exp_inj_games"] - f["exp_other_games"]).clip(0, FULL)
        # centred on the typical rotation player, so the average guy gets no discount
        f["val_adj"] = sum(b * (f[k].fillna(center if k == "norm_blend" else 0)
                                - (center if k == "norm_blend" else p.loc[p["min_last"] >= ROTATION, k].mean()))
                           for k, b in use.items())
        out.append(f[["player_id", "season", "exp_inj_games", "exp_other_games", "exp_games", "val_adj",
                      "norm_blend", "serious_3y"]])
    risk = pd.concat(out, ignore_index=True)
    risk.to_parquet(PROCESSED_DIR / "injury_risk.parquet", index=False)

    last = risk[risk["season"] == risk["season"].max()].merge(
        d[d["season"] == d["season"].max()][["player_id", "name", "min"]], on="player_id")
    last = last[last["min"] >= 1000].sort_values("exp_inj_games", ascending=False)
    print(f"\nhighest projected injury games {season_label(int(last['season'].iloc[0]) + 1)} (1000+ min this season):")
    print(last[["name", "exp_inj_games", "exp_games", "val_adj"]].head(12).round(2).to_string(index=False))
