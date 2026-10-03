#!/usr/bin/env python3
"""
Market value: what the league would pay a player on the open market, from what the market actually paid.

Our value model (RAPM -> WAR -> $) says what a player is worth. Other GMs price players more like free agency
does: points, minutes, a few years of it, age. The gap between the two is where a trade can be fair to the
other team and still good for us. So this fits the market's price on its own:

  signings    every new veteran contract since 2013-14 (salaries_with_ids + current contracts): a player's
              salary jumps or drops outside what raises inside one deal allow (-10% / +9%), or he shows up
              with a salary after a year without one. first-round rookie deals (years 1-4) are left out,
              the CBA fixes those
  target      first-year salary as a share of that season's cap
  censored    near-minimum deals only say "worth the minimum or less" (share <= 3%), max deals only say
              "worth the max or more" (>= 24.5%, the lowest max tier). Tobit, so neither end drags the fit
  features    the season before + the 3 seasons before (per game, games-weighted): points, rebounds,
              assists, minutes, games played, true shooting, age. and our RAPM value, to see how much of it
              the market pays for on top of the box score
  test        leave one signing season out, error on the deals in between the two censoring limits

Then for every player now: his market price per season of his contract (age moved on each year, stats as
they are), minus what he's paid = market surplus. The trade search uses it for the other side of the table.

Output
  data/processed/market_signings.parquet   the signings + predictions (out of sample)
  data/processed/market_value.parquet      per current player: market share now, market surplus of the
                                           remaining contract
  data/processed/market_model.json         coefficients
  figures/market_value.png

  python src/market_value.py
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label

# salary cap by season (season = the year it starts), NBA.com / CBA
CAP = {2010: 58.044, 2011: 58.044, 2012: 58.044, 2013: 58.679, 2014: 63.065, 2015: 70.0, 2016: 94.143,
       2017: 99.093, 2018: 101.869, 2019: 109.14, 2020: 109.14, 2021: 112.414, 2022: 123.655, 2023: 136.021,
       2024: 140.588, 2025: 154.647, 2026: 164.961}
CAP = {k: v * 1e6 for k, v in CAP.items()}
LO, HI = 0.03, 0.245          # censoring: near-minimum, lowest max tier
RAISE = (0.90, 1.09)          # year-to-year salary change that still counts as the same deal
FIRST_SIGNING = 2013          # need 3 seasons of stats before it
GROWTH = 0.091                # cap growth for future seasons, same as surplus.py (payroll, last 10 seasons)

BOX = ["pts1", "reb1", "ast1", "mpg1", "gp1", "pts3", "reb3", "ast3", "mpg3", "gp3", "ts3",
       "age", "age2", "pts3_2"]
RAPM = ["val"]
_CACHE: dict = {}


# ── data ──────────────────────────────────────────────────────────────────────

def salary_series() -> pd.DataFrame:
    """season, player_id, salary: past seasons from the team pages, next season on from current contracts"""
    from surplus import contracts_with_ids
    from war import salaries_with_ids
    past = salaries_with_ids()
    fut = contracts_with_ids()[["season", "player_id", "salary"]]
    fut = fut.groupby(["season", "player_id"], as_index=False)["salary"].max()
    s = pd.concat([past, fut[fut["season"] > past["season"].max()]], ignore_index=True)
    s["player_id"] = s["player_id"].astype(int)
    return s


def stats_table() -> pd.DataFrame:
    p = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet") for y in SEASONS])
    p = p.rename(columns={"SEASON": "season", "PLAYER_ID": "player_id", "AGE": "age_s", "GP": "gp"})
    g = p.groupby(["season", "player_id"])
    t = g[["gp", "MIN", "PTS", "REB", "AST", "FGA", "FTA"]].sum()
    t["age_s"] = g["age_s"].max()
    return t.reset_index()


def features(stats: pd.DataFrame, rows: pd.DataFrame, skip_missed: bool = False) -> pd.DataFrame:
    """for each (player_id, season s): box features from s-1 and s-3..s-1, RAPM value as of s-1.
    skip_missed: a player who sat out all of s-1 (injury) gets priced on the 3 seasons before that, like a GM
    would (Haliburton isn't a 0-point player). only for pricing now, the fit uses what was there"""
    st = stats.set_index(["player_id", "season"])
    out = rows[["player_id", "season"]].copy()
    shift = np.zeros(len(out), int)
    if skip_missed:
        k1 = pd.MultiIndex.from_arrays([out["player_id"], out["season"] - 1])
        k2 = pd.MultiIndex.from_arrays([out["player_id"], out["season"] - 2])
        gp1 = st["gp"].reindex(k1).fillna(0).to_numpy()
        gp2 = st["gp"].reindex(k2).fillna(0).to_numpy()
        shift = ((gp1 == 0) & (gp2 > 0)).astype(int)
    agg = {k: np.zeros(len(out)) for k in ("gp", "MIN", "PTS", "REB", "AST", "FGA", "FTA")}
    last = {k: np.zeros(len(out)) for k in agg}
    age = np.full(len(out), np.nan)
    for lag in (1, 2, 3):
        key = pd.MultiIndex.from_arrays([out["player_id"], out["season"] - lag - shift])
        x = st.reindex(key)
        for k in agg:
            v = x[k].fillna(0).to_numpy()
            agg[k] += v
            if lag == 1:
                last[k] = v
        a = x["age_s"].to_numpy() + lag + shift
        age = np.where(np.isnan(age), a, age)
    gp1, gp3 = np.maximum(last["gp"], 1), np.maximum(agg["gp"], 1)
    out["pts1"], out["reb1"], out["ast1"] = last["PTS"] / gp1, last["REB"] / gp1, last["AST"] / gp1
    out["mpg1"], out["gp1"] = last["MIN"] / gp1, last["gp"]
    out["pts3"], out["reb3"], out["ast3"] = agg["PTS"] / gp3, agg["REB"] / gp3, agg["AST"] / gp3
    out["mpg3"], out["gp3"] = agg["MIN"] / gp3, agg["gp"]
    tsa = 2 * (agg["FGA"] + 0.44 * agg["FTA"])
    out["ts3"] = np.where(tsa > 50, agg["PTS"] / np.maximum(tsa, 1), 0.53)     # league-ish average if no volume
    out["age"] = age
    out["age2"] = (age - 28) ** 2
    out["pts3_2"] = out["pts3"] ** 2 / 10         # stars get paid more than linear in points
    if "val" not in _CACHE:
        v = pd.read_parquet(PROCESSED_DIR / "value.parquet")[["season", "player_id", "val"]]
        _CACHE["val"] = v.assign(season=v["season"] + 1)    # value as of s-1 = what a GM knew at signing
    val = _CACHE["val"]
    out = out.assign(vs=out["season"] - shift).merge(val.rename(columns={"season": "vs"}), on=["vs", "player_id"],
                                                     how="left").drop(columns="vs")
    out["val"] = out["val"].fillna(0.0)
    return out


def signings(sal: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    s = sal.sort_values(["player_id", "season"])
    prev = s.groupby("player_id")["salary"].shift(1)
    prev_season = s.groupby("player_id")["season"].shift(1)
    r = s["salary"] / prev
    new = prev.isna() | (prev_season != s["season"] - 1) | (r < RAISE[0]) | (r > RAISE[1])
    s = s[new & (s["season"] >= FIRST_SIGNING) & (s["season"] <= max(CAP))].copy()
    # first-round rookie scale, years 1-4
    d = draft.drop_duplicates("player_id").set_index("player_id")
    rnd, yr = s["player_id"].map(d["round"]), s["season"] - s["player_id"].map(d["draft_year"])
    s = s[~((rnd == 1) & (yr <= 3))]
    s["share"] = s["salary"] / s["season"].map(CAP)
    return s


# ── Tobit ─────────────────────────────────────────────────────────────────────

def tobit_fit(X: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """linear Tobit, censored below at LO and above at HI. returns [beta..., log sigma]"""
    def nll(th):
        b, s = th[:-1], np.exp(th[-1])
        mu = X @ b
        ll = np.where(lo, norm.logcdf((LO - mu) / s),
                      np.where(hi, norm.logsf((HI - mu) / s), norm.logpdf((y - mu) / s) - np.log(s)))
        return -ll.sum()
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    th = minimize(nll, np.r_[b0, np.log(0.04)], method="L-BFGS-B").x
    return th


def design(df: pd.DataFrame, cols: list, mu=None, sd=None):
    Z = df[cols].to_numpy(float)
    if mu is None:
        mu, sd = Z.mean(0), Z.std(0) + 1e-9
    return np.column_stack([np.ones(len(Z)), (Z - mu) / sd]), mu, sd


def fit(df: pd.DataFrame, cols: list) -> dict:
    X, mu, sd = design(df, cols)
    th = tobit_fit(X, df["share"].to_numpy(), df["share"].to_numpy() <= LO, df["share"].to_numpy() >= HI)
    return dict(cols=cols, mu=mu, sd=sd, beta=th[:-1], sigma=float(np.exp(th[-1])))


def predict(m: dict, df: pd.DataFrame, expected: bool = True) -> np.ndarray:
    """market share of the cap. expected=True: E[share] with the min floor (nobody signs below the minimum),
    the max isn't a floor for the value so it's left open"""
    X, _, _ = design(df, m["cols"], m["mu"], m["sd"])
    mu = X @ m["beta"]
    if not expected:
        return mu
    s = m["sigma"]
    # E[max(y*, LO)]: below the minimum a player is still paid the minimum
    z = (LO - mu) / s
    return LO * norm.cdf(z) + mu * norm.sf(z) + s * norm.pdf(z)


def backtest(df: pd.DataFrame, cols: list) -> dict:
    pred = np.zeros(len(df))
    for y in sorted(df["season"].unique()):
        tr, te = df["season"] != y, df["season"] == y
        pred[te.to_numpy()] = predict(fit(df[tr], cols), df[te])
    mid = (df["share"] > LO) & (df["share"] < HI)
    e = (df["share"] - pred)[mid]
    # max deals: did we at least say he's worth a max?
    mx = df["share"] >= HI
    return dict(mae=float(e.abs().mean()), rmse=float(np.sqrt((e ** 2).mean())), n=int(mid.sum()),
                r=float(np.corrcoef(df["share"][mid], pred[mid])[0, 1]),
                max_hit=float((pred[mx] >= HI * 0.9).mean()), pred=pred)


# ── market surplus now ────────────────────────────────────────────────────────

def market_now(m: dict, stats: pd.DataFrame) -> pd.DataFrame:
    """every player with a contract next season: market price per contract year - salary"""
    from surplus import contracts_with_ids
    nxt = max(SEASONS) + 1
    con = contracts_with_ids()
    con = con.groupby(["player_id", "season"], as_index=False)["salary"].max()
    con["player_id"] = con["player_id"].astype(int)
    base = features(stats, pd.DataFrame({"player_id": con["player_id"].unique(), "season": nxt}), skip_missed=True)
    base = base.set_index("player_id")
    rows = []
    for (pid, s), sal in zip(zip(con["player_id"], con["season"]), con["salary"]):
        f = base.loc[[pid]].copy()
        k = s - nxt
        # the market's own age discount: same stats, k years older
        f["age"] = f["age"] + k
        f["age2"] = (f["age"] - 28) ** 2
        share = float(predict(m, f)[0])
        cap = CAP[nxt] * (1 + GROWTH) ** k
        rows.append(dict(player_id=pid, season=s, salary=sal, share=share, market=share * cap))
    t = pd.DataFrame(rows)
    # same floor: a minimum contract is never "overpaid", the team can't pay less
    t["market_surplus"] = t["market"] - np.maximum(t["salary"], LO * CAP[nxt] * (1 + GROWTH) ** (t["season"] - nxt))
    g = t.groupby("player_id")
    out = pd.DataFrame({"years": g.size(), "salary": g["salary"].sum(), "market": g["market"].sum(),
                        "market_surplus": g["market_surplus"].sum(),
                        "share_now": g.apply(lambda x: x.loc[x["season"].idxmin(), "share"], include_groups=False)})
    return out.join(base[["age", "pts3", "mpg3", "val"]]).reset_index()


def pick_market_value(m: dict, stats: pd.DataFrame, sal: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """what the market says a pick turned into: for every drafted player, rookie seasons 1-4, the market price
    of what he did that season (stats through that season, priced like a signing the summer after) minus his
    salary, as shares of the cap. busts count as 0. smoothed over log(pick) like rookies.py"""
    from rookies import PICKS, design, wls
    last = max(SEASONS)
    d = draft[draft["draft_year"] <= last - 3].drop_duplicates("player_id")
    rows = d.loc[d.index.repeat(4), ["player_id", "draft_year", "pick"]].copy()
    rows["season"] = rows["draft_year"] + np.tile(np.arange(4), len(d))
    f = features(stats, rows.assign(season=rows["season"] + 1))
    rows["mv"] = predict(m, f)
    played = pd.MultiIndex.from_arrays([rows["player_id"], rows["season"]]).isin(
        pd.MultiIndex.from_arrays([stats["player_id"], stats["season"]]))
    rows.loc[~played, "mv"] = 0.0                  # not in the league that season: worth nothing to the team
    s = sal.set_index(["player_id", "season"])["salary"]
    rows["sal"] = pd.MultiIndex.from_arrays([rows["player_id"], rows["season"]]).map(s).fillna(0.0).to_numpy() \
        / rows["season"].map(CAP).to_numpy()
    # nobody is paid under the minimum, so a near-minimum guy on a near-minimum deal is worth what he's paid
    rows["surplus"] = np.where(played, rows["mv"] - np.maximum(rows["sal"], LO), 0.0)
    per = rows.groupby("player_id").agg(pick=("pick", "first"), surplus=("surplus", "sum"))
    b = wls(design(per["pick"]), per["surplus"].to_numpy(), np.ones(len(per)))
    out = pd.DataFrame({"pick": PICKS, "mkt_share_4y": design(PICKS) @ b})
    out["mkt_share_4y_raw"] = per.groupby("pick")["surplus"].mean().reindex(PICKS).to_numpy()
    return out, len(per)


def plot(df: pd.DataFrame, pred: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    c = np.where(df["share"] >= HI, orange, np.where(df["share"] <= LO, grid, blue))
    ax.scatter(pred * 100, df["share"] * 100, s=7, c=c, alpha=0.6, lw=0)
    ax.plot([0, 40], [0, 40], color=muted, lw=0.8)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xlabel("predicted, % of cap (season left out)", color=muted, fontsize=8.5)
    ax.set_ylabel("signed for, % of cap", color=muted, fontsize=8.5)
    ax.set_title("What the market pays: new veteran contracts", loc="left", fontsize=10, color=ink)
    ax.text(1, 37, "orange = max deals, grey = minimum deals", fontsize=7.5, color=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "market_value.png", facecolor=surface)


if __name__ == "__main__":
    sal = salary_series()
    draft = pd.read_parquet(RAW_DIR / "draft.parquet")
    stats = stats_table()
    sg = signings(sal, draft)
    sg = sg.merge(features(stats, sg), on=["player_id", "season"])
    # nobody in the data the 3 seasons before (overseas, G League): the market has no box score for him
    sg = sg[sg["gp3"] > 0].reset_index(drop=True)
    print(f"{len(sg):,} veteran signings {season_label(sg['season'].min())} to {season_label(sg['season'].max())}: "
          f"{(sg['share'] <= LO).mean():.0%} near-minimum, {(sg['share'] >= HI).mean():.0%} max")

    print("\nleave one season out, deals between min and max, error in % of cap:")
    res = {}
    for name, cols in (("age only", ["age", "age2"]), ("last season box", ["pts1", "reb1", "ast1", "mpg1", "gp1",
                                                                            "age", "age2"]),
                       ("box, 1 + 3 seasons", BOX), ("box + our RAPM", BOX + RAPM), ("RAPM only", RAPM + ["age", "age2", "mpg3", "gp3"])):
        b = backtest(sg, cols)
        res[name] = b
        print(f"  {name:20s} MAE {b['mae'] * 100:.2f}  RMSE {b['rmse'] * 100:.2f}  r {b['r']:.3f}  "
              f"max deals called max {b['max_hit']:.0%}   (n={b['n']})")

    use = BOX + RAPM if res["box + our RAPM"]["rmse"] < res["box, 1 + 3 seasons"]["rmse"] else BOX
    m = fit(sg, use)
    print(f"\nmodel used: {len(use)} features, sigma {m['sigma'] * 100:.1f}% of cap. standardized coefficients:")
    for c, b in sorted(zip(use, m["beta"][1:]), key=lambda x: -abs(x[1])):
        print(f"  {c:7s} {b * 100:+.2f}% of cap per sd")
    sg["pred"] = res["box + our RAPM" if use == BOX + RAPM else "box, 1 + 3 seasons"]["pred"]
    sg.to_parquet(PROCESSED_DIR / "market_signings.parquet", index=False)
    (PROCESSED_DIR / "market_model.json").write_text(json.dumps(
        dict(cols=use, mu=list(m["mu"]), sd=list(m["sd"]), beta=list(m["beta"]), sigma=m["sigma"]), indent=1))

    mv = market_now(m, stats)
    names = pd.read_parquet(PROCESSED_DIR / "surplus.parquet").set_index("player_id")[["name", "surplus"]]
    mv = mv.join(names, on="player_id")
    mv.to_parquet(PROCESSED_DIR / "market_value.parquet", index=False)
    show = mv.assign(**{c: mv[c] / 1e6 for c in ("salary", "market", "market_surplus", "surplus")})
    cols = ["name", "age", "pts3", "years", "salary", "market", "market_surplus", "surplus"]
    print("\nmost underpaid by the market's own price ($M, whole contract):")
    print(show.sort_values("market_surplus", ascending=False)[cols].head(12).round(1).to_string(index=False))
    print("\nmost overpaid by the market's own price:")
    print(show.sort_values("market_surplus")[cols].head(8).round(1).to_string(index=False))
    plot(sg, sg["pred"].to_numpy())

    pk, n = pick_market_value(m, stats, sal, draft)
    cap_next = CAP[max(SEASONS) + 1]
    pk["mkt_4y"] = pk["mkt_share_4y"] * cap_next
    pk.to_parquet(PROCESSED_DIR / "pick_market_value.parquet", index=False)
    print(f"\npick value in market terms, first 4 seasons ({n} picks), at the {season_label(max(SEASONS) + 1)} cap:")
    print(pk.iloc[[0, 2, 4, 9, 14, 19, 29, 44, 59]].assign(mkt_4y=lambda x: x["mkt_4y"] / 1e6)
          [["pick", "mkt_share_4y", "mkt_4y"]].round(3).to_string(index=False))
    print(f"  average 1st (1-30) ${pk['mkt_4y'][:30].mean() / 1e6:.1f}M, average 2nd ${pk['mkt_4y'][30:].mean() / 1e6:.1f}M")
