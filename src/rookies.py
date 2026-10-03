#!/usr/bin/env python3
"""
Rookie model: what a draft pick is worth in his first seasons, and what a pick is worth in a trade.

Until now every rookie got the same value (the average first-year player) and 800 minutes. A #1 pick and
a #50 pick are not the same player, so:

  1. rookie season    value per 100 (O and D, from RAPM with the box prior) and minutes, as a smooth
                      function of log(pick), fitted on the 2010-2025 drafts. backtest: leave one draft out,
                      compare with the flat average we used before
  2. young players    our projection (value.py) is too low for players in seasons 2-4, most for top picks
                      (they still get better, the aging curve only knows age). fix: a bump by pick group x
                      season in the league, learned from earlier seasons' errors. tested with value.py's
                      backtest, only using seasons before the target
  3. pick value       realized WAR in the 4 rookie-scale seasons, minus what those players were paid, by
                      pick, in wins at each season's price. that's the trade value of a pick

needs data/raw/draft.parquet (python src/draft.py, on your own machine)

Output
  data/processed/rookie_model.parquet    per pick 1-60: expected o, d, value, minutes, pick surplus
  data/processed/rookies_next.parquet    next season's rookies (last draft) with their projection
  data/processed/rookie_dev.parquet      development bump for seasons 2-4, by pick group (season_sim.py)
  figures/rookies.png

  python src/rookies.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label

MIN_MIN = 300          # rookies with real minutes teach the value model
PICKS = np.arange(1, 61)
RNG = np.random.default_rng(4)
GHOST = 250.0          # value.py's ghost minutes
GROUPS = [0, 5, 14, 30, 60]    # pick groups: 1-5, 6-14 (lottery), 15-30, 2nd round
DEV_SHRINK = 25_000    # ghost minutes pulling the development bump to 0. backtest: 0 1.5869, 10k-25k 1.5852, 100k 1.5915


def load():
    draft = pd.read_parquet(RAW_DIR / "draft.parquet")
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")[["season", "player_id", "o_rapm", "d_rapm", "rapm",
                                                             "minutes"]]
    ages = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["PLAYER_ID", "AGE"]].assign(season=y)
                      for y in SEASONS])
    ages.columns = ["player_id", "age", "season"]
    return draft, rapm, ages


def design(pick) -> np.ndarray:
    lp = np.log(np.asarray(pick, float))
    return np.column_stack([np.ones_like(lp), lp, lp ** 2])


def wls(X, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def rookie_table(draft, rapm, ages) -> pd.DataFrame:
    r = draft[draft["draft_year"] <= max(SEASONS)].copy()
    r["season"] = r["draft_year"]              # rookie season = the one starting that fall
    r = r.merge(rapm, on=["season", "player_id"], how="left").merge(ages, on=["season", "player_id"], how="left")
    r["minutes"] = r["minutes"].fillna(0)
    return r


# ── 1. rookie season ──────────────────────────────────────────────────────────

def fit_rookie(r: pd.DataFrame, years=None) -> dict:
    """coefficients for o, d (value, minutes-weighted) and minutes (everyone, zeros included)."""
    t = r if years is None else r[r["draft_year"].isin(years)]
    v = t[t["minutes"] >= MIN_MIN]
    w = v["minutes"].clip(upper=2500).to_numpy()
    out = {side: wls(design(v["pick"]), v[f"{side}_rapm"].to_numpy(), w) for side in ("o", "d")}
    out["min"] = wls(design(t["pick"]), t["minutes"].to_numpy(), np.ones(len(t)))
    return out


def predict(coef: dict, pick) -> pd.DataFrame:
    X = design(pick)
    o, d, m = X @ coef["o"], X @ coef["d"], np.clip(X @ coef["min"], 0, None)
    return pd.DataFrame({"pick": pick, "o": o, "d": d, "val": o + d, "min": m})


def backtest_rookies(r: pd.DataFrame, flat_val: float, flat_min: float) -> dict:
    """leave one draft out. value: minutes-weighted rmse on rookies with 500+ min. minutes: MAE, everyone."""
    ev = []
    for y in sorted(r["draft_year"].unique()):
        coef = fit_rookie(r, [x for x in r["draft_year"].unique() if x != y])
        t = r[r["draft_year"] == y]
        p = predict(coef, t["pick"].to_numpy())
        ev.append(t.assign(p_val=p["val"].to_numpy(), p_min=p["min"].to_numpy()))
    ev = pd.concat(ev)
    v = ev[ev["minutes"] >= 500]
    w = v["minutes"]
    rm = lambda e: float(np.sqrt(np.average(e ** 2, weights=w)))
    return {"n_val": len(v), "rmse_flat": rm(v["rapm"] - flat_val), "rmse_model": rm(v["rapm"] - v["p_val"]),
            "mae_min_flat": float((ev["minutes"] - flat_min).abs().mean()),
            "mae_min_model": float((ev["minutes"] - ev["p_min"]).abs().mean()), "n_min": len(ev)}


# ── 2. young players ──────────────────────────────────────────────────────────

def young_player_bias(draft, rapm) -> pd.DataFrame:
    """value.py projection (as of s-1) vs RAPM in s, for players in their 2nd-4th season, by pick group."""
    val = pd.read_parquet(PROCESSED_DIR / "value.parquet")[["season", "player_id", "val"]]
    val = val.assign(season=val["season"] + 1)          # projection made as of s-1 for season s
    d = rapm.merge(val, on=["season", "player_id"]).merge(draft[["player_id", "draft_year", "pick"]], on="player_id")
    d["year_in_league"] = d["season"] - d["draft_year"] + 1
    d = d[d["year_in_league"].between(2, 4) & (d["minutes"] >= 500)]
    d["err"] = d["rapm"] - d["val"]
    d["group"] = pd.cut(d["pick"], [0, 5, 14, 30, 60], labels=["1-5", "6-14", "15-30", "31-60"])
    g = d.groupby("group", observed=True)
    out = pd.DataFrame({"n": g.size(), "mean_err": g.apply(lambda x: np.average(x["err"], weights=x["minutes"]),
                                                           include_groups=False)})
    # bootstrap interval per group
    for k, sub in g:
        e, w = sub["err"].to_numpy(), sub["minutes"].to_numpy()
        b = [np.average(e[i], weights=w[i]) for i in (RNG.integers(0, len(e), len(e)) for _ in range(500))]
        out.loc[k, "lo"], out.loc[k, "hi"] = np.percentile(b, [2.5, 97.5])
    return out


def pick_group(pick) -> pd.Series:
    return pd.cut(pd.Series(pick, dtype=float), GROUPS, labels=False)


def young_errors(draft, rapm) -> pd.DataFrame:
    """value.py's backtest rows (projection as of t-1 vs RAPM in t, 500+ min) + pick and season in the league"""
    import value as V
    V.AGING = V.load_aging()
    r2 = V.add_ages(rapm.copy())
    dm = draft.drop_duplicates("player_id").set_index("player_id")
    rows = []
    for t in range(V.FIRST_TARGET, int(rapm["season"].max()) + 1):
        pred = V.blend(r2, t - 1, (0.5, 0.3, 0.2), GHOST)[["player_id", "o_val", "d_val", "val"]]
        tgt = rapm[(rapm["season"] == t) & (rapm["minutes"] >= V.TARGET_MIN)][
            ["player_id", "minutes", "o_rapm", "d_rapm", "rapm"]]
        m = tgt.merge(pred, on="player_id", how="left").assign(season=t)
        m["pick"] = m["player_id"].map(dm["pick"])
        m["yr"] = t - m["player_id"].map(dm["draft_year"]) + 1
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    d["grp"] = pick_group(d["pick"]).to_numpy()
    return d


def dev_bump(d: pd.DataFrame, side: str = "") -> pd.Series:
    """how much young players beat their projection, by pick group x season in the league (2-4).
    minutes-weighted mean error, shrunk to 0 with DEV_SHRINK ghost minutes. side "o_"/"d_" for O or D only"""
    pp = d[d["yr"].between(2, 4) & d["val"].notna() & d["grp"].notna()]
    e = (pp[f"{side}rapm"] - pp[f"{side}val"]) * pp["minutes"]
    g = pd.DataFrame({"e": e, "m": pp["minutes"], "grp": pp["grp"], "yr": pp["yr"]}).groupby(["grp", "yr"])
    return (g["e"].sum() / (g["m"].sum() + DEV_SHRINK)).rename("bump")


def young_backtest(draft, rapm, ages, d: pd.DataFrame) -> dict:
    """value.py's backtest three ways, everything learned only from earlier seasons:
    base = as now (rookies at 0), rookie = true rookies at their draft-slot value, dev = + the bump for yr 2-4"""
    tot = {k: np.zeros(2) for k in ("base", "rookie", "dev")}
    young = {k: np.zeros(2) for k in tot}
    first = int(d["season"].min()) + 3      # a few seasons to learn the bump from first
    for t in range(first, int(d["season"].max()) + 1):
        past, cur = d[d["season"] < t], d[d["season"] == t]
        coef = fit_rookie(rookie_table(draft[draft["draft_year"] < t], rapm, ages))
        pr = predict(coef, PICKS).set_index("pick")["val"]
        base = cur["val"].fillna(0.0)
        rook = base.where(cur["val"].notna() | cur["pick"].isna(), cur["pick"].map(pr))
        b = dev_bump(past)
        bump = pd.Series([b.get((g, y), 0.0) for g, y in zip(cur["grp"], cur["yr"])], index=cur.index)
        dev = rook + bump.where(cur["val"].notna(), 0.0)
        y = cur["yr"].between(1, 4)
        for k, p in (("base", base), ("rookie", rook), ("dev", dev)):
            se = cur["minutes"] * (cur["rapm"] - p) ** 2
            tot[k] += [se.sum(), cur["minutes"].sum()]
            young[k] += [se[y].sum(), cur["minutes"][y].sum()]
    out = {f"all_{k}": float(np.sqrt(v[0] / v[1])) for k, v in tot.items()}
    out.update({f"young_{k}": float(np.sqrt(v[0] / v[1])) for k, v in young.items()})
    out["targets"] = f"{first}-{int(d['season'].max())}"
    return out


# ── 3. pick value ─────────────────────────────────────────────────────────────

def pick_value(draft, rapm) -> pd.DataFrame:
    """realized WAR and salary over the 4 rookie-scale seasons, by pick, in wins at that season's price."""
    from surplus import dollars_per_war
    from war import salaries_with_ids
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    sal = salaries_with_ids()
    dpw = dollars_per_war(war, sal)
    a, c, repl, ppw = prm["a"], prm["c"], prm["repl"], prm["ppw"]
    r = rapm.merge(draft[["player_id", "draft_year", "pick"]], on="player_id")
    r = r[(r["season"] >= r["draft_year"]) & (r["season"] < r["draft_year"] + 4)]
    k = r["season"].astype(str).map(prm["pace"]) / 100 / ppw
    # realized WAR: season RAPM on the calibrated scale. a guy who's below replacement just doesn't play much
    r["war"] = ((a * r["rapm"] + c - repl) * r["minutes"] * k).clip(lower=0)
    r = r.merge(sal, on=["season", "player_id"], how="left")
    r["dpw"] = r["season"].map(dpw)
    r = r[r["dpw"].notna()]
    # in wins, so seasons with different price levels add up
    r["surplus_war"] = r["war"] - r["salary"].fillna(0) / r["dpw"]
    # only drafts with all 4 seasons in the data, and every drafted player counts (the busts too)
    full = draft[draft["draft_year"] <= max(SEASONS) - 3][["player_id", "draft_year", "pick"]]
    tot = r.groupby("player_id")[["war", "surplus_war"]].sum()
    full = full.join(tot, on="player_id").fillna({"war": 0.0, "surplus_war": 0.0})
    X = design(full["pick"])
    b_w = wls(X, full["war"].to_numpy(), np.ones(len(full)))
    b_s = wls(X, full["surplus_war"].to_numpy(), np.ones(len(full)))
    out = pd.DataFrame({"pick": PICKS, "war_4y": design(PICKS) @ b_w, "surplus_war_4y": design(PICKS) @ b_s})
    raw = full.groupby("pick")[["war", "surplus_war"]].mean().reindex(PICKS)
    out["war_4y_raw"], out["surplus_war_4y_raw"] = raw["war"].to_numpy(), raw["surplus_war"].to_numpy()
    return out, len(full)


def plot(model: pd.DataFrame, r: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    v = r[r["minutes"] >= MIN_MIN]
    a1.scatter(v["pick"], v["rapm"], s=np.clip(v["minutes"] / 120, 2, 25), color=muted, alpha=0.3, lw=0,
               label="rookie seasons (size = minutes)")
    a1.plot(model["pick"], model["val"], color=blue, lw=2.2, label="model: value per 100")
    a1.axhline(0, color=grid, lw=0.8)
    a1.set_xlabel("draft pick", color=muted, fontsize=8.5)
    a1.set_ylabel("rookie-season RAPM, per 100", color=muted, fontsize=8.5)
    a1.set_title("Rookie season by draft slot", loc="left", fontsize=10, color=ink)
    a1.legend(frameon=False, fontsize=8, labelcolor=ink)
    a2.bar(model["pick"], model["surplus_war_4y_raw"], color=grid, width=0.8, label="average, by pick")
    a2.plot(model["pick"], model["surplus_war_4y"], color=orange, lw=2.2, label="smoothed")
    a2.axhline(0, color=muted, lw=0.8)
    a2.set_xlabel("draft pick", color=muted, fontsize=8.5)
    a2.set_ylabel("wins above salary, first 4 seasons", color=muted, fontsize=8.5)
    a2.set_title("What a pick is worth: surplus over the rookie deal", loc="left", fontsize=10, color=ink)
    a2.legend(frameon=False, fontsize=8, labelcolor=ink)
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="y", color=grid, lw=0.6)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "rookies.png", facecolor=surface)
    print(f"\nplot -> {out / 'rookies.png'}")


if __name__ == "__main__":
    draft, rapm, ages = load()
    r = rookie_table(draft, rapm, ages)
    print(f"{len(r):,} drafted players {r['draft_year'].min()}-{r['draft_year'].max()}, "
          f"{(r['minutes'] >= MIN_MIN).sum():,} with {MIN_MIN}+ rookie minutes")

    # the flat numbers we used before (freeze.py / war.py): minutes-weighted first-season RAPM, 800 minutes
    first = rapm.sort_values("season").drop_duplicates("player_id")
    first = first[first["season"] > min(SEASONS)]
    flat_val = float(np.average(first["rapm"], weights=first["minutes"]))

    bt = backtest_rookies(r, flat_val, 800.0)
    print(f"\n1. ROOKIE SEASON, leave one draft out")
    print(f"   value (rookies with 500+ min, n={bt['n_val']}): RMSE flat {bt['rmse_flat']:.3f} -> model "
          f"{bt['rmse_model']:.3f} per 100")
    print(f"   minutes (all {bt['n_min']} picks): MAE flat 800 {bt['mae_min_flat']:.0f} -> model {bt['mae_min_model']:.0f}")
    coef = fit_rookie(r)
    model = predict(coef, PICKS)
    print(model.iloc[[0, 1, 2, 4, 9, 14, 19, 29, 39, 59]].round(2).to_string(index=False))

    print("\n2. YOUNG PLAYERS (seasons 2-4): projection error by pick (RAPM - projection, per 100)")
    print(young_player_bias(draft, rapm).round(2).to_string())
    d = young_errors(draft, rapm)
    yb = young_backtest(draft, rapm, ages, d)
    print(f"   value.py backtest {yb['targets']}, RMSE per 100:  now | + rookies at draft value | + development bump")
    print(f"     all players      {yb['all_base']:.4f} | {yb['all_rookie']:.4f} | {yb['all_dev']:.4f}")
    print(f"     seasons 1-4      {yb['young_base']:.4f} | {yb['young_rookie']:.4f} | {yb['young_dev']:.4f}")
    bump = dev_bump(d).reset_index()
    bump.to_parquet(PROCESSED_DIR / "rookie_dev.parquet", index=False)
    print("   bump used for next season (all seasons), per 100:")
    print(bump.assign(group=bump["grp"].map(dict(enumerate(["1-5", "6-14", "15-30", "31-60"]))))
          .pivot(index="group", columns="yr", values="bump").reindex(["1-5", "6-14", "15-30", "31-60"])
          .round(2).to_string())

    pv, n = pick_value(draft, rapm)
    model = model.merge(pv, on="pick")
    print(f"\n3. PICK VALUE, first 4 seasons ({n} picks with 4 seasons of data), smoothed")
    print(model.iloc[[0, 2, 4, 9, 14, 19, 29, 44, 59]][["pick", "war_4y", "surplus_war_4y"]].round(2)
          .to_string(index=False))
    model.to_parquet(PROCESSED_DIR / "rookie_model.parquet", index=False)

    # next season's rookies: the last draft
    last = draft[draft["draft_year"] == max(SEASONS) + 1]
    if len(last):
        nxt = last.merge(model[["pick", "o", "d", "val", "min"]], on="pick", how="left")
        nxt.to_parquet(PROCESSED_DIR / "rookies_next.parquet", index=False)
        print(f"\n{season_label(max(SEASONS) + 1)} rookies ({len(nxt)}), top of the draft:")
        print(nxt[["pick", "name", "team", "val", "min"]].head(10).round(2).to_string(index=False))
    plot(model, r)
