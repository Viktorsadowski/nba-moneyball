#!/usr/bin/env python3
"""
Step 10: what is every player worth, against what he's paid.

  $ per WAR   what the market pays for a win: every dollar above the near-minimum salary, divided
              by the WAR it bought (projected WAR, since that's what teams pay for when they sign).
              per season, so it follows the cap up
  future WAR  year 1 = war.py's projection (age-adjusted value, expected games from the injury model).
              later years: same player moved further along the aging curve, same availability.
              floored at 0, a team can always bench a guy who's below replacement
  surplus     sum over his remaining contract years of  WAR * $/WAR (grown with the cap) - salary

Contracts come from bbref's current contracts page. Option years (options.py): a team option year is worth
E[max(worth - salary, 0)] to the team, a player option year E[min(worth - salary, 0)], with the spread of the
projection from our own history. Non-guaranteed years count as team options. surplus_noopt = the old way,
every year as if guaranteed.

Output: data/processed/surplus.parquet

  python src/surplus.py
"""

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from options import contract_options, player_option_value, projection_sd, team_option_value
from scrape_salaries import unmojibake
from war import MIN_SALARY_Q, compact, last_name, salaries_with_ids, unsuffix

GROWTH_YEARS = 10     # cap growth estimated over the last 10 seasons


def aging_fn():
    c = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    return np.poly1d(np.polyfit(c["age"], c["curve"], 2))


def contracts_with_ids() -> pd.DataFrame:
    """future contract years -> player ids, matched on the last three seasons' players."""
    c = pd.read_parquet(RAW_DIR / "contracts.parquet")
    c["player"] = c["player"].map(unmojibake)
    p = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet") for y in list(SEASONS)[-3:]])
    p = p.sort_values("SEASON").drop_duplicates("PLAYER_ID", keep="last")
    full, by_team = {}, {}
    for pid, n, t in zip(p["PLAYER_ID"], p["PLAYER_NAME"], p["TEAM_ABBREVIATION"]):
        for k in {compact(n), unsuffix(compact(n))}:
            full.setdefault(k, set()).add(pid)
        by_team.setdefault((last_name(n), t), set()).add(pid)

    def find(n, t):
        for hit in (full.get(compact(n)), full.get(unsuffix(compact(n))), by_team.get((last_name(n), t))):
            if hit and len(hit) == 1:
                return next(iter(hit))
        return None

    c["player_id"] = [find(n, t) for n, t in zip(c["player"], c["team"])]
    miss = c[c["player_id"].isna()]["player"].unique()
    print(f"contracts: {c['player'].nunique()} players, {len(miss)} not matched to our data "
          f"(mostly 2026 draftees / never played): {', '.join(miss[:6])}...")
    return c.dropna(subset=["player_id"])


def dollars_per_war(war: pd.DataFrame, sal: pd.DataFrame) -> pd.Series:
    # season s salaries vs the WAR projected for season s (as of s-1)
    proj = war[["player_id", "season", "war"]].assign(season=war["season"] + 1)
    m = sal.merge(proj, on=["season", "player_id"], how="inner")
    floor = m.groupby("season")["salary"].transform(lambda x: x.quantile(MIN_SALARY_Q))
    m["above_min"] = (m["salary"] - floor).clip(lower=0)
    g = m.groupby("season")
    return g["above_min"].sum() / g["war"].apply(lambda x: x.clip(lower=0).sum())


if __name__ == "__main__":
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    sal = salaries_with_ids()

    dpw = dollars_per_war(war, sal)
    payroll = sal.groupby("season")["salary"].sum()
    last = int(payroll.index.max())
    g = (payroll[last] / payroll[last - GROWTH_YEARS]) ** (1 / GROWTH_YEARS) - 1
    print("$ per WAR by season: " + ", ".join(f"{season_label(int(s))} ${v / 1e6:.1f}M" for s, v in dpw.items()
                                             if s >= last - 4))
    print(f"league payroll growth: {g:.1%} a year over the last {GROWTH_YEARS} seasons, used for future years")

    # ── year by year projection over the contract ──
    now = war[war["season"] == last].set_index("player_id")
    age_fn = aging_fn()
    pace = prm["pace"][str(last)]
    k = pace / 100 / prm["ppw"]
    con = contracts_with_ids()
    con = con[con["player_id"].isin(now.index)].copy()
    # a few players show up twice in a season (two name spellings, stretched money), keep the bigger one
    con = con.sort_values("salary").drop_duplicates(["player_id", "season"], keep="last")
    con["year"] = con["season"] - last            # 1 = next season
    p = now.loc[con["player_id"]]
    # value in raw units is already aged to next season, move it (year - 1) more years
    raw = p["val"].to_numpy() + age_fn(p["age_next"].to_numpy() + con["year"].to_numpy() - 1) \
        - age_fn(p["age_next"].to_numpy())
    # next season carries the current injury (fewer games, rust in val_adj), later seasons don't
    first = con["year"].to_numpy() == 1
    adj = np.where(first, p["val_adj"].fillna(0).to_numpy(), 0)
    later = p["proj_min_later"].to_numpy() if "proj_min_later" in p.columns else p["proj_min"].to_numpy()
    mins = np.where(first, p["proj_min"].to_numpy(), later)
    val_cal = prm["a"] * raw + prm["c"] + prm["a"] * adj
    con["war_mu"] = (val_cal - prm["repl"]) * mins * k
    con["war"] = con["war_mu"].clip(lower=0)
    con["dpw"] = dpw[last] * (1 + g) ** con["year"]
    con["worth"] = con["war"] * con["dpw"]
    con["surplus_noopt"] = con["worth"] - con["salary"]

    # ── options (options.py): a team option year is worth E[max(W - salary, 0)], a player option year
    # E[min(W - salary, 0)], W uncertain. next season's options count as decided (picked up = guaranteed)
    opt = contract_options()
    # same player twice in a season (stretched money from an old team): keep the bigger row, like above
    opt = opt.sort_values("salary").drop_duplicates(["player", "team", "season"], keep="last")
    con = con.merge(opt[["player", "team", "season", "option"]], on=["player", "team", "season"], how="left")
    con.loc[con["year"] == 1, "option"] = None
    sd_tab = projection_sd()
    h = con["year"].clip(2, max(sd_tab)).to_numpy()
    alpha = np.array([sd_tab[x][0] for x in h])
    beta = np.array([sd_tab[x][1] for x in h])
    sd = alpha + beta * con["war"].to_numpy()
    mu, sal, d = con["war_mu"].to_numpy(), con["salary"].to_numpy(), con["dpw"].to_numpy()
    con["surplus"] = con["surplus_noopt"]
    tm, pl = (con["option"] == "team").to_numpy(), (con["option"] == "player").to_numpy()
    con.loc[tm, "surplus"] = team_option_value(mu[tm], sd[tm], sal[tm], d[tm])
    con.loc[pl, "surplus"] = player_option_value(mu[pl], sd[pl], sal[pl], d[pl])
    con["option_value"] = con["surplus"] - con["surplus_noopt"]
    print(f"option years valued: {tm.sum()} team (incl. non-guaranteed), {pl.sum()} player")

    s = con.groupby("player_id").agg(years=("year", "size"), salary=("salary", "sum"), war=("war", "sum"),
                                     worth=("worth", "sum"), surplus=("surplus", "sum"),
                                     surplus_noopt=("surplus_noopt", "sum"), option_value=("option_value", "sum"),
                                     team_opt=("option", lambda x: (x == "team").sum()),
                                     player_opt=("option", lambda x: (x == "player").sum()))
    nxt = con[con["year"] == 1].set_index("player_id")
    s["salary_next"] = nxt["salary"]
    s["war_next"] = nxt["war"]
    s["surplus_next"] = nxt["surplus"]
    s["name"] = now["name"]
    s["age_next"] = now["age_next"]
    s.reset_index().to_parquet(PROCESSED_DIR / "surplus.parquet", index=False)

    def show(df, title):
        print(f"\n{title}")
        t = df.assign(salary=df["salary"] / 1e6, worth=df["worth"] / 1e6, surplus=df["surplus"] / 1e6)
        print(t[["name", "age_next", "years", "salary", "war", "worth", "surplus"]].round(1).to_string(index=False))

    print(f"\n{len(s)} players with a contract past {season_label(last)}. money in $M, whole remaining contract")
    show(s.sort_values("surplus", ascending=False).head(15), "MOST UNDERPAID (surplus value)")
    show(s.sort_values("surplus").head(15), "MOST OVERPAID")

    t = s.assign(option_value=s["option_value"] / 1e6, surplus=s["surplus"] / 1e6, surplus_noopt=s["surplus_noopt"] / 1e6)
    cols = ["name", "age_next", "years", "team_opt", "player_opt", "surplus_noopt", "option_value", "surplus"]
    print("\nOPTIONS: biggest gains for the team (team options on uncertain players)")
    print(t.sort_values("option_value", ascending=False)[cols].head(8).round(1).to_string(index=False))
    print("\nOPTIONS: biggest losses for the team (player options, he leaves when he's underpaid)")
    print(t.sort_values("option_value")[cols].head(8).round(1).to_string(index=False))
