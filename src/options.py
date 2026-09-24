#!/usr/bin/env python3
"""
Contract options, for surplus.py.

  which years      from the saved bbref contracts page (scrape_salaries.py caches it): cells marked
                   salary-tm = team option, salary-pl = player option. years that aren't options but aren't
                   in "Guaranteed" either are non-guaranteed; the team can cut him, so they count as a team
                   option too (the last years of the deal first)
  how uncertain    how far off is our WAR projection h years out, compared with the projection the team
                   will have when it decides (the summer before that season)? measured on our own history:
                   projection made as of season s for s+h vs the one made as of s+h-1. players who dropped
                   out of the league count as 0. spread grows with the projection, so sd = alpha + beta * WAR
  value            worth in a year W = $/WAR * max(WAR, 0), WAR ~ Normal(projection, sd)
                   team option:    team keeps the year only if W > salary   -> E[max(W - salary, 0)]
                   player option:  he leaves if W > salary (he gets more elsewhere) -> E[min(W - salary, 0)]
                   closed form (Bachelier, normal outcome), no simulation needed

Options in next season (year 1) are treated as already decided: the decisions for 2026-27 were due
over the summer, so what's still on the page is what got picked up.

  python src/options.py      # just prints the sd table + option counts, surplus.py uses the functions
"""

import json
import re

import lxml.html
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import PROCESSED_DIR, RAW_DIR
from scrape_salaries import unmojibake

PAGE = RAW_DIR / "salary_pages" / "contracts_players.html"
MAX_H = 5


def money(s: str) -> float:
    s = re.sub(r"[^\d]", "", s or "")
    return float(s) if s else np.nan


def contract_options() -> pd.DataFrame:
    """one row per player, team, season: salary + option ('team' / 'player' / None)."""
    html = PAGE.read_text(encoding="utf-8").replace("<!--", "").replace("-->", "")
    tab = lxml.html.fromstring(html).get_element_by_id("player-contracts")
    head = [th.text_content().strip() for th in tab.xpath(".//thead/tr[last()]/th")]
    seasons = {f"y{i + 1}": int(h[:4]) for i, h in enumerate(h for h in head if re.fullmatch(r"\d{4}-\d{2}", h))}
    rows = []
    for tr in tab.xpath(".//tbody/tr[not(contains(@class,'thead'))]"):
        cells = {c.get("data-stat"): c for c in tr}
        if "player" not in cells:
            continue
        name = unmojibake(cells["player"].text_content().strip())
        team = cells["team_id"].text_content().strip()
        gtd = money(cells["remain_gtd"].text_content())
        yrs = []
        for k, season in seasons.items():
            c = cells.get(k)
            sal = money(c.text_content()) if c is not None else np.nan
            if np.isnan(sal):
                continue
            cls = c.get("class") or ""
            opt = "team" if "salary-tm" in cls else "player" if "salary-pl" in cls else None
            yrs.append([name, team, season, sal, opt])
        # non-guaranteed money on non-option years = the team can walk away: team option, latest years first
        ng = sum(y[3] for y in yrs if y[4] is None) - (gtd if not np.isnan(gtd) else 0)
        for y in reversed(yrs):
            if ng <= 0:
                break
            if y[4] is None and y[2] > min(seasons.values()):
                y[4] = "team"
                ng -= y[3]
        rows += yrs
    out = pd.DataFrame(rows, columns=["player", "team", "season", "salary", "option"])
    # bbref abbreviations -> nba.com, same as the contracts table
    out["team"] = out["team"].replace({"PHO": "PHX", "BRK": "BKN", "CHO": "CHA", "NOH": "NOP", "NJN": "BKN"})
    return out


def projection_sd() -> dict:
    """{h: (alpha, beta)}: sd of the WAR projection h years out vs the one made the summer before."""
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    c = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    age_fn = np.poly1d(np.polyfit(c["age"], c["curve"], 2))
    by = {s: g.set_index("player_id") for s, g in war.groupby("season")}
    last = max(by)
    out = {}
    for h in range(2, MAX_H + 1):
        err, pred = [], []
        for s, p in by.items():
            if s + h - 1 > last:
                continue
            p = p[p["proj_min"] >= 500]
            k = prm["pace"][str(s)] / 100 / prm["ppw"]
            raw = p["val"] + age_fn(p["age_next"] + h - 1) - age_fn(p["age_next"])
            hat = (prm["a"] * raw + prm["c"] - prm["repl"]) * p["proj_min"] * k
            tgt = by[s + h - 1]["war"].reindex(p.index).fillna(0).clip(lower=0)   # gone from the league = 0
            err.append(tgt - hat)
            pred.append(hat.clip(lower=0))
        e, w = np.concatenate(err), np.concatenate(pred)
        # sd as a straight line in the projection: fit |error| * sqrt(pi/2) (mean abs error -> sd for a normal)
        b = np.polyfit(w, np.abs(e - e.mean()) * np.sqrt(np.pi / 2), 1)
        out[h] = (max(b[1], 0.1), max(b[0], 0.0), float(e.mean()), len(e))
    return out


def team_option_value(m, sd, salary, dpw):
    """E[max(W - salary, 0)], W = dpw * max(N(m, sd), 0)."""
    s = salary / dpw
    d = (m - s) / sd
    return dpw * (sd * norm.pdf(d) + (m - s) * norm.cdf(d))


def expected_worth(m, sd, dpw):
    d = m / sd
    return dpw * (sd * norm.pdf(d) + m * norm.cdf(d))


def player_option_value(m, sd, salary, dpw):
    """E[min(W - salary, 0)] = E[W] - salary - E[max(W - salary, 0)] ... + the part above: team's side when he
    stays only if he's worth less than he's paid."""
    return expected_worth(m, sd, dpw) - salary - team_option_value(m, sd, salary, dpw)


if __name__ == "__main__":
    o = contract_options()
    print(o["option"].value_counts(dropna=False).to_string())
    print(o.groupby(["season", "option"]).size().unstack(fill_value=0).to_string())
    print("\nWAR projection spread, h seasons out (vs the projection the summer before):")
    for h, (a, b, bias, n) in projection_sd().items():
        print(f"  h={h}: sd = {a:.2f} + {b:.2f} * WAR   (mean error {bias:+.2f}, n={n})")
