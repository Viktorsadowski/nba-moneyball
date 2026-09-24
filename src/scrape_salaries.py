#!/usr/bin/env python3
"""
Step 5c: salaries from basketball-reference.com.

  salaries_{year}.parquet   what every player was paid that season, from the 30 team pages
                            (teams/BOS/2015.html etc.), 2010-11 onwards
  contracts.parquet         current contracts incl. future years, from contracts/players.html

bbref allows ~20 requests a minute before it blocks you for a while, so 3.5 s between pages.
All seasons = ~480 pages = ~30 min. Pages are cached in data/raw/salary_pages/, safe to stop and rerun.

  python src/scrape_salaries.py
  python src/scrape_salaries.py --seasons 2025 --teams BOS LAL    # quick test
  python src/scrape_salaries.py --contracts-only
"""

import argparse
import io
import re
import time

import pandas as pd
import requests

from config import RAW_DIR, SEASONS, season_label

BASE = "https://www.basketball-reference.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
DELAY = 3.5
CACHE = RAW_DIR / "salary_pages"

FIXED = ["ATL", "BOS", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
# bbref -> nba.com abbreviation, so we can join on team ids later
TO_NBA = {"PHO": "PHX", "BRK": "BKN", "NJN": "BKN", "CHO": "CHA", "NOH": "NOP"}


def teams_for(year: int) -> list[str]:
    # the three franchises that changed name/abbreviation in our window
    end = year + 1
    nets = "NJN" if end <= 2012 else "BRK"
    cha = "CHA" if end <= 2014 else "CHO"
    nop = "NOH" if end <= 2013 else "NOP"
    return FIXED + [nets, cha, nop]


def unmojibake(s):
    # bbref sends utf-8 without saying so, requests then decodes it as latin-1: "JokiÄ" instead of
    # "Jokić". undo that when it round-trips cleanly, leave proper text alone
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def get(sess: requests.Session, url: str, cache_name: str) -> str:
    f = CACHE / cache_name
    if f.exists():
        return unmojibake(f.read_text(encoding="utf-8"))
    for attempt in range(4):
        r = sess.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            # rate limited, bbref wants you gone for a while
            print("    429 from bbref, waiting 60s")
            time.sleep(60)
            continue
        r.raise_for_status()
        r.encoding = "utf-8"
        break
    else:
        raise RuntimeError(f"gave up on {url}")
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    time.sleep(DELAY)
    return r.text


def money(s) -> float:
    s = re.sub(r"[^\d]", "", str(s))
    return float(s) if s else float("nan")


# ── season salaries from team pages ───────────────────────────────────────────

def team_salaries(sess, abbr: str, year: int) -> pd.DataFrame:
    html = get(sess, f"{BASE}/teams/{abbr}/{year + 1}.html", f"{abbr}_{year + 1}.html")
    # most bbref tables sit inside html comments, unhide them
    html = html.replace("<!--", "").replace("-->", "")
    try:
        t = pd.read_html(io.StringIO(html), attrs={"id": "salaries2"})[0]
    except ValueError:
        # fallback: any table with a Salary column
        t = next((x for x in pd.read_html(io.StringIO(html)) if "Salary" in map(str, x.columns)), None)
        if t is None:
            print(f"    no salary table for {abbr} {year + 1}")
            return pd.DataFrame()
    name_col = next(c for c in t.columns if str(c) in ("Player", "Unnamed: 1"))
    t = t.rename(columns={name_col: "player"})
    t = t[t["player"].notna() & (t["player"] != "Player")]
    return pd.DataFrame({
        "season": year,
        "team_bbref": abbr,
        "team": TO_NBA.get(abbr, abbr),
        "player": t["player"].astype(str),
        "salary": t["Salary"].map(money),
    })


def scrape_season(sess, year: int, teams: list[str] | None) -> None:
    out = RAW_DIR / f"salaries_{year}.parquet"
    if out.exists() and teams is None:
        print(f"  {season_label(year)}: already there")
        return
    parts = [team_salaries(sess, a, year) for a in (teams or teams_for(year))]
    df = pd.concat(parts, ignore_index=True)
    if teams is None:
        df.to_parquet(out, index=False)
    print(f"  {season_label(year)}: {len(df)} player-team rows, total ${df['salary'].sum() / 1e9:.2f}B")


# ── current contracts ─────────────────────────────────────────────────────────

def scrape_contracts(sess) -> None:
    html = get(sess, f"{BASE}/contracts/players.html", "contracts_players.html")
    t = pd.read_html(io.StringIO(html.replace("<!--", "").replace("-->", "")),
                     attrs={"id": "player-contracts"})[0]
    # two header rows ("Salary" over the season columns), keep the bottom one
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [c[-1] for c in t.columns]
    t = t[t["Player"].notna() & (t["Player"] != "Player")]
    seasons = [c for c in t.columns if re.fullmatch(r"\d{4}-\d{2}", str(c))]
    long = t.melt(id_vars=["Player", "Tm"], value_vars=seasons, var_name="season_label", value_name="salary")
    long["salary"] = long["salary"].map(money)
    long = long.dropna(subset=["salary"])
    long["season"] = long["season_label"].str[:4].astype(int)
    long["team"] = long["Tm"].map(lambda a: TO_NBA.get(a, a))
    long = long.rename(columns={"Player": "player"})[["player", "team", "season", "salary"]]
    long.to_parquet(RAW_DIR / "contracts.parquet", index=False)
    print(f"  contracts: {long['player'].nunique()} players, seasons {min(seasons)} to {max(seasons)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    ap.add_argument("--teams", nargs="*", default=None, help="bbref abbreviations, for testing (saves nothing)")
    ap.add_argument("--contracts-only", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    if not args.contracts_only:
        for y in args.seasons:
            print(season_label(y))
            scrape_season(sess, y, args.teams)
    scrape_contracts(sess)
