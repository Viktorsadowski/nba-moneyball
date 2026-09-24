#!/usr/bin/env python3
"""
Step 5b: injury log 2010-11 to 2019-20.

prosportstransactions.com sits behind a Cloudflare bot check now, so no live scraping.
Someone already scraped it in 2020 (github.com/gboogy/nba-injury-data-scraper), same 5 columns:
  Date | Team | Acquired (= came back) | Relinquished (= went out) | Notes

Covers Oct 2010 to Oct 2020 = seasons 2010-11 .. 2019-20.
2021-22 onwards comes from the official NBA injury reports instead (injury_reports.py).
2020-21 has neither, availability for that season is games played only.

Output: data/raw/injuries_{year}.parquet

  python src/injuries_pst.py
"""

import io
import urllib.request

import pandas as pd

from config import RAW_DIR, season_label

URL = "https://raw.githubusercontent.com/gboogy/nba-injury-data-scraper/master/data/injuries_2010-2020.csv"


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "pst_injuries_2010-2020.csv"
    if not cache.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            cache.write_bytes(r.read())

    df = pd.read_csv(cache, dtype=str, keep_default_na=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in ("Acquired", "Relinquished"):
        df[c] = df[c].str.replace("•", "", regex=False).str.strip()

    # season = the year it started in, anything from august on belongs to the next season
    df["season"] = df["Date"].dt.year - (df["Date"].dt.month < 8).astype(int)
    for y, g in df.groupby("season"):
        if y > 2019:
            continue  # only the preseason bit of 2020-21 is in there
        g[["season", "Date", "Team", "Acquired", "Relinquished", "Notes"]].to_parquet(
            RAW_DIR / f"injuries_{y}.parquet", index=False)
        print(f"  {season_label(y)}: {len(g)} log entries")
