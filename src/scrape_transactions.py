#!/usr/bin/env python3
"""
Every transaction since 2011-12 from basketball-reference.com (leagues/NBA_2024_transactions.html etc.),
one page per season, for the trade value model (trade_value.py): in real trades, what did teams give for
what, so we can back out what the league pays for a pick.

Only downloads, the parsing is in trade_value.py. Pages cached in data/raw/salary_pages/ like the salary
pages (same 3.5 s between requests), safe to rerun. 16 pages, about a minute.

bbref blocks cloud servers, run it on your own machine:
  python src/scrape_transactions.py
"""

import requests

from config import SEASONS
from scrape_salaries import BASE, get

FIRST = 2012        # bbref names seasons by the year they end: NBA_2012 = 2011-12 (July 2011 to June 2012)


if __name__ == "__main__":
    sess = requests.Session()
    last = max(SEASONS) + 2       # the current offseason sits on next season's page (NBA_2027 = summer 2026 on)
    for y in range(FIRST, last + 1):
        html = get(sess, f"{BASE}/leagues/NBA_{y}_transactions.html", f"transactions_{y}.html")
        print(f"NBA_{y}: {len(html) / 1e3:.0f} kB, {html.count(' traded ')} trades mentioned")
