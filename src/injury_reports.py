#!/usr/bin/env python3
"""
Step 5c: official NBA injury reports, 2021-22 onwards (the league only publishes them since then).

One PDF per game day, the 5 PM ET snapshot. Lists every player on that day's report with
  Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason
where Reason separates actual injuries ("Injury/Illness - Left Ankle; Sprain") from
rest, G League, suspension, personal reasons etc. Better than the PST log for the question.

Download first (cached in data/raw/injury_reports/), then --parse turns the pdfs into
data/raw/injury_report_rows_{year}.parquet. Game days come from the team schedules, so run players.py first.

URL format changed on 2025-12-22: "..._05PM.pdf" before, "..._05_00PM.pdf" after.

  python src/injury_reports.py
  python src/injury_reports.py --seasons 2024 --max-days 3     # quick test
  python src/injury_reports.py --parse
"""

import argparse
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from config import RAW_DIR, season_label

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
NEW_FORMAT = pd.Timestamp("2025-12-22")
FIRST = 2021
CACHE = RAW_DIR / "injury_reports"


def report_url(day: pd.Timestamp) -> tuple[str, str]:
    t = "05PM" if day < NEW_FORMAT else "05_00PM"
    d = day.strftime("%Y-%m-%d")
    return URL.format(date=d, time=t), f"Injury-Report_{d}_{t}.pdf"


def download_season(year: int, max_days: int | None) -> None:
    sched = pd.read_parquet(RAW_DIR / f"team_games_{year}.parquet")
    days = sorted(pd.to_datetime(sched["GAME_DATE"]).dt.normalize().unique())
    if max_days:
        days = days[:max_days]

    folder = CACHE / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    got = have = missing = 0
    for day in days:
        url, name = report_url(pd.Timestamp(day))
        f = folder / name
        if f.exists():
            have += 1
            continue
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                f.write_bytes(r.read())
            got += 1
        except urllib.error.HTTPError as e:
            missing += 1
            if missing <= 3:
                print(f"    {name}: HTTP {e.code}")
        time.sleep(0.5)
    print(f"  {season_label(year)}: {len(days)} game days, {got} downloaded, {have} cached, {missing} not found")


# ── parsing ───────────────────────────────────────────────────────────────────
# The PDFs have no spaces in the text layer ("HardawayJr.,Tim", "IndianaPacers"), and a long
# reason wraps onto lines above and below the player's row. So: read words with positions,
# find the columns from the header, then hang every reason fragment on the nearest player row.
# Game/time/matchup/team only show on the first row of their block, carried down from there.

COLS = {"date": "Date", "time": "Time", "matchup": "Matchup", "team": "Team",
        "player": "Player", "status": "Status", "reason": "Reason"}


def _columns(words) -> dict[str, float] | None:
    hdr = next((w for w in words if w["text"].startswith("Reason")), None)
    if hdr is None:
        return None
    line = sorted((w for w in words if abs(w["top"] - hdr["top"]) < 3), key=lambda w: w["x0"])
    # 2021-23 pdfs have real spaces, so "Game Date" is two words. glue words with a small gap
    # into one header phrase, the column starts where the phrase starts
    phrases = []
    for w in line:
        if phrases and w["x0"] - phrases[-1]["x1"] < 8:
            phrases[-1]["text"] += w["text"]
            phrases[-1]["x1"] = w["x1"]
        else:
            phrases.append(dict(text=w["text"], x0=w["x0"], x1=w["x1"]))
    x = {}
    for k, key in COLS.items():
        hit = [ph["x0"] for ph in phrases if key in ph["text"]]
        if hit:
            x[k] = min(hit)
    return x if {"player", "status", "reason", "team"} <= set(x) else None


def parse_pdf(path) -> pd.DataFrame:
    import pdfplumber

    rows, carry = [], {}
    cols = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            c = _columns(words)
            if c is not None:
                cols = c
                hdr_top = next(w["top"] for w in words if w["text"].startswith("Reason"))
            elif cols is None:
                continue
            else:
                hdr_top = 60  # later pages: just the title line on top
            edges = sorted(cols.items(), key=lambda kv: kv[1])

            def col_of(x0):
                name = edges[0][0]
                for k, x in edges:
                    if x0 >= x - 6:
                        name = k
                return name

            body = [w for w in words if w["top"] > hdr_top + 3 and not w["text"].startswith("Page")]
            # one line = words within 2pt of each other vertically, per column
            cells: dict[tuple, list] = {}
            for w in body:
                cells.setdefault((col_of(w["x0"]), round(w["top"])), []).append(w)

            players = sorted((t, " ".join(x["text"] for x in ws)) for (k, t), ws in cells.items() if k == "player")
            if not players:
                continue
            p_tops = np.array([t for t, _ in players], dtype=float)

            # carry-down fields, keyed on the top they sit at
            marks = {k: sorted((t, " ".join(x["text"] for x in ws)) for (kk, t), ws in cells.items() if kk == k)
                     for k in ("date", "time", "matchup", "team")}
            status = {t: " ".join(x["text"] for x in ws) for (k, t), ws in cells.items() if k == "status"}
            reason_bits: dict[int, list] = {}
            for (k, t), ws in cells.items():
                if k == "reason":
                    i = int(np.argmin(np.abs(p_tops - t)))
                    reason_bits.setdefault(i, []).append((t, " ".join(x["text"] for x in ws)))

            for i, (t, name) in enumerate(players):
                for k, ms in marks.items():
                    for mt, mv in ms:
                        if mt <= t + 2:
                            carry[k] = mv
                st = next((v for tt, v in status.items() if abs(tt - t) <= 2), "")
                reason = "".join(v for _, v in sorted(reason_bits.get(i, [])))
                # teams that haven't filed yet show "NOT YET SUBMITTED" in the reason column, not his
                reason = re.sub(r"NOT\s*YET\s*SUBMITTED", "", reason).strip()
                rows.append(dict(**{k: carry.get(k, "") for k in ("date", "time", "matchup", "team")},
                                 player=name, status=st, reason=reason))
    return pd.DataFrame(rows)


def parse_season(year: int) -> None:
    folder = CACHE / str(year)
    files = sorted(folder.glob("*.pdf"))
    parts = []
    for f in files:
        try:
            d = parse_pdf(f)
        except Exception as e:  # a broken pdf shouldn't kill the season
            print(f"    {f.name}: {e.__class__.__name__} {e}")
            continue
        d["report"] = f.stem
        parts.append(d)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df = df[df["player"].str.contains(",")]  # drops "NOT YET SUBMITTED" rows
        d = df["date"].astype(str).str.extract(r"(\d{2}/\d{2}/\d{4})", expand=False)
        df["date"] = pd.to_datetime(d, format="%m/%d/%Y", errors="coerce")
        df.insert(0, "season", year)
        df.to_parquet(RAW_DIR / f"injury_report_rows_{year}.parquet", index=False)
    n_inj = df["reason"].str.startswith("Injury").sum() if len(df) else 0
    no_date = df["date"].isna().mean() if len(df) else 0
    print(f"  {season_label(year)}: {len(files)} reports -> {len(df)} rows, {n_inj} injury rows, "
          f"{no_date:.1%} without a date")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(range(FIRST, 2026)))
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--parse", action="store_true", help="parse the downloaded pdfs instead of downloading")
    args = ap.parse_args()
    for y in args.seasons:
        if args.parse:
            parse_season(y)
        else:
            download_season(y, args.max_days)
