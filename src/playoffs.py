#!/usr/bin/env python3
"""
Playoff game results, for defense.py.

Same archive as ingest.py (github.com/shufinskiy/nba_data), the _po_ files. Only the final score per game
is kept: game id, round (7th digit of the id: 004YY00RSG = round R, series S, game G), home, away, points.

Output: data/raw/playoff_games.parquet

  python src/playoffs.py
"""

import io
import tarfile
import urllib.request

import pandas as pd

from config import RAW_DIR, SEASONS, season_label

URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/nbastatsv3_po_{year}.tar.xz"


def season_games(year: int) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(year=year), timeout=120) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
        d = pd.read_csv(tf.extractfile(member), dtype={"gameId": str}, low_memory=False,
                        usecols=["gameId", "teamId", "location", "scoreHome", "scoreAway"])
    d["gameId"] = d["gameId"].str.zfill(10)
    rows = []
    for gid, g in d.groupby("gameId"):
        # home team = whoever has actions with location h, final score = the last score shown
        home = g.loc[g["location"] == "h", "teamId"]
        away = g.loc[g["location"] == "v", "teamId"]
        if home.empty or away.empty:
            continue
        rows.append(dict(season=year, game_id=gid, round=int(gid[7]), home=int(home.mode()[0]),
                         away=int(away.mode()[0]), pts_h=g["scoreHome"].max(), pts_a=g["scoreAway"].max()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for y in SEASONS:
        g = season_games(y)
        print(f"  {season_label(y)}: {len(g)} playoff games")
        out.append(g)
    pd.concat(out, ignore_index=True).to_parquet(RAW_DIR / "playoff_games.parquet", index=False)
