#!/usr/bin/env python3
"""
Playoff stints: same as lineups.py, for the playoff play-by-play (the _po_ files of the same archive).
Only needed for who played how much in the playoffs (playoff_rotation.py), so no official box score
matching, the names come from the play-by-play itself.

Output: data/processed/stints_po_{year}.parquet

  python src/playoff_stints.py
"""

import io
import tarfile
import urllib.request
from collections import Counter

import pandas as pd

from config import PROCESSED_DIR, SEASONS, season_label
from lineups import GameError, build_aliases, build_game, prep

URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/nbastatsv3_po_{year}.tar.xz"


def season(year: int) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(year=year), timeout=120) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
        pbp = pd.read_csv(tf.extractfile(member), dtype={"gameId": str}, low_memory=False)
    pbp["gameId"] = pbp["gameId"].str.zfill(10)
    df = prep(pbp)
    aliases = build_aliases(df)
    stints, errors = [], Counter()
    for gid, g in df.groupby("gameId", sort=False):
        try:
            stints.extend(build_game(g, None, year, aliases))
        except GameError as e:
            errors[str(e)] += 1
    out = pd.DataFrame(stints)
    for s in ("h", "a"):
        out[f"poss_{s}"] = out[f"fga_{s}"] - out[f"oreb_{s}"] + out[f"tov_{s}"] + 0.44 * out[f"fta_{s}"]
    out["poss"] = (out["poss_h"] + out["poss_a"]) / 2
    out.to_parquet(PROCESSED_DIR / f"stints_po_{year}.parquet", index=False)
    n = df["gameId"].nunique()
    print(f"{season_label(year)} playoffs: {out['game_id'].nunique()}/{n} games kept, {len(out):,} stints"
          + (f"  ({', '.join(f'{k}: {v}' for k, v in errors.most_common(2))})" if errors else ""))
    return out


if __name__ == "__main__":
    for y in SEASONS:
        season(y)
