#!/usr/bin/env python3
"""
Side quest: does the MVP vote line up with RAPM?

MVP vote shares 2010-11 to 2025-26 from basketball-reference.com (2024-25 from si.com, points / 1000),
saved by hand in data/mvp_votes.csv.

  python src/mvp.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PROCESSED_DIR, ROOT, season_label
from lineups import norm, strip_suffix

MIN_MINUTES = 1000


def key(name) -> str:
    # "Jimmy Butler III" vs "Jimmy Butler", "Nikola Jokić" vs "Nikola Jokic"
    return strip_suffix(norm(name)).replace(".", "")


if __name__ == "__main__":
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    rapm = rapm[rapm["minutes"] >= MIN_MINUTES].copy()
    rapm["rank"] = rapm.groupby("season")["rapm"].rank(ascending=False, method="min").astype(int)
    rapm["key"] = rapm["name"].map(key)

    votes = pd.read_csv(ROOT / "data" / "mvp_votes.csv")
    votes = votes.rename(columns={"rank": "vote_rank"})
    votes["key"] = votes["player"].map(key)
    m = votes.merge(rapm, on=["season", "key"], how="left")

    missing = m[m["rapm"].isna()]
    if len(missing):
        print("not matched (under 1000 min or name mismatch):", ", ".join(missing["player"] + " " + missing["season"].astype(str)))

    # ── season by season: winner vs RAPM leader ───────────────────────────────
    print(f"\n{'season':8} {'MVP':26} {'rapm':>5} {'rank':>4}   {'RAPM #1':26} {'rapm':>5} {'votes':>6}")
    for s, g in m.groupby("season"):
        w = g.sort_values("share", ascending=False).iloc[0]
        top = rapm[rapm["season"] == s].sort_values("rapm", ascending=False).iloc[0]
        top_share = votes.loc[(votes["season"] == s) & (votes["key"] == top["key"]), "share"]
        ts = f"{top_share.iloc[0]:.3f}" if len(top_share) else "-"
        wr = f"{w['rapm']:5.2f} {int(w['rank']):4d}" if pd.notna(w["rapm"]) else "    -    -"
        print(f"{season_label(s):8} {w['player']:26} {wr}   {top['name']:26} {top['rapm']:5.2f} {ts:>6}")

    # ── overall correlation among vote getters ────────────────────────────────
    ok = m.dropna(subset=["rapm"])
    print(f"\nvote getters with 1000+ min: {len(ok)} player-seasons")
    for c in ("rapm", "o_rapm", "d_rapm"):
        rho, p = spearmanr(ok["share"], ok[c])
        print(f"  spearman(vote share, {c:6}) = {rho:+.2f}  (p={p:.3f})")

    winners = ok.sort_values("share").groupby("season").tail(1)
    print(f"\nMVP winners: median RAPM rank {winners['rank'].median():.0f}, "
          f"top 5 in {np.mean(winners['rank'] <= 5):.0%} of seasons, #1 in {np.mean(winners['rank'] == 1):.0%}")

    # how many of each season's RAPM top 5 got any votes at all
    top5 = rapm[rapm["rank"] <= 5].merge(votes[["season", "key", "share"]], on=["season", "key"], how="left")
    print(f"RAPM top-5 players who got any MVP votes: {top5['share'].notna().mean():.0%}")
