#!/usr/bin/env python3
"""
Quick check that stats.nba.com answers from this machine before doing the big pull.
Grabs the 2010-11 game list and the play-by-play for one game.
"""

from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

SEASON = "2010-11"

# ── game list ─────────────────────────────────────────────────────────────────
# returns one row per team per game, so ~2460 rows = 1230 games
games = leaguegamefinder.LeagueGameFinder(
    season_nullable=SEASON,
    league_id_nullable="00",           # 00 = NBA, otherwise it also pulls WNBA/G-League
    season_type_nullable="Regular Season",
    timeout=30,
).get_data_frames()[0]

n_games = games["GAME_ID"].nunique()
print(f"{SEASON}: {len(games)} team-rows, {n_games} games (expect 1230)")

# ── play-by-play for one game ─────────────────────────────────────────────────
# v3 is the one still maintained, v2 has gaps in newer seasons
game_id = games["GAME_ID"].iloc[0]
pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30).get_data_frames()[0]

print(f"game {game_id}: {len(pbp)} events")
print("columns:", list(pbp.columns))
print(pbp.head(10).to_string())
