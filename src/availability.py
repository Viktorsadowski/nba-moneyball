#!/usr/bin/env python3
"""
Step 6: how many games did each player miss, and how many of those were injuries.

  1. injury log rows -> player ids (name match against the game logs, team used to split duplicates)
  2. "Relinquished" = went out, next "Acquired" for the same guy = came back. That's one injury spell.
     No return logged -> spell runs to the end of the season.
  3. games missed to injury = his team's games inside a spell that he didn't play in

From 2021-22 on the official NBA injury reports are used instead (injury_reports.py --parse):
  listed with an "Injury/Illness" reason for that day's game + not in that game's log = injury game missed.
  inj_spells there = number of different injury reasons he was listed with.
2020-21 has neither source.

Plus plain availability from the game logs: games played vs his team's games that season.

Output: data/processed/availability.parquet, one row per player-season
  gp, team_gp, avail (= gp / team_gp), inj_games, inj_spells, inj_note (longest spell's note)

  python src/availability.py
"""

import argparse
import re

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from lineups import norm, strip_suffix

NBA_IDS = None  # filled lazily from nba_api static data (no web call)


def key(name: str) -> str:
    # "Nene Hilario (a)" -> "nene hilario", "Otto Porter Jr." -> "otto porter"
    name = re.sub(r"\(.*?\)", "", str(name))
    return strip_suffix(norm(name)).replace(".", "").strip()


def team_id(nickname: str, year: int) -> int | None:
    global NBA_IDS
    if NBA_IDS is None:
        from nba_api.stats.static import teams
        NBA_IDS = {t["nickname"].lower(): t["id"] for t in teams.get_teams()}
        NBA_IDS["bobcats"] = NBA_IDS["hornets"]           # charlotte
        NBA_IDS["sixers"] = NBA_IDS["76ers"]
        NBA_IDS["blazers"] = NBA_IDS["trail blazers"]
    n = str(nickname).strip().lower()
    # "Hornets" was New Orleans until 2012-13, Charlotte from 2014-15
    if n == "hornets" and year <= 2012:
        return NBA_IDS["pelicans"]
    return NBA_IDS.get(n)


# ── names -> ids ──────────────────────────────────────────────────────────────

def build_name_index(rosters: dict[int, pd.DataFrame]) -> dict:
    """{key: {(season, team_id, pid), ...}} over all seasons we have game logs for."""
    idx: dict[str, set] = {}
    for y, r in rosters.items():
        for pid, tid, name in r[["PLAYER_ID", "TEAM_ID", "PLAYER_NAME"]].drop_duplicates().itertuples(index=False):
            idx.setdefault(key(name), set()).add((y, int(tid), int(pid)))
    return idx


def resolve(name: str, year: int, tid: int | None, idx: dict) -> int | None:
    # names can come as "Nene / Nene Hilario", try every alias
    for alias in str(name).split("/"):
        hits = idx.get(key(alias))
        if not hits:
            continue
        pids = {p for _, _, p in hits}
        if len(pids) == 1:
            return next(iter(pids))
        # same name, different players: same season + same team first, then same season
        for pick in (lambda h: h[0] == year and h[1] == tid, lambda h: h[0] == year):
            cand = {p for h in hits if pick(h) for p in [h[2]]}
            if len(cand) == 1:
                return next(iter(cand))
    return None


# ── spells ────────────────────────────────────────────────────────────────────

# not injuries: rest, suspensions, personal stuff, G League trips. and a bare "placed on IL" is
# just the inactive list before 2017, often healthy scratches, only count it with a reason attached
NON_INJURY = re.compile(r"\brest\b|resting|personal|suspen|coach|g league|g-league|d-league|not with team|"
                        r"family|birth|bereavement|two-way|conditioning|trade|contract", re.I)


def is_injury(note: str) -> bool:
    n = str(note).strip().lower()
    if not n or n in ("placed on il", "placed on inactive list", "activated from il", "returned to lineup"):
        return False
    return not NON_INJURY.search(n)


def note_rank(note: str) -> tuple:
    # which note names a merged absence: serious words first (tear, fracture, surgery...),
    # then the most specific type (injury_types.TYPES order, Achilles/ACL first)
    from injury import SERIOUS
    from injury_types import TYPES, injury_type
    order = [n for n, _ in TYPES] + ["Other"]
    return (0 if SERIOUS.search(str(note)) else 1, order.index(injury_type(note)))


def build_spells(inj: pd.DataFrame, year: int, idx: dict) -> tuple[pd.DataFrame, int]:
    ev = []
    for r in inj.itertuples(index=False):
        tid = team_id(r.Team, year)
        if r.Relinquished and is_injury(r.Notes):
            ev.append((r.Date, "out", r.Relinquished, tid, r.Notes))
        if r.Acquired:
            ev.append((r.Date, "back", r.Acquired, tid, r.Notes))
    ev = pd.DataFrame(ev, columns=["date", "kind", "name", "team_id", "note"])
    ev["pid"] = [resolve(n, year, t, idx) for n, t in zip(ev["name"], ev["team_id"])]
    unmatched = int(ev["pid"].isna().sum())
    ev = ev.dropna(subset=["pid", "date"]).sort_values(["pid", "date", "kind"], ascending=[True, True, False])

    # every "out" row is its own spell, up to the next logged return (or season end).
    # used to fold later outs into an open one ("DNP" in Nov without a return + "torn ACL" in Jan
    # = one spell), then games_missed cut it at his next game and the ACL was gone.
    # games_missed merges the ones that end up covering the same absence
    season_end = pd.Timestamp(f"{year + 1}-07-31")
    spells = []
    for pid, g in ev.groupby("pid"):
        backs = g.loc[g["kind"] == "back", "date"].to_numpy()
        for r in g[g["kind"] == "out"].itertuples(index=False):
            later = backs[backs > np.datetime64(r.date)]
            end = pd.Timestamp(later[0]) if len(later) else season_end
            spells.append((int(pid), r.team_id, r.date, end, r.note))
    return pd.DataFrame(spells, columns=["pid", "team_id", "start", "end", "note"]), unmatched


def games_missed(spells: pd.DataFrame, roster: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    played = set(zip(roster["GAME_ID"], roster["PLAYER_ID"]))
    # his team if the log didn't give one: the team he played most games for
    main_team = roster.groupby(["PLAYER_ID", "TEAM_ID"]).size().reset_index().sort_values(0) \
        .drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["TEAM_ID"]
    by_team = {t: g.sort_values("GAME_DATE") for t, g in sched.groupby("TEAM_ID")}

    # dates he actually played, a spell can't run past his next game (the log often skips the "returned")
    dates = roster.merge(sched[["GAME_ID", "TEAM_ID", "GAME_DATE"]], on=["GAME_ID", "TEAM_ID"])
    played_on = {p: np.sort(g["GAME_DATE"].to_numpy()) for p, g in dates.groupby("PLAYER_ID")}

    ends = []
    for s in spells.itertuples(index=False):
        end = s.end
        pd_ = played_on.get(s.pid)
        if pd_ is not None:
            later = pd_[pd_ > np.datetime64(s.start)]
            if len(later) and later[0] < np.datetime64(end):
                end = pd.Timestamp(later[0])
        ends.append(end)
    spells = spells.copy()
    spells["end"] = ends

    # rows that end at the same point are one absence (DTD -> out -> out for season).
    # first start, most serious note
    spells["rank"] = [note_rank(n) for n in spells["note"]]
    note = spells.sort_values("rank").drop_duplicates(["pid", "end"]).set_index(["pid", "end"])["note"]
    spells = spells.sort_values("start").drop_duplicates(["pid", "end"], keep="first").drop(columns="rank")
    spells["note"] = [note[(p, e)] for p, e in zip(spells["pid"], spells["end"])]

    out = []
    for s in spells.itertuples(index=False):
        t = s.team_id if pd.notna(s.team_id) else main_team.get(s.pid)
        g = by_team.get(t)
        if g is None:
            out.append(0)
            continue
        inside = g[(g["GAME_DATE"] >= s.start) & (g["GAME_DATE"] < s.end)]
        out.append(sum((gid, s.pid) not in played for gid in inside["GAME_ID"]))
    spells["games"] = out
    return spells.reset_index(drop=True)


# ── official injury reports (2021-22 on) ──────────────────────────────────────

def compact(name: str) -> str:
    # report names come without spaces ("HardawayJr.,Tim"), so compare everything squashed
    return re.sub(r"[^a-z0-9]", "", norm(name))


def unsuffix(c: str) -> str:
    return re.sub(r"(jr|sr|ii|iii|iv)$", "", c)


def report_injuries(year: int, roster: pd.DataFrame, sched: pd.DataFrame, rosters: dict):
    rep = pd.read_parquet(RAW_DIR / f"injury_report_rows_{year}.parquet")
    rep = rep[rep["reason"].str.startswith("Injury/Illness")].dropna(subset=["date"])
    # a game can show up on two reports (tonight's + tomorrow's early list), keep the latest one
    rep = rep.sort_values("report").drop_duplicates(["date", "team", "player"], keep="last")

    from nba_api.stats.static import teams
    nick = {compact(t["nickname"]): t["id"] for t in teams.get_teams()}
    rep["team_id"] = [next((i for n, i in nick.items() if compact(t).endswith(n)), None) for t in rep["team"]]

    # "Last,First" -> "firstlast", matched on his team that season first, then anywhere
    def flip(n):
        last, _, first = n.partition(",")
        return compact(first + last)
    rep["key"] = rep["player"].map(flip)
    team_idx, any_idx = {}, {}
    for pid, tid, name in roster[["PLAYER_ID", "TEAM_ID", "PLAYER_NAME"]].drop_duplicates().itertuples(index=False):
        for k in {compact(name), unsuffix(compact(name))}:
            team_idx.setdefault((tid, k), set()).add(pid)
    for r in rosters.values():
        for pid, name in r[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates().itertuples(index=False):
            for k in {compact(name), unsuffix(compact(name))}:
                any_idx.setdefault(k, set()).add(pid)

    def pid_of(k, tid):
        for kk in (k, unsuffix(k)):
            for hit in (team_idx.get((tid, kk)), any_idx.get(kk)):
                if hit and len(hit) == 1:
                    return next(iter(hit))
        return None
    rep["pid"] = [pid_of(k, t) for k, t in zip(rep["key"], rep["team_id"])]
    unmatched = int(rep["pid"].isna().sum())

    # the team's game that day, missed if he's not in that game's log
    g = sched[["GAME_ID", "TEAM_ID", "GAME_DATE"]].rename(columns={"TEAM_ID": "team_id", "GAME_DATE": "date"})
    m = rep.dropna(subset=["pid"]).merge(g, on=["team_id", "date"], how="inner")
    played = set(zip(roster["GAME_ID"], roster["PLAYER_ID"]))
    # np.array, a plain empty list would be read as "select no columns"
    m = m[np.array([(gid, p) not in played for gid, p in zip(m["GAME_ID"], m["pid"])], dtype=bool)]
    m["pid"] = m["pid"].astype(int)

    agg = m.groupby("pid").agg(inj_games=("GAME_ID", "nunique"), inj_spells=("reason", "nunique"))
    agg["inj_note"] = m.groupby("pid")["reason"].agg(lambda x: x.value_counts().index[0])
    msg = f"{len(m)} injury games from reports, {unmatched}/{len(rep)} report rows without a player match"
    return agg, msg


# ── one season ────────────────────────────────────────────────────────────────

def season_availability(year: int, rosters: dict, idx: dict) -> pd.DataFrame:
    roster, sched = rosters[year], pd.read_parquet(RAW_DIR / f"team_games_{year}.parquet")

    # plain availability: games played vs team games, summed over every team he played for
    team_gp = sched.groupby("TEAM_ID")["GAME_ID"].nunique()
    gp = roster.groupby("PLAYER_ID")["GAME_ID"].nunique().rename("gp")
    season_len = int(team_gp.median())
    base = gp.to_frame()
    base["team_gp"] = season_len
    base["avail"] = base["gp"] / season_len

    ipath = RAW_DIR / f"injuries_{year}.parquet"
    rpath = RAW_DIR / f"injury_report_rows_{year}.parquet"
    if rpath.exists():
        agg, msg = report_injuries(year, roster, sched, rosters)
    elif ipath.exists():
        inj = pd.read_parquet(ipath).fillna("")
        spells, unmatched = build_spells(inj, year, idx)
        spells = games_missed(spells, roster, sched)
        spells = spells[spells["games"] > 0]
        agg = spells.groupby("pid").agg(inj_games=("games", "sum"), inj_spells=("games", "size"))
        agg["inj_note"] = spells.sort_values("games").drop_duplicates("pid", keep="last").set_index("pid")["note"]
        n_ev = int((inj["Relinquished"] != "").sum() + (inj["Acquired"] != "").sum())
        msg = (f"{len(spells)} injury spells, {int(spells['games'].sum())} games missed, "
               f"{unmatched}/{n_ev} log entries without a player match")
    else:
        print(f"  {season_label(year)}: no injury log, availability only")
        base[["inj_games", "inj_spells"]] = np.nan
        base["inj_note"] = ""
        return base.reset_index().rename(columns={"PLAYER_ID": "player_id"}).assign(season=year)

    # guys who missed the whole season have no game log row, keep them anyway
    res = base.join(agg, how="outer")
    res["gp"] = res["gp"].fillna(0)
    res["team_gp"] = season_len
    res["avail"] = res["gp"] / season_len
    res[["inj_games", "inj_spells"]] = res[["inj_games", "inj_spells"]].fillna(0)
    # can't miss more games than the ones he didn't play (traded/waived while "out" otherwise overcounts)
    res["inj_games"] = res["inj_games"].clip(upper=season_len - res["gp"]).clip(lower=0)
    res["inj_note"] = res["inj_note"].fillna("")

    print(f"  {season_label(year)}: {msg}")
    return res.reset_index().rename(columns={"index": "player_id", "PLAYER_ID": "player_id"}).assign(season=year)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    args = ap.parse_args()

    rosters = {y: pd.read_parquet(RAW_DIR / f"roster_{y}.parquet") for y in SEASONS
               if (RAW_DIR / f"roster_{y}.parquet").exists()}
    idx = build_name_index(rosters)
    res = pd.concat([season_availability(y, rosters, idx) for y in args.seasons], ignore_index=True)
    res.to_parquet(PROCESSED_DIR / "availability.parquet", index=False)

    names = pd.concat(rosters.values()).groupby("PLAYER_ID")["PLAYER_NAME"].last()
    last = res[res["season"] == res["season"].max()].sort_values("inj_games", ascending=False)
    last = last.assign(name=last["player_id"].map(names))
    print(f"\nmost games lost to injury, {season_label(int(last['season'].iloc[0]))}:")
    print(last[["name", "gp", "inj_games", "inj_spells", "inj_note"]].head(10).to_string(index=False))
