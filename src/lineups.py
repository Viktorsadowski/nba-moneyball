#!/usr/bin/env python3
"""
Step 2: split every game into stints, stretches where the same 10 players are on the floor.
One row per stint: who was on, how long, points each way, and counts for estimating possessions.

Two annoying things about the v3 play-by-play:
  1. a sub row only has the id of the player going OUT, the one coming IN is just a name in the
     description ("SUB: Davis FOR S. O'Neal"), so it gets matched against the game's roster
  2. nobody is listed at the start of a period, so starters are inferred from who shows up
     in the events before being subbed in

Games where either of these can't be resolved get dropped and counted in the summary.

  python src/lineups.py                    # all seasons
  python src/lineups.py --seasons 2010
"""

import argparse
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label


class GameError(Exception):
    pass


# ── names ─────────────────────────────────────────────────────────────────────

def norm(s) -> str:
    # lowercase + strip accents, so "Dončić" in one place matches "Doncic" in another
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


def strip_suffix(s: str) -> str:
    # "porter jr." -> "porter", "otto porter jr." -> "otto porter". subs often leave the suffix out
    parts = s.split(" ")
    while len(parts) > 1 and parts[-1] in SUFFIXES:
        parts.pop()
    return " ".join(parts)


def split_sub_name(name: str) -> tuple[str, str]:
    # "ar. johnson" -> ("ar", "johnson"), "shaw. williams" too, "davis" -> ("", "davis")
    m = re.match(r"^([a-z]{1,5})\.\s*(.+)$", name)
    return (m.group(1), m.group(2)) if m else ("", name)


def build_roster(g: pd.DataFrame, official: pd.DataFrame | None) -> dict[int, dict[int, dict]]:
    """{team_id: {player_id: {last, namei, full, first}}} from the events + game log if we have it."""
    roster: dict[int, dict[int, dict]] = {}

    ppl = g.loc[(g["personId"] > 0) & (g["teamId"] > 0),
                ["teamId", "personId", "playerName", "playerNameI"]].drop_duplicates(["teamId", "personId"])
    for t, p, last, ni in ppl.itertuples(index=False):
        roster.setdefault(int(t), {})[int(p)] = dict(last=norm(last), namei=norm(ni), full="", first="")

    # game log has everyone who played incl. guys with zero events, and full first names
    if official is not None:
        for t, p, full in official[["TEAM_ID", "PLAYER_ID", "PLAYER_NAME"]].itertuples(index=False):
            d = roster.setdefault(int(t), {}).setdefault(int(p), dict(last="", namei="", full="", first=""))
            d["full"] = norm(full)
            d["first"] = d["full"].split(" ")[0]
    return roster


def sub_candidates(team_roster: dict[int, dict], raw_name: str) -> set[int]:
    n = norm(raw_name)
    ini, last = split_sub_name(n)

    # exact first, suffixes stripped only as backup. "Morris" vs "Morris Sr." are two different twins
    for strip in (False, True):
        out = _match(team_roster, n, ini, last, strip)
        # "Marc Morris" / "Mark Morris" (Marcus vs Markieff): first name cut short, no period
        if not out and not ini and " " in last:
            first, rest = last.split(" ", 1)
            out = _match(team_roster, n, first, rest, strip)
        if out:
            return out
    return set()


def _match(team_roster: dict[int, dict], n: str, ini: str, last: str, strip: bool) -> set[int]:
    f = strip_suffix if strip else (lambda x: x)
    last = f(last)
    out = set()
    for pid, d in team_roster.items():
        if d["namei"] and d["namei"] == n:
            out.add(pid)
            continue
        dl, df = f(d["last"]), f(d["full"])
        hit = dl == last or (df and (df == last or df.endswith(" " + last)))
        if not hit:
            continue
        # initials only show up when two guys share a last name, use them to split
        if ini:
            if d["first"]:
                if not d["first"].startswith(ini):
                    continue
            elif d["namei"] and not ini.startswith(d["namei"].split(".")[0]):
                continue
        out.add(pid)
    return out


# ── prep, vectorised over the whole season ────────────────────────────────────

def prep(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["personId"] = df["personId"].fillna(0).astype(np.int64)
    df["teamId"] = df["teamId"].fillna(0).astype(np.int64)
    at = df["actionType"].fillna("")
    st = df["subType"].fillna("")

    # clock is time REMAINING in the period ("PT11M49.00S"), turn it into seconds since tip-off
    mm = df["clock"].str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
    remaining = mm[0] * 60 + mm[1]
    p = df["period"]
    plen = np.where(p <= 4, 720, 300)
    pstart = np.where(p <= 4, 720 * (p - 1), 2880 + 300 * (p - 5))
    df["pstart"] = pstart
    df["pend"] = pstart + plen
    df["t"] = pstart + (plen - remaining)

    df["is_sub"] = at == "Substitution"
    df["in_name"] = df["description"].str.extract(r"SUB:\s*(.+?)\s+FOR\s+", expand=False).str.strip()

    # evidence that a player was on the floor. technicals and ejections can hit guys on the bench
    df["evidence"] = (
        (df["personId"] > 0) & (df["teamId"] > 0) & ~df["is_sub"]
        & ~at.isin(["Ejection", "Timeout", "Instant Replay", "period"])
        & ~((at == "Foul") & st.str.contains("Technical"))
    )

    # scores are only filled on scoring rows (some games put 0-0 on every other row instead of NaN),
    # so blank out 0-0, ffill within game, cummax against glitches, then per-row deltas
    zero = (df["scoreHome"] == 0) & (df["scoreAway"] == 0)
    for side in ("Home", "Away"):
        s = df[f"score{side}"].mask(zero)
        s = s.groupby(df["gameId"]).ffill().fillna(0)
        s = s.groupby(df["gameId"]).cummax()
        df[f"d_{side.lower()}"] = s.groupby(df["gameId"]).diff().fillna(s)

    df["is_fga"] = at.isin(["Made Shot", "Missed Shot", "Heave"])
    df["is_fta"] = at == "Free Throw"
    df["is_tov"] = at == "Turnover"
    df["is_reb"] = at == "Rebound"

    # a miss only makes the next rebound "live" if it's a shot or the last FT of a trip
    ft = st.str.extract(r"(\d) of (\d)").astype(float)
    last_ft = (ft[0] == ft[1]) & ~st.str.contains("Technical|Flagrant")
    missed_shot = df["is_fga"] & (df["shotResult"] == "Missed")
    missed_ft = df["is_fta"] & df["description"].fillna("").str.startswith("MISS") & last_ft
    df["live_miss"] = missed_shot | missed_ft
    return df


# ── one game ──────────────────────────────────────────────────────────────────

def infer_starters(rows: pd.DataFrame, team: int, subs_in: dict, prev: set | None) -> set[int]:
    # walk the period: first time we see a player, was it him subbing IN or doing something/subbing OUT
    first_seen: dict[int, str] = {}
    for r in rows.itertuples():
        if r.teamId != team:
            continue
        if r.is_sub:
            first_seen.setdefault(r.personId, "on")
            cand = subs_in[r.Index]
            if len(cand) == 1:
                first_seen.setdefault(next(iter(cand)), "in")
        elif r.evidence:
            first_seen.setdefault(r.personId, "on")

    starters = {p for p, v in first_seen.items() if v == "on"}

    # someone played the whole period without a single logged event, try last period's five
    if len(starters) < 5 and prev:
        unseen = [p for p in prev if p not in first_seen]
        if len(unseen) == 5 - len(starters):
            starters |= set(unseen)

    if len(starters) != 5:
        raise GameError(f"starters: found {len(starters)}")
    return starters


def build_aliases(df: pd.DataFrame) -> dict[tuple[int, str], set[int]]:
    """{(team, name): ids} from the OUT half of every sub row this season, where the name comes with an id.
    Catches renamed players: the data says Enes Freedom, 2014 descriptions still say "Kanter"."""
    subs = df.loc[df["is_sub"], ["teamId", "personId", "description"]]
    out_name = subs["description"].str.extract(r"\s+FOR\s+(.+)$", expand=False)
    aliases: dict[tuple[int, str], set[int]] = {}
    for t, p, name in zip(subs["teamId"], subs["personId"], out_name):
        if isinstance(name, str):
            aliases.setdefault((int(t), strip_suffix(norm(name))), set()).add(int(p))
    return aliases


def build_game(g: pd.DataFrame, official: pd.DataFrame | None, season: int,
               aliases: dict[tuple[int, str], set[int]]) -> list[dict]:
    gid = g["gameId"].iloc[0]
    home = g.loc[g["location"] == "h", "teamId"]
    away = g.loc[g["location"] == "v", "teamId"]
    if home.empty or away.empty:
        raise GameError("no home/away team")
    home, away = int(home.mode()[0]), int(away.mode()[0])

    roster = build_roster(g, official)
    subs_in: dict[int, set[int]] = {}
    for r in g.loc[g["is_sub"]].itertuples():
        if not isinstance(r.in_name, str):
            raise GameError("sub row without name")
        cand = sub_candidates(roster.get(r.teamId, {}), r.in_name)
        if not cand:
            cand = aliases.get((r.teamId, strip_suffix(norm(r.in_name))), set())
        if not cand:
            raise GameError("sub-in name not found")
        subs_in[r.Index] = cand

    stints: list[dict] = []
    prev = {home: None, away: None}

    for period, rows in g.groupby("period", sort=True):
        lineup = {t: infer_starters(rows, t, subs_in, prev[t]) for t in (home, away)}
        t0, t_end = rows["pstart"].iloc[0], rows["pend"].iloc[0]

        cur = new_stint(t0)
        last_miss = None
        for r in rows.itertuples():
            if r.is_sub:
                cur["pts_h"] += r.d_home  # should be 0 on a sub row, but keeps the totals honest
                cur["pts_a"] += r.d_away
                team = r.teamId
                if r.personId not in lineup[team]:
                    raise GameError("sub-out not on court")
                ins = [c for c in subs_in[r.Index] if c not in lineup[team]]
                if len(ins) != 1:
                    raise GameError("sub-in ambiguous or already on court")
                close(cur, r.t, lineup, home, away, stints, gid, period, season)
                lineup[team].remove(r.personId)
                lineup[team].add(ins[0])
                cur = new_stint(r.t)
                continue

            cur["pts_h"] += r.d_home
            cur["pts_a"] += r.d_away

            # team doing the thing, team-level rows (team rebound, shot clock tov) put the team in personId
            tm = r.teamId if r.teamId > 0 else (r.personId if r.personId in (home, away) else None)
            if tm is None:
                continue
            side = "h" if tm == home else "a"
            if r.is_fga:
                cur[f"fga_{side}"] += 1
            elif r.is_fta:
                cur[f"fta_{side}"] += 1
            elif r.is_tov:
                cur[f"tov_{side}"] += 1
            elif r.is_reb and last_miss is not None:
                if tm == last_miss:
                    cur[f"oreb_{side}"] += 1
                last_miss = None
            if r.live_miss:
                last_miss = tm

        close(cur, t_end, lineup, home, away, stints, gid, period, season)
        prev = {t: set(lineup[t]) for t in (home, away)}

    return stints


def new_stint(t0: float) -> dict:
    d = dict(start=t0, pts_h=0.0, pts_a=0.0)
    for k in ("fga", "fta", "oreb", "tov"):
        d[f"{k}_h"] = d[f"{k}_a"] = 0
    return d


def close(cur, t1, lineup, home, away, stints, gid, period, season) -> None:
    cur["end"] = t1
    cur["dur"] = t1 - cur["start"]
    stats = cur["pts_h"] + cur["pts_a"] + sum(cur[k] for k in cur if k[:3] in ("fga", "fta", "tov", "ore"))
    if cur["dur"] <= 0 and stats == 0:
        return  # back-to-back subs at the same clock, nothing happened in between
    h, a = sorted(lineup[home]), sorted(lineup[away])
    stints.append(dict(
        season=season, game_id=gid, period=period, home_team=home, away_team=away,
        **{f"h{i + 1}": p for i, p in enumerate(h)},
        **{f"a{i + 1}": p for i, p in enumerate(a)},
        **cur,
    ))


# ── one season ────────────────────────────────────────────────────────────────

def build_season(year: int) -> pd.DataFrame:
    pbp = pd.read_parquet(RAW_DIR / f"pbp_{year}.parquet")
    rpath = RAW_DIR / f"roster_{year}.parquet"
    roster = pd.read_parquet(rpath) if rpath.exists() else None
    roster_by_game = dict(tuple(roster.groupby("GAME_ID"))) if roster is not None else {}

    df = prep(pbp)
    aliases = build_aliases(df)
    stints: list[dict] = []
    errors: Counter = Counter()
    for gid, g in df.groupby("gameId", sort=False):
        try:
            stints.extend(build_game(g, roster_by_game.get(gid), year, aliases))
        except GameError as e:
            errors[str(e)] += 1

    out = pd.DataFrame(stints)
    # standard possession estimate per team, stint possessions = average of the two sides
    for s in ("h", "a"):
        out[f"poss_{s}"] = out[f"fga_{s}"] - out[f"oreb_{s}"] + out[f"tov_{s}"] + 0.44 * out[f"fta_{s}"]
    out["poss"] = (out["poss_h"] + out["poss_a"]) / 2

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / f"stints_{year}.parquet", index=False)
    summary(year, df, out, errors, roster)
    return out


def summary(year, df, out, errors, roster) -> None:
    n_games = df["gameId"].nunique()
    n_ok = out["game_id"].nunique()
    print(f"\n{season_label(year)}: {n_ok}/{n_games} games kept ({n_ok / n_games:.1%}), {len(out):,} stints")
    for k, v in errors.most_common():
        print(f"  dropped {v:4d}  {k}")

    # sanity: game length should be 48 min + 5 per OT, and 5 guys each side is enforced above
    mins = out.groupby("game_id")["dur"].sum() / 60
    print(f"  game length min/median/max: {mins.min():.1f} / {mins.median():.1f} / {mins.max():.1f}")
    ppg = (out["pts_h"] + out["pts_a"]).sum() / n_ok / 2
    print(f"  points per team-game: {ppg:.1f}   possessions per team-game: {out['poss'].sum() / n_ok:.1f}")

    # compare our minutes with the official ones from the game log
    if roster is not None:
        cols = [f"h{i}" for i in range(1, 6)] + [f"a{i}" for i in range(1, 6)]
        long = out.melt(id_vars=["game_id", "dur"], value_vars=cols, value_name="PLAYER_ID")
        ours = long.groupby(["game_id", "PLAYER_ID"])["dur"].sum().div(60).rename("ours").reset_index()
        m = ours.merge(roster.rename(columns={"GAME_ID": "game_id"}), on=["game_id", "PLAYER_ID"], how="inner")
        err = (m["ours"] - m["MIN"]).abs()
        print(f"  minutes vs official: MAE {err.mean():.2f}, within 1 min {(err <= 1).mean():.1%}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SEASONS))
    args = ap.parse_args()
    for y in args.seasons:
        build_season(y)
