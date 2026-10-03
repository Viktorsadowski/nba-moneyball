#!/usr/bin/env python3
"""
Season simulator: rosters -> team strength -> 10,000 seasons -> playoff / Finals / title odds.
Also the win value curve: what an extra win (or point of net rating) is worth in playoff and Finals odds,
depending on where a team starts.

  rosters       next season's contracts (who plays where now) + our projections (value per 100 incl. the
                current-injury rust, expected minutes). draftees without a projection get their draft
                slot's value and minutes (rookies.py), anyone else without one plays as the average rookie
  minutes       best players first: everyone gets his projected minutes (expected games x mpg) until the
                team's 5 x 48 x 82 minutes are full, the rest is played at replacement level. so a new
                star takes minutes from the worst guys, not from everyone a bit
  strength      same calibration as war.py: net per 100 = a * sum(val * min) / (48 * 82) + 5c, centered so
                the league is 0 (the offset is fixed at the baseline, so trades don't move everyone). then
                stretched by STRETCH: preseason forecasts built this way sat too close to the middle
                (forecast_backtest.py). the lineup-shape effect is measured on real stints, it isn't stretched
  games         margin = net gap per game + home court + noise, home court and noise sd fitted on our last
                5 seasons of games
  uncertainty   every simulated season draws each team's real strength around the projection. sd from how
                far the preseason forecast was off in the backtest (MAE 6.8 wins), minus the part that's just
                game-to-game luck
  schedule      82 games like the real format: 4 vs division, 3 or 4 vs the rest of the conference,
                2 vs the other conference. then play-in (7-10), 4 rounds of best-of-7, 2-2-1-1-1

  python src/season_sim.py                 # league odds + the PHI win curve
  python src/season_sim.py --team PHI
"""

import argparse
import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS, season_label

TEAM_MIN = 5 * 48 * 82
GAMES = 82
N_SIMS = 10000
# both from forecast_backtest.py (the simulator run as of each October 2013-2025), last run:
# how far the preseason forecast is off per team. war.py's check says 6.1, but that one knows who played
MAE_WINS = 6.8
# teams ended up further from average than the rosters said (projected minutes and values are both pulled
# to the middle). this factor on the player-value part of team strength makes forecast and outcome line up
# (teams 1.00x as far from 41 wins as we say, 1.23x without it). 1.0 = off
STRETCH = 1.28
ROOKIE_MIN = 800           # rookies / no history: a bench role until they show otherwise
RNG = np.random.default_rng(26)

DIVISIONS = {
    "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"], "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
    "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"], "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"], "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}
EAST = ["Atlantic", "Central", "Southeast"]
WEST = ["Northwest", "Pacific", "Southwest"]


# ── rosters + strength ────────────────────────────────────────────────────────

def load_rosters() -> tuple[pd.DataFrame, dict]:
    from surplus import contracts_with_ids   # prints a matching line, fine
    as_of = max(SEASONS)
    prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    war = war[war["season"] == as_of].set_index("player_id")
    rapm = pd.read_parquet(PROCESSED_DIR / "rapm.parquet")
    first = rapm.sort_values("season").drop_duplicates("player_id")
    first = first[first["season"] > min(SEASONS)]
    rookie = np.average(first["rapm"], weights=first["minutes"])

    # dead money (bought out / waived, rosters.py) counts for the payroll, not on the court
    con = contracts_with_ids(include_dead=True)
    con = con[con["season"] == as_of + 1]
    prm["payroll"] = con.groupby("team")["salary"].sum().to_dict()
    prm["dead"] = con[con["dead"]].groupby("team")["salary"].sum().to_dict()
    con = con[~con["dead"]].sort_values("salary").drop_duplicates(["player", "team"], keep="last")
    r = con[["player", "team", "salary", "player_id", "guaranteed"]].copy()
    # roster spots: standard contract = at least the rookie minimum (below that it's a two-way), guaranteed =
    # at least half of this season's money guaranteed (camp deals and non-guaranteed minimums can be cut free)
    r["std"] = r["salary"] >= 1.3e6
    r["gtd"] = r["std"] & (r["guaranteed"].fillna(0) >= 0.5 * r["salary"])
    ok = r["player_id"].notna()
    # value for next season in raw units, incl. rust from current injuries (val_adj)
    r["val"] = float(rookie)
    r.loc[ok, "val"] = r.loc[ok, "player_id"].map(war["val"] + war["val_adj"].fillna(0)).fillna(rookie)
    r["min"] = float(ROOKIE_MIN)
    r.loc[ok, "min"] = r.loc[ok, "player_id"].map(war["proj_min"]).fillna(ROOKIE_MIN)
    # share of the season he's expected to be available (injuries): the chance he's there in April
    r["avail"] = (r["player_id"].map(war["proj_min"] / war["healthy_min"])).clip(0, 1).fillna(1.0)
    r = add_rookie_model(r, war, as_of)
    # role from the box score (positions.py), no stats: nba.com's listed position (rosters.py), else wing
    rp = PROCESSED_DIR / "roles.parquet"
    if rp.exists() and (PROCESSED_DIR / "position_effects.json").exists():
        roles = pd.read_parquet(rp)
        roles = roles.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id")["role"]
        r["role"] = r["player_id"].map(roles)
        ro = RAW_DIR / "rosters.parquet"
        if ro.exists():
            listed = pd.read_parquet(ro)
            if "position" in listed:
                # G, G-F, F, F-C, C: the first letter decides
                lp = listed.set_index("player_id")["position"].astype(str).str[0].map({"G": "guard", "F": "wing",
                                                                                        "C": "big"})
                r["role"] = r["role"].fillna(r["player_id"].map(lp))
        r["role"] = r["role"].fillna("wing")
        prm["pos"] = json.loads((PROCESSED_DIR / "position_effects.json").read_text())
    r["name"] = r["player"]
    r["age"] = r["player_id"].map(war["age_next"])
    r["war"] = r["player_id"].map(war["war"])
    prm["rookie"] = rookie
    prm["repl_raw"] = (prm["repl"] - prm["c"]) / prm["a"]
    return r.reset_index(drop=True), prm


def _key(name: str) -> str:
    # name key for contracts vs draft list: no accents, no suffix, letters only
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def add_rookie_model(r: pd.DataFrame, war: pd.DataFrame, as_of: int) -> pd.DataFrame:
    """rookies.py on top of the flat rookie numbers: drafted, no projection (mostly the new draft class)
    gets value + minutes of his draft slot. the seasons 2-4 bump is in value.py already"""
    from config import RAW_DIR
    if not (PROCESSED_DIR / "rookie_model.parquet").exists():
        print("no rookie_model.parquet (python src/rookies.py), rookies stay flat")
        return r
    model = pd.read_parquet(PROCESSED_DIR / "rookie_model.parquet").set_index("pick")
    draft = pd.read_parquet(RAW_DIR / "draft.parquet").sort_values("draft_year").drop_duplicates("player_id", keep="last")
    by_id = draft.set_index("player_id")
    by_name = draft.assign(k=draft["name"].map(_key)).drop_duplicates("k", keep="last").set_index("k")
    r = r.copy()
    pid = r["player_id"]
    pick = pid.map(by_id["pick"])
    year = pid.map(by_id["draft_year"])
    miss = pick.isna()
    k = r["player"].map(_key)
    pick[miss], year[miss] = k[miss].map(by_name["pick"]), k[miss].map(by_name["draft_year"])
    has_proj = pid.isin(war.index) & pid.notna()
    new = pick.notna() & ~has_proj
    r.loc[new, "val"] = pick[new].map(model["val"])
    r.loc[new, "min"] = pick[new].map(model["min"])
    r["pick"], r["draft_year"] = pick, year
    print(f"rookie model: {int(new.sum())} draftees at their draft-slot value")
    return r


def lineup_effect(e_big: float, e_guard: float, pos: dict) -> float:
    """per 100, vs a normally shaped team: how often a roster like this ends up with no big, 3+ bigs, no guard
    or one guard on the floor (positions.py), times what those lineups cost on top of the players' values"""
    def share(k, x, rng):
        a, b, c = pos["polys"][k]
        # quadratic in the average number on the floor, only inside what teams actually did, and flat past
        # the bottom of the curve (more bigs can't make 0-big time go back up)
        x = float(np.clip(x, *rng))
        if a > 0:
            x = min(x, -b / (2 * a))
        return float(np.clip(a * x * x + b * x + c, 0, 1))
    e = pos["effect"]
    tot = (e["0 bigs"] * share("p0b", e_big, pos["Eb_range"]) + e["3+ bigs"] * share("p3b", e_big, pos["Eb_range"])
           + e["0 guards"] * share("p0g", e_guard, pos["Eg_range"]) + e["1 guard"] * share("p1g", e_guard, pos["Eg_range"]))
    return tot - pos["avg"]


def team_net(roster: pd.DataFrame, prm: dict) -> float:
    """net per 100 before centering. best players first, leftovers at replacement. with positions.py's output
    also the lineup-shape effect of who ends up playing those minutes (bigs and guards)"""
    r = roster.sort_values("val", ascending=False)
    left, tot = float(TEAM_MIN), 0.0
    big = guard = 0.0
    roles = r["role"] if "role" in r else pd.Series("wing", index=r.index)
    for v, m, rl in zip(r["val"], r["min"], roles):
        use = min(m, left)
        tot += v * use
        big += use * (rl == "big")
        guard += use * (rl == "guard")
        left -= use
        if left <= 0:
            break
    tot += prm["repl_raw"] * left
    net = STRETCH * (prm["a"] * tot / (TEAM_MIN / 5) + 5 * prm["c"])
    if prm.get("pos"):
        net += lineup_effect(5 * big / TEAM_MIN, 5 * guard / TEAM_MIN, prm["pos"])
    return net


def team_net_po(roster: pd.DataFrame, prm: dict, shares: np.ndarray) -> float:
    """playoff strength, same scale as team_net: players ranked by value get the playoff minute shares by rank
    (playoff_rotation.py), each only as far as he's likely to be healthy, the rest at replacement"""
    r = roster.sort_values("val", ascending=False).head(len(shares))
    v, a, s = r["val"].to_numpy(), r["avail"].to_numpy(), shares[:len(r)]
    tot = np.sum(s * (a * v + (1 - a) * prm["repl_raw"])) + prm["repl_raw"] * (1 - s.sum())
    return STRETCH * (prm["a"] * 5 * tot + 5 * prm["c"])


def playoff_shift(rosters: pd.DataFrame, prm: dict, offset: float = 0.0) -> pd.Series | None:
    """what the shorter playoff rotation adds to (or takes from) each team, per 100. both sides with the
    same method (rank shares, availability), only the shares differ, so it's just the rotation effect.
    None without playoff_shares.parquet"""
    path = PROCESSED_DIR / "playoff_shares.parquet"
    if not path.exists():
        return None
    sh = pd.read_parquet(path)
    po, rg = sh["po_share"].to_numpy(), sh["reg_share"].to_numpy()
    return rosters.groupby("team").apply(lambda g: team_net_po(g, prm, po) - team_net_po(g, prm, rg),
                                         include_groups=False)


def league_nets(rosters: pd.DataFrame, prm: dict, offset: float | None = None) -> tuple[pd.Series, float]:
    raw = rosters.groupby("team").apply(lambda g: team_net(g, prm), include_groups=False)
    if offset is None:
        offset = float(raw.mean())
    return raw - offset, offset


# ── game model ────────────────────────────────────────────────────────────────

def game_model(prm: dict) -> dict:
    """home court + noise sd of a game margin, given the two teams' season net per game. last 5 seasons."""
    rows = []
    for y in list(SEASONS)[-5:]:
        st = pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet",
                             columns=["game_id", "home_team", "away_team", "pts_h", "pts_a"])
        g = st.groupby("game_id").agg(h=("home_team", "first"), a=("away_team", "first"),
                                      ph=("pts_h", "sum"), pa=("pts_a", "sum"))
        g["m"] = g["ph"] - g["pa"]
        per = pd.concat([pd.DataFrame({"t": g["h"], "m": g["m"]}), pd.DataFrame({"t": g["a"], "m": -g["m"]})])
        strength = per.groupby("t")["m"].mean()
        # team strength without home court: take the home edge out afterwards via the intercept
        x = g["h"].map(strength) - g["a"].map(strength)
        b = np.polyfit(x, g["m"], 1)
        rows.append(dict(slope=b[0], home=b[1], sd=float(np.std(g["m"] - np.polyval(b, x)))))
    d = pd.DataFrame(rows).mean()
    pace = prm["pace"][str(max(SEASONS))] * 48       # team possessions per game
    # per 100 -> per game
    d["per_game"] = pace / 100
    return d.to_dict()


def strength_sd(gm: dict, prm: dict) -> float:
    """sd of a team's real strength around our projection, in net per 100."""
    sd_wins = MAE_WINS * np.sqrt(np.pi / 2)                  # MAE -> sd for a normal
    luck = np.sqrt(GAMES * 0.25)                              # binomial part, even-ish teams
    true_wins = np.sqrt(max(sd_wins ** 2 - luck ** 2, 1.0))
    wins_per_point = GAMES * gm["per_game"] / prm["ppw"]      # one point per 100 = this many wins
    return true_wins / wins_per_point


# ── schedule + simulation ─────────────────────────────────────────────────────

def schedule() -> list[tuple[str, str]]:
    games = []
    div_of = {t: d for d, ts in DIVISIONS.items() for t in ts}
    conf = {d: (EAST if d in EAST else WEST) for d in DIVISIONS}
    teams = [t for ts in DIVISIONS.values() for t in ts]
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            da, db = div_of[a], div_of[b]
            if da == db:
                n = 4
            elif db in conf[da]:
                # 6 conference opponents x4, 4 x3: pair gets 4 when the position gap is 0-2 (mod 5)
                ia, ib = DIVISIONS[da].index(a), DIVISIONS[db].index(b)
                n = 4 if (ia - ib) % 5 in (0, 1, 2) else 3
            else:
                n = 2
            for k in range(n):
                games.append((a, b) if (k + i) % 2 == 0 else (b, a))
    return games


def series_win(p_hi_home, p_hi_away, n):
    """P(higher seed wins a best-of-7), 2-2-1-1-1. simulated, vectorised over sims."""
    home = np.array([1, 1, 0, 0, 1, 0, 1], bool)
    w = np.zeros(n, int)
    l = np.zeros(n, int)
    for g in range(7):
        p = np.where(home[g], p_hi_home, p_hi_away)
        win = RNG.random(n) < p
        live = (w < 4) & (l < 4)
        w += win & live
        l += ~win & live
    return w == 4


def simulate(nets: pd.Series, gm: dict, sd: float, n_sims: int = N_SIMS, po_shift: pd.Series | None = None) -> pd.DataFrame:
    """po_shift: playoff strength - regular strength per team (playoff_shift), used from the first round on"""
    from scipy.stats import norm
    teams = list(nets.index)
    ix = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    true_reg = nets.to_numpy()[None, :] + RNG.normal(0, sd, (n_sims, T))   # this season's real strength
    true = true_reg
    per_game = gm["per_game"] * gm["slope"]

    def p_home(h, a):
        # h, a: arrays of team strength (per 100) for the home and away team
        return norm.cdf((per_game * (h - a) + gm["home"]) / gm["sd"])

    sch = schedule()
    H = np.array([ix[h] for h, _ in sch])
    A = np.array([ix[a] for _, a in sch])
    wins = np.zeros((n_sims, T))
    for g in range(len(sch)):
        h, a = H[g], A[g]
        hw = RNG.random(n_sims) < p_home(true[:, h], true[:, a])
        wins[:, h] += hw
        wins[:, a] += ~hw

    # playoffs: shorter rotations (playoff_rotation.py), same luck draw
    true_po = true_reg + (po_shift.reindex(teams).fillna(0).to_numpy()[None, :] if po_shift is not None else 0)
    east = [ix[t] for d in EAST for t in DIVISIONS[d]]
    west = [ix[t] for d in WEST for t in DIVISIONS[d]]
    out = {k: np.zeros((n_sims, T), bool) for k in ("top6", "playoffs", "conf_finals", "finals", "title")}
    champs = {}
    for cname, cidx in (("E", east), ("W", west)):
        true = true_reg                                                  # play-in is still regular season
        cidx = np.array(cidx)
        w = wins[:, cidx] + RNG.random((n_sims, len(cidx))) * 0.01      # random tiebreak
        order = cidx[np.argsort(-w, axis=1)]                             # seeds 1..15 per sim
        rows = np.arange(n_sims)
        out["top6"][rows[:, None], order[:, :6]] = True

        def game(hi, lo):
            # single game, hi at home. returns winner index array
            p = p_home(true[rows, hi], true[rows, lo])
            return np.where(RNG.random(n_sims) < p, hi, lo)

        # play-in: 7v8 -> 7 seed, 9v10, loser(7v8) v winner(9v10) -> 8 seed
        s7, s8, s9, s10 = order[:, 6], order[:, 7], order[:, 8], order[:, 9]
        w78 = game(s7, s8)
        l78 = np.where(w78 == s7, s8, s7)
        w910 = game(s9, s10)
        w_last = game(l78, w910)
        seeds = np.column_stack([order[:, :6], w78, w_last])            # 8 playoff teams, seed order
        out["playoffs"][rows[:, None], seeds] = True
        true = true_po

        def series(hi, lo):
            ph = p_home(true[rows, hi], true[rows, lo])
            pa = 1 - p_home(true[rows, lo], true[rows, hi])
            return np.where(series_win(ph, pa, n_sims), hi, lo)

        # 1v8, 4v5, 2v7, 3v6 -> (1/8 v 4/5), (2/7 v 3/6) -> conf finals
        r1 = [series(seeds[:, a], seeds[:, b]) for a, b in ((0, 7), (3, 4), (1, 6), (2, 5))]
        # better regular-season record gets home court from here on
        def better(x, y):
            wx, wy = wins[rows, x], wins[rows, y]
            return np.where(wx >= wy, x, y), np.where(wx >= wy, y, x)
        a, b = better(r1[0], r1[1])
        sf1 = series(a, b)
        a, b = better(r1[2], r1[3])
        sf2 = series(a, b)
        out["conf_finals"][rows, sf1] = True
        out["conf_finals"][rows, sf2] = True
        a, b = better(sf1, sf2)
        champs[cname] = series(a, b)
        out["finals"][rows, champs[cname]] = True
    # finals: better record has home court
    e_better = wins[rows, champs["E"]] >= wins[rows, champs["W"]]
    a = np.where(e_better, champs["E"], champs["W"])
    b = np.where(e_better, champs["W"], champs["E"])
    ph = p_home(true[rows, a], true[rows, b])
    pa = 1 - p_home(true[rows, b], true[rows, a])
    out["title"][rows, np.where(series_win(ph, pa, n_sims), a, b)] = True

    res = pd.DataFrame({"team": teams, "net": nets.to_numpy(), "wins": wins.mean(0)})
    for k, v in out.items():
        res[k] = v.mean(0)
    res["conf"] = ["E" if i in east else "W" for i in range(T)]
    return res.sort_values("wins", ascending=False).reset_index(drop=True)


def win_curve(nets: pd.Series, team: str, gm: dict, sd: float, shifts=np.arange(-4, 10.5, 1.0),
              po_shift: pd.Series | None = None) -> pd.DataFrame:
    """move one team's strength up / down, everything else fixed: odds per projected win total."""
    rows = []
    for s in shifts:
        n2 = nets.copy()
        n2[team] += s
        r = simulate(n2, gm, sd, n_sims=4000, po_shift=po_shift).set_index("team").loc[team]
        rows.append(dict(shift=s, net=n2[team], wins=r["wins"], playoffs=r["playoffs"], top6=r["top6"],
                         finals=r["finals"], title=r["title"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="PHI")
    args = ap.parse_args()

    rosters, prm = load_rosters()
    nets, offset = league_nets(rosters, prm)
    gm = game_model(prm)
    sd = strength_sd(gm, prm)
    print(f"game model: home court {gm['home']:+.2f} pts, margin sd {gm['sd']:.1f}, slope {gm['slope']:.2f}; "
          f"team strength sd {sd:.2f} per 100")
    po = playoff_shift(rosters, prm, offset)
    if po is not None:
        print("playoff strength - regular season, per 100 (shorter rotations): "
              + ", ".join(f"{t} {v:+.2f}" for t, v in po.sort_values().iloc[[0, 1, 2, -3, -2, -1]].items()))
    res = simulate(nets, gm, sd, po_shift=po)
    show = res.copy()
    for k in ("top6", "playoffs", "conf_finals", "finals", "title"):
        show[k] = (show[k] * 100).round(1)
    print(f"\n{season_label(max(SEASONS) + 1)} odds (%), {N_SIMS:,} seasons")
    for c in ("E", "W"):
        print(show[show["conf"] == c][["team", "net", "wins", "top6", "playoffs", "conf_finals", "finals", "title"]]
              .round(1).to_string(index=False))
    res.to_parquet(PROCESSED_DIR / "season_odds.parquet", index=False)

    cur = win_curve(nets, args.team, gm, sd, po_shift=po)
    print(f"\nwin value curve, {args.team}: odds by projected strength (everything else fixed)")
    print(cur.assign(**{k: (cur[k] * 100).round(1) for k in ("playoffs", "top6", "finals", "title")})
          .round(1).to_string(index=False))
    cur.to_parquet(PROCESSED_DIR / f"win_curve_{args.team}.parquet", index=False)
    r = rosters[rosters["team"] == args.team].sort_values("val", ascending=False)
    print(f"\n{args.team} roster used:")
    print(r[["name", "age", "salary", "val", "min", "war"]].round(2).to_string(index=False))
