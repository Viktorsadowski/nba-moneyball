#!/usr/bin/env python3
"""
Trade search for one team (PHI): the best roster we can trade our way to, under a payroll limit.

  budgets     under the tax line, under the 1st apron, under the 2nd apron (2026-27 numbers). the owner
              picks how far past the tax he's willing to go, the GM finds the best roster under it
  trades      2-team trades, up to 3 in a row (different partners or the same one). PHI sends 1-3 players
              + picks, gets 0-2 players back. 2026 draftees stay out (nobody trades them now), LeBron stays
  CBA         salary matching by apron (2023 CBA, amounts indexed to the 2026-27 cap):
                over the cap, below the 1st apron after the trade: out <= $8.8M 200% + $0.3M,
                $8.8-35.2M +$9.1M, more 125% + $0.3M
                above the 1st apron after the trade: incoming <= outgoing
                above the 2nd apron after the trade: no aggregating (1 outgoing player)
                under the cap after the trade: anything goes (cap room)
              a non-guaranteed salary only counts as outgoing for what's guaranteed (camp deals = $0 for the
              team that sends them), the team that gets him takes on all of it
              PHI can also take one player into each trade exception ($4.2M, $2.3M) without sending salary.
              both teams have to pass
  fair        the partner prices the trade like the market does (market_value.py: market price of each
              contract minus the salary, whole contract), picks at what the league pays for one (trade_value.py
              from real trades, else market_value.py). he has to come out even or better, and a contender (50%+
              playoff odds) can't get weaker this season. PHI adds picks until he's even, the 2nds first.
              --price surplus = the old way, the partner uses our own model
  strength    PHI net per 100 with season_sim.team_net (best players first), partners too. odds from the win
              curve during the search, the full simulator for the finalists (so taking a player from BOS
              also makes BOS worse)
  roster      at most 15 guaranteed contracts after a trade (a team that goes over waives its cheapest guaranteed
              guys, their money stays and the partner's loss counts in the fairness check), at least 14
              standard contracts for PHI (minimum signings, on the payroll)
  kickers     trade bonuses (data/trade_kickers.csv): the team that gets the player carries this season's part
              on the cap (salary matching and payroll), the team that sends him pays it in cash (the partner's
              fairness, our cost). worst case, as if every kicker still applies and nobody waives it
  positions   team strength includes the lineup-shape effect (positions.py): a roster that ends up with no big
              or too few guards on the floor pays for it
  cost        what PHI gives away in value (surplus + picks - surplus back) + the extra luxury tax this season,
              by our model (what we think we lose) and by the market (what it would cost to buy back)

  python src/trade_search.py                  # partner prices like the market
  python src/trade_search.py --price surplus  # partner prices like our model
"""

import itertools
import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label

TEAM = "PHI"
KEEP = {"LeBron James"}                # the whole point is his last ring
CAP, TAX, APRON1, APRON2 = 164.961e6, 200.428e6, 209.015e6, 221.686e6
IDX = CAP / 136.021e6                  # CBA trade amounts are 2023-24 numbers, they move with the cap
T1, T2, PLUS, CUSH = 7.25e6 * IDX, 29e6 * IDX, 7.5e6 * IDX, 0.25e6 * IDX
TPES = [4_221_360.0, 2_296_274.0]      # McCain and Gordon trade exceptions (sixershoops, capmath)
BRACKET = 5e6 * IDX                    # tax brackets, $5M in 2023-24
RATES = [1.0, 1.25, 3.5, 4.75]         # non-repeater from 2025-26 on (1.5 / 1.75 / 2.5 / 3.25 before), +0.5 per
                                       # bracket after the 4th
SECONDS = 8                            # 2nd-rounders PHI can trade (PhillyVoice, Aug 1)
FIRSTS = ["2033 1st"]                  # own 2027/29/30/32 locked (Stepien), 2028 own + LAC tied up with BKN/BOS
BUDGETS = {"tax": TAX, "apron1": APRON1, "apron2": APRON2}
BEAM, TOP, STEPS = 8, 250, 3
CONTENDERS: set = set()               # filled in main: 50%+ playoff odds in season_sim
# how the partner prices a trade: "mkt" = the market's price (market_value.py), "surplus" = our own model
PRICE = dict(col="mkt", first=None, second=None, source=None)
MAX_OUT, MAX_IN = 3, 2
MAX_GTD, MIN_STD = 15, 14             # roster: at most 15 guaranteed contracts, at least 14 standard ones
MIN_SAL = 2_449_421.0                 # filling a spot = a veteran minimum (what it counts on the cap)


# ── money ─────────────────────────────────────────────────────────────────────

def tax_bill(payroll: float) -> float:
    over, bill, i = max(0.0, payroll - TAX), 0.0, 0
    while over > 0:
        rate = RATES[i] if i < len(RATES) else RATES[-1] + 0.5 * (i - len(RATES) + 1)
        bill += min(over, BRACKET) * rate
        over -= BRACKET
        i += 1
    return bill


def max_incoming(out: float, post: float) -> float:
    # over the cap: how much salary can come back for `out` going out
    if post > APRON1:
        return out
    if out <= T1:
        return 2 * out + CUSH
    if out <= T2:
        return out + PLUS
    return 1.25 * out + CUSH


def legal(before: float, out: float, inc: float, n_out: int, out_match: float = None) -> bool:
    # out = what leaves the payroll, out_match = what of it counts for matching (only the guaranteed part)
    post = before - out + inc
    if post <= CAP:
        return True
    if post > APRON2 and n_out > 1:
        return False
    return inc <= max_incoming(out if out_match is None else out_match, post) + 1.0


# ── data ──────────────────────────────────────────────────────────────────────

def load():
    import season_sim as S
    from surplus import dollars_per_war
    from war import salaries_with_ids
    rosters, prm = S.load_rosters()
    nets, offset = S.league_nets(rosters, prm)
    sur = pd.read_parquet(PROCESSED_DIR / "surplus.parquet").set_index("player_id")["surplus"]
    rosters["surplus"] = rosters["player_id"].map(sur).fillna(0.0)
    last = max(SEASONS)
    rosters["rookie26"] = rosters["draft_year"] == last + 1
    # $ per win next season, like surplus.py (last season's price grown with the payroll)
    war = pd.read_parquet(PROCESSED_DIR / "war.parquet")
    sal = salaries_with_ids()
    dpw = dollars_per_war(war, sal)
    pay = sal.groupby("season")["salary"].sum()
    g = (pay[last] / pay[last - 10]) ** 0.1 - 1
    dpw_next = float(dpw[last] * (1 + g))
    rm = pd.read_parquet(PROCESSED_DIR / "rookie_model.parquet").set_index("pick")["surplus_war_4y"]
    # far-away 1st: no idea where it lands, average over picks 1-30. a 2nd: average over 31-60
    picks = {"first": float(rm.loc[1:30].mean() * dpw_next), "second": float(rm.loc[31:60].mean() * dpw_next)}
    curve = pd.read_parquet(PROCESSED_DIR / f"win_curve_{TEAM}.parquet")
    # the other side of the table: the market's price (market_value.py), picks at what the league pays for them
    mvt = pd.read_parquet(PROCESSED_DIR / "market_value.parquet").set_index("player_id")
    rosters["mkt"] = rosters["player_id"].map(mvt["market_surplus"]).fillna(0.0)
    # what waiving a player costs: the partner loses his market price, we lose what he's worth to us
    rosters["mkt_price"] = rosters["player_id"].map(mvt["market"]).fillna(rosters["salary"])
    worth = pd.read_parquet(PROCESSED_DIR / "surplus.parquet").set_index("player_id")["worth"]
    rosters["worth"] = rosters["player_id"].map(worth).fillna(0.0)
    rosters = rosters.join(trade_kickers(rosters))
    # salary that counts as outgoing in a trade: only what's guaranteed this season. `guaranteed` is the total
    # left on the deal, blank = nothing. traded at game 1, so nothing earned yet (in season it grows by the day)
    rosters["out_sal"] = np.minimum(rosters["salary"], rosters["guaranteed"].fillna(0.0))
    tv = PROCESSED_DIR / "trade_value.json"
    if tv.exists() and PRICE.get("picks") != "realized":
        # what real trades say a pick costs (trade_value.py)
        t = json.loads(tv.read_text())
        PRICE.update(first=t["first"], second=t["second"], source="real trades (trade_value.py)")
    else:
        pk = pd.read_parquet(PROCESSED_DIR / "pick_market_value.parquet").set_index("pick")["mkt_4y"]
        PRICE.update(first=float(pk.loc[1:30].mean()), second=float(pk.loc[31:60].mean()),
                     source="market value of what picks turned into (market_value.py)")
    if PRICE["col"] == "surplus":
        PRICE.update(first=picks["first"], second=picks["second"], source="our model")
    return S, rosters, prm, nets, offset, picks, curve, dpw_next


def trade_kickers(rosters: pd.DataFrame) -> pd.DataFrame:
    """trade bonuses (data/trade_kickers.csv, ShamSports list for 2026-27): worst case if he's traded now.
    bonus = pct of what's left of the contract (or the flat amount if that's smaller), spread over the seasons
    like the salary, each season only up to his max (25 / 30 / 35% of the cap by years in the league, the cap
    growing the most it can, 10% a year). the team that trades him pays it in cash, the team that gets him
    carries it on the cap. returns kick_now (this season's cap hit on top) and kick_cash per roster row"""
    from config import ROOT
    from surplus import contracts_with_ids
    path = ROOT / "data" / "trade_kickers.csv"
    out = pd.DataFrame({"kick_now": 0.0, "kick_cash": 0.0}, index=rosters.index)
    if not path.exists():
        return out
    k = pd.read_csv(path)
    con = contracts_with_ids()
    draft = pd.read_parquet(RAW_DIR / "draft.parquet").drop_duplicates("player_id").set_index("player_id")["draft_year"]
    nxt = max(SEASONS) + 1
    from war import compact
    rk, ck = rosters["player"].map(compact), con["player"].map(compact)        # accents, dots: "Jokic" = "Jokić"
    for r in k.itertuples():
        rows = rosters.index[(rk == compact(r.player)) & (rosters["team"] == r.team)]
        c = con[(ck == compact(r.player)) & (con["team"] == r.team)].groupby("season")["salary"].max()
        if rows.empty or c.empty:
            continue
        pid = rosters.loc[rows[0], "player_id"]
        dy = draft.get(pid, nxt - 12) if pd.notna(pid) else nxt - 12       # not in our draft list = a veteran
        service = nxt - dy + 1
        mx = 0.25 if service <= 6 else 0.30 if service <= 9 else 0.35
        cap = pd.Series({y: CAP * 1.10 ** (y - nxt) for y in c.index})
        bonus = r.pct * c.sum() if pd.isna(r.flat) else min(r.pct * c.sum(), r.flat)
        got = np.minimum(bonus * c / c.sum(), (cap * mx - c).clip(lower=0))
        out.loc[rows, "kick_now"] = float(got.get(nxt, 0.0))
        out.loc[rows, "kick_cash"] = float(got.sum())
    return out


def odds_from_curve(curve: pd.DataFrame, net: float) -> dict:
    return {k: float(np.interp(net, curve["net"], curve[k])) for k in ("wins", "playoffs", "finals", "title")}


# ── one trade ─────────────────────────────────────────────────────────────────

def pick_bundle(need: float, seconds_left: int, firsts_left: list, picks: dict):
    """cheapest picks that cover `need` $ for the partner: 2nds first, a 1st only if the 2nds can't do it"""
    if need <= 0:
        return 0.0, 0, []
    n2 = int(np.ceil(need / picks["second"]))
    if n2 <= seconds_left:
        return n2 * picks["second"], n2, []
    for k in range(1, len(firsts_left) + 1):
        rest = need - k * picks["first"]
        n2 = max(0, int(np.ceil(rest / picks["second"])))
        if n2 <= seconds_left:
            return k * picks["first"] + n2 * picks["second"], n2, firsts_left[:k]
    return None


def max_incoming_v(out: np.ndarray, post: np.ndarray) -> np.ndarray:
    m = np.where(out <= T1, 2 * out + CUSH, np.where(out <= T2, out + PLUS, 1.25 * out + CUSH))
    return np.where(post > APRON1, out, m)


def legal_v(before: float, out: np.ndarray, inc: np.ndarray, n_out: np.ndarray, out_match: np.ndarray) -> np.ndarray:
    # out leaves the payroll, out_match is the part of it that counts for matching (guaranteed money only)
    post = before - out + inc
    ok = inc <= max_incoming_v(out_match, post) + 1.0
    ok &= ~((post > APRON2) & (n_out > 1))
    return ok | (post <= CAP)


def candidates(state: dict, picks: dict, budget: float) -> list:
    """every legal, fair trade from this state (numpy over all out x in combinations per partner)"""
    me = state["roster"]
    # a player we just got can't be flipped on (CBA: 2 months before he can be aggregated, and nobody trades
    # for a guy to pass him on the same week)
    out_pool = me[~me["name"].isin(KEEP | {"minimum signing"}) & ~me["rookie26"] & ~me.index.isin(state["moved"])]
    outs = [()] + [c for k in range(1, MAX_OUT + 1) for c in itertools.combinations(out_pool.index, k)]
    o_sal = np.array([me.loc[list(c), "salary"].sum() for c in outs], float)
    o_match = np.array([me.loc[list(c), "out_sal"].sum() for c in outs], float)
    # what the partner takes on: the salary + this season's part of any trade bonus. and what we pay in cash
    o_recv = o_sal + np.array([me.loc[list(c), "kick_now"].sum() for c in outs], float)
    o_cash = np.array([me.loc[list(c), "kick_cash"].sum() for c in outs], float)
    o_sur = np.array([me.loc[list(c), PRICE["col"]].sum() for c in outs], float)
    o_war = np.array([me.loc[list(c), "war"].fillna(0).sum() for c in outs], float)
    o_n = np.array([len(c) for c in outs])
    o_gtd = np.array([me.loc[list(c), "gtd"].sum() for c in outs])
    o_std = np.array([me.loc[list(c), "std"].sum() for c in outs])
    my_pay = state["payroll"][TEAM]
    # our roster count: over 15 guaranteed = waive the least valuable guaranteed guys (their money stays),
    # under 14 standard = sign minimum guys
    my_gtd, my_std = int(me["gtd"].sum()), int(me["std"].sum())
    my_waive = np.r_[0.0, np.cumsum(np.sort(me.loc[me["gtd"], "worth"].to_numpy()))]
    # the most the partner can be short and still get covered by our picks
    pick_max = state["seconds"] * PRICE["second"] + len(state["firsts"]) * PRICE["first"]
    res = []
    for team, their in state["others"].groupby("team"):
        pool = their[~their["rookie26"] & ~their.index.isin(state["moved"])]
        # only guys who'd play for us as a real upgrade are worth getting, the 2nd man can be anyone (salary)
        good = pool[pool["val"] > 0.5].index
        ins = [()] + [(i,) for i in good] + [c for c in itertools.combinations(pool.index, 2)
                                             if c[0] in good or c[1] in good]
        i_sal = np.array([pool.loc[list(c), "salary"].sum() for c in ins], float)
        i_match = np.array([pool.loc[list(c), "out_sal"].sum() for c in ins], float)
        i_recv = i_sal + np.array([pool.loc[list(c), "kick_now"].sum() for c in ins], float)
        i_cash = np.array([pool.loc[list(c), "kick_cash"].sum() for c in ins], float)
        i_sur = np.array([pool.loc[list(c), PRICE["col"]].sum() for c in ins], float)
        i_war = np.array([pool.loc[list(c), "war"].fillna(0).sum() for c in ins], float)
        i_n = np.array([len(c) for c in ins])
        i_gtd = np.array([pool.loc[list(c), "gtd"].sum() for c in ins])
        i_std = np.array([pool.loc[list(c), "std"].sum() for c in ins])
        t_pay = state["payroll"][team]
        # partner over 15 guaranteed because of this trade: he waives his cheapest guaranteed guys, and that's
        # a cost the trade has to cover (a team already over 15 in camp isn't charged for that)
        t_gtd = int(their["gtd"].sum())
        t_waive = np.r_[0.0, np.cumsum(np.sort(their.loc[their["gtd"], "mkt_price"].to_numpy()))]
        O, I = np.meshgrid(np.arange(len(outs)), np.arange(len(ins)))     # rows = ins, cols = outs
        OS, IS = o_sal[O], i_sal[I]
        OR, IR = o_recv[O], i_recv[I]
        fill = np.maximum(0, MIN_STD - (my_std - o_std[O] + i_std[I]))
        post = my_pay - OS + IR + fill * MIN_SAL
        ok = post <= budget
        ok &= ~((o_n[O] == 0) & (i_n[I] == 0))
        # nothing going out = a trade exception, one player that fits
        tp = np.zeros_like(IS)
        if state["tpes"]:
            fit = np.array(sorted(state["tpes"]))
            j = np.searchsorted(fit, IS - 1.0)
            tp = np.where(j < len(fit), fit[np.minimum(j, len(fit) - 1)], np.inf)
        tpe_ok = (o_n[O] == 0) & (i_n[I] == 1) & np.isfinite(tp) & (len(state["tpes"]) > 0)
        ok &= np.where(o_n[O] == 0, tpe_ok, legal_v(my_pay, OS, IR, o_n[O], o_match[O]))
        ok &= legal_v(t_pay, IS, OR, i_n[I], i_match[I])
        ex_t = np.maximum(0, t_gtd - i_gtd[I] + o_gtd[O] - max(MAX_GTD, t_gtd))
        ex_me = np.maximum(0, my_gtd - o_gtd[O] + i_gtd[I] - max(MAX_GTD, my_gtd))
        cost_t = t_waive[np.minimum(ex_t, len(t_waive) - 1)]
        # the partner also pays the trade bonus of anyone he sends us, in cash
        need = i_sur[I] - o_sur[O] + (cost_t if PRICE["col"] == "mkt" else 0.0) + i_cash[I]
        ok &= need <= pick_max
        for r, c in zip(*np.nonzero(ok)):
            pb = pick_bundle(need[r, c], state["seconds"], state["firsts"], PRICE)
            if pb is None:
                continue
            res.append(dict(team=team, out=outs[c], inc=ins[r], o_sal=o_sal[c], in_sal=i_sal[r], post=post[r, c],
                            o_match=o_match[c], in_match=i_match[r],
                            o_recv=o_recv[c], in_recv=i_recv[r], kick_cash_me=o_cash[c], kick_cash_t=i_cash[r],
                            # pick_val = what the picks are worth to us, mkt_pick = what the partner counts them at
                            pick_val=pb[1] * picks["second"] + len(pb[2]) * picks["first"], mkt_pick=pb[0],
                            n2=pb[1], firsts=pb[2],
                            tpe=float(tp[r, c]) if o_n[c] == 0 else None,
                            d_war=float(i_war[r] - o_war[c]), fill=int(fill[r, c]),
                            waive_t=int(ex_t[r, c]), waive_t_cost=float(cost_t[r, c]),
                            waive_me=int(ex_me[r, c]),
                            waive_me_cost=float(my_waive[min(int(ex_me[r, c]), len(my_waive) - 1)])))
    return res


def apply(state: dict, t: dict, rosters_all: pd.DataFrame) -> dict:
    me, others = state["roster"], state["others"]
    got = others.loc[list(t["inc"])].assign(team=TEAM)
    sent = me.loc[list(t["out"])].assign(team=t["team"])
    # a bonus that got paid is now part of the salary, and it's used up
    for d in (got, sent):
        d["salary"] = d["salary"] + d["kick_now"]
        d[["kick_now", "kick_cash"]] = 0.0
    new = dict(state)
    new["roster"] = pd.concat([me.drop(list(t["out"])), got])
    new["others"] = pd.concat([others.drop(list(t["inc"])), sent])
    new["payroll"] = dict(state["payroll"])
    new["payroll"][TEAM] = t["post"]
    new["payroll"][t["team"]] = state["payroll"][t["team"]] - t["in_sal"] + t.get("o_recv", t["o_sal"])
    new["seconds"] = state["seconds"] - t["n2"]
    new["firsts"] = [f for f in state["firsts"] if f not in t["firsts"]]
    new["tpes"] = list(state["tpes"])
    if t["tpe"] is not None:
        new["tpes"].remove(t["tpe"])
    new["trades"] = state["trades"] + [t]
    new["moved"] = state["moved"] | set(t["inc"]) | set(t["out"])
    # value PHI gave away: surplus out + picks - surplus back
    new["value_out"] = state["value_out"] + (
        float(me.loc[list(t["out"]), "surplus"].sum()) + t["pick_val"] - float(got["surplus"].sum())
        + t.get("waive_me_cost", 0.0) + t.get("kick_cash_me", 0.0))
    new["kick_cash"] = state.get("kick_cash", 0.0) + t.get("kick_cash_me", 0.0)
    # whoever we waive leaves the roster (his money stays in the payroll), the least valuable guaranteed guys
    if t.get("waive_me", 0):
        g = new["roster"][new["roster"]["gtd"]].sort_values("worth").index[:t["waive_me"]]
        new["roster"] = new["roster"].drop(g)
    # minimum signings to get to 14 standard contracts: a replacement-level guy each
    if t.get("fill", 0):
        filler = pd.DataFrame([dict(player=f"minimum signing {len(new['trades'])}-{k}", team=TEAM, salary=MIN_SAL,
                                    player_id=np.nan, val=-1.0, min=0.0, std=True, gtd=True, rookie26=False,
                                    surplus=0.0, mkt=0.0, mkt_price=MIN_SAL, worth=0.0, avail=1.0, role="wing",
                                    out_sal=MIN_SAL,
                                    name="minimum signing") for k in range(t["fill"])],
                              index=[-(10_000 + 10 * len(new["trades"]) + k) for k in range(t["fill"])])
        new["roster"] = pd.concat([new["roster"], filler])
    # same in the market's money
    new["mkt_out"] = state.get("mkt_out", 0.0) + (
        float(me.loc[list(t["out"]), "mkt"].sum()) + t["mkt_pick"] - float(got["mkt"].sum()))
    return new


def search(budget: float, start: dict, rosters, prm, picks, curve, S, score) -> list:
    """beam search, returns every state it kept on the way (for the cheapest-to-target question too)"""
    beam, seen = [start], [start]
    for step in range(STEPS):
        nxt = []
        for st in beam:
            cands = candidates(st, picks, budget)
            # cheap filter on the WAR change, then the exact team strength for the best ones
            cands.sort(key=lambda t: -(t["d_war"] - 0.02 * t["pick_val"] / 1e6))
            kept = 0
            for t in cands:
                if kept >= TOP:
                    break
                # a contender (50%+ playoffs now) doesn't take a step back this season for future value
                if t["team"] in CONTENDERS:
                    them = st["others"][st["others"]["team"] == t["team"]]
                    after = pd.concat([them.drop(list(t["inc"])), st["roster"].loc[list(t["out"])]])
                    if S.team_net(after, prm) < S.team_net(them, prm) - 0.1:
                        continue
                kept += 1
                ns = apply(st, t, rosters)
                ns["net"] = S.team_net(ns["roster"], prm) - st["offset"]
                ns["tax"] = tax_bill(ns["payroll"][TEAM])
                nxt.append(ns)
        if not nxt:
            break
        nxt.sort(key=score, reverse=True)
        # no two states with the same roster
        uniq, keys = [], set()
        for ns in nxt:
            k = tuple(sorted(ns["roster"]["player"]))
            if k not in keys:
                keys.add(k)
                uniq.append(ns)
        beam = uniq[:BEAM]
        seen += uniq[:BEAM * 5]
    return seen


def describe(st: dict, base: dict, picks: dict) -> str:
    lines = []
    for t in st["trades"]:
        me_all = pd.concat([base["roster"], base["others"]])
        o = ", ".join(me_all.loc[list(t["out"]), "player"]) or "nothing"
        i = ", ".join(me_all.loc[list(t["inc"]), "player"]) or "nothing"
        pk = []
        if t["firsts"]:
            pk += t["firsts"]
        if t["n2"]:
            pk.append(f"{t['n2']} 2nd{'s' if t['n2'] > 1 else ''}")
        extra = f" + {' + '.join(pk)}" if pk else ""
        tp = f" (into the ${t['tpe'] / 1e6:.1f}M exception)" if t["tpe"] else ""
        lines.append(f"  {t['team']}: send {o}{extra} (${t['o_sal'] / 1e6:.1f}M) for {i} (${t['in_sal'] / 1e6:.1f}M){tp}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", choices=["mkt", "surplus"], default="mkt",
                    help="how the other team prices a trade: the market (default) or our own model")
    ap.add_argument("--picks", choices=["trades", "realized"], default="trades",
                    help="pick price for the partner: from real trades (default) or what picks turned into")
    ap.add_argument("--po-search", action="store_true",
                    help="search again with playoff strength as the goal (on top of the last run's states)")
    ap.add_argument("--reuse", action="store_true",
                    help="skip the search, pick the finalists again from the last run's states (new sim settings)")
    args = ap.parse_args()
    PRICE["col"], PRICE["picks"] = args.price, args.picks
    tag = args.price + ("" if args.picks == "trades" else "_realized")
    S, rosters, prm, nets, offset, picks, curve, dpw = load()
    print(f"$ per win next season ${dpw / 1e6:.1f}M. picks to us: a far-away 1st ${picks['first'] / 1e6:.1f}M, "
          f"a 2nd ${picks['second'] / 1e6:.1f}M. to the partner: ${PRICE['first'] / 1e6:.1f}M and "
          f"${PRICE['second'] / 1e6:.1f}M ({PRICE['source']})")
    me = rosters[rosters["team"] == TEAM].copy()
    start = dict(roster=me, others=rosters[rosters["team"] != TEAM].copy(), payroll=dict(prm["payroll"]),
                 seconds=SECONDS, firsts=list(FIRSTS), tpes=list(TPES), trades=[], value_out=0.0, mkt_out=0.0,
                 offset=offset, net=float(nets[TEAM]), moved=set())
    odds = pd.read_parquet(PROCESSED_DIR / "season_odds.parquet")
    CONTENDERS.update(odds.loc[(odds["playoffs"] >= 0.5) & (odds["team"] != TEAM), "team"])
    print("contenders (won't get weaker this season in a trade):", ", ".join(sorted(CONTENDERS)))
    start["tax"] = tax_bill(start["payroll"][TEAM])
    base_odds = odds_from_curve(curve, start["net"])
    print(f"{TEAM} now: payroll ${start['payroll'][TEAM] / 1e6:.1f}M, tax ${start['tax'] / 1e6:.1f}M, "
          f"net {start['net']:+.2f}, {base_odds['wins']:.1f} wins, title {base_odds['title']:.1%}")

    cost = lambda st: st["value_out"] + st["tax"] - start["tax"]
    cost_score = lambda st: st["net"] - 0.03 * cost(st) / 1e6
    shares = pd.read_parquet(PROCESSED_DIR / "playoff_shares.parquet")["po_share"].to_numpy()

    reg_shares = pd.read_parquet(PROCESSED_DIR / "playoff_shares.parquet")["reg_share"].to_numpy()

    def po_net(st):
        # PHI's playoff strength: regular-season strength + what the shorter rotation does to it
        return st["net"] + S.team_net_po(st["roster"], prm, shares) - S.team_net_po(st["roster"], prm, reg_shares)

    seen = []
    if args.reuse or args.po_search:
        seen = pd.read_pickle(PROCESSED_DIR / f"trade_search_{tag}.pkl")["seen"]
        print(f"reusing {len(seen)} states from the last search")
        # columns added since that run come from today's rosters
        av = rosters.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")["avail"]
        for st in seen:
            for k in ("roster", "others"):
                if "avail" not in st[k]:
                    st[k] = st[k].assign(avail=st[k]["player_id"].map(av).fillna(1.0))
        if args.po_search:
            for name, budget in BUDGETS.items():
                seen += search(budget, start, rosters, prm, picks, curve, S, score=po_net)
            tag += "_po"
    else:
        for name, budget in BUDGETS.items():
            if start["payroll"][TEAM] > budget:
                print(f"{name}: start is ${(start['payroll'][TEAM] - budget) / 1e6:.1f}M over, has to cut")
            seen += search(budget, start, rosters, prm, picks, curve, S, score=lambda st: st["net"])
            # a cost-aware pass too: more strength per $ given away, finds the cheap ways to a target
            seen += search(budget, start, rosters, prm, picks, curve, S, score=cost_score)

    def full_odds(st, n_sims=S.N_SIMS):
        # whole league with the partners changed too, playoffs with playoff rotations, 10,000 seasons
        league = pd.concat([st["roster"], st["others"]])
        n2, _ = S.league_nets(league, prm, offset)
        return S.simulate(n2, gm, sd, n_sims=n_sims, po_shift=S.playoff_shift(league, prm, offset)).set_index("team").loc[TEAM]



    gm = S.game_model(prm)
    sd = S.strength_sd(gm, prm)
    base_full = full_odds(start)
    print(f"\nfull sim, no trades: {base_full['wins']:.1f} wins, playoffs {base_full['playoffs']:.0%}, "
          f"finals {base_full['finals']:.0%}, title {base_full['title']:.1%}")

    def show(tag, st):
        o = full_odds(st)
        print(f"\n== {tag}: net {st['net']:+.2f}, {o['wins']:.1f} wins, playoffs {o['playoffs']:.0%}, "
              f"finals {o['finals']:.0%}, title {o['title']:.1%}")
        print(describe(st, start, picks))
        print(f"  payroll ${st['payroll'][TEAM] / 1e6:.1f}M, tax ${st['tax'] / 1e6:.1f}M "
              f"(now ${start['tax'] / 1e6:.1f}M). value given: ${st['value_out'] / 1e6:.0f}M by our model, "
              f"${st['mkt_out'] / 1e6:.0f}M by the market")
        return dict(tag=tag, net=st["net"], payroll=st["payroll"][TEAM], tax=st["tax"], value_out=st["value_out"],
                    mkt_out=st["mkt_out"], trades=describe(st, start, picks),
                    **{k: float(o[k]) for k in ("wins", "playoffs", "conf_finals", "finals", "title")})

    rows = []
    for name, budget in BUDGETS.items():
        ok = [st for st in seen if st["payroll"][TEAM] <= budget]
        # most title odds: the 10 strongest in the playoffs + the 5 strongest in the regular season, then the
        # full sim decides (3,000 seasons to pick, 10,000 for the one we show)
        top = sorted(ok, key=lambda st: -po_net(st))[:10] + sorted(ok, key=lambda st: -st["net"])[:5]
        best = max(top, key=lambda st: full_odds(st, 3000)["title"])
        rows.append(show(f"max title, under {name}", best))
        # one deal only, for an owner who won't sign off on three: same pick, from the one-trade states
        one = [st for st in ok if len(st["trades"]) == 1]
        if one:
            top1 = sorted(one, key=lambda st: -po_net(st))[:10] + sorted(one, key=lambda st: -st["net"])[:5]
            rows.append(show(f"one trade, under {name}", max(top1, key=lambda st: full_odds(st, 3000)["title"])))
        for tgt, w in (("52 wins", 52.0), ("60 wins", 60.0)):
            hit = [st for st in ok if odds_from_curve(curve, st["net"])["wins"] >= w]
            if hit:
                rows.append(show(f"{tgt} cheapest, under {name}", min(hit, key=cost)))
            else:
                print(f"\n== {tgt} under {name}: not reachable with PHI's assets")
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / f"trade_scenarios_{tag}.csv", index=False)
    pd.to_pickle(dict(seen=seen, start=start, picks=picks, price=PRICE), PROCESSED_DIR / f"trade_search_{tag}.pkl")
