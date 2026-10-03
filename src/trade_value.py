#!/usr/bin/env python3
"""
What the league pays for a pick, backed out of real trades.

If teams price players like the market does (market_value.py), then in every trade the gap in player value
between the two sides is what the picks (and cash) had to make up. Over ~400 trades that's a regression, one
equation per team per trade:

    market value of players in - out  =  p1 * (1sts out - in) + p2 * (2nds out - in) + p_cash * ... + noise

  market value of a player = market price of every season left on his contract (market_value.py, from the
  stats he had at the time, age moved on) - his salary (at least the minimum), in shares of that season's cap.
  a trade in February only counts the rest of this season. 1sts split into the next 2 drafts and further out.
  least squares, bootstrap over trades for the intervals. trades with a player we can't price (not matched,
  or the rights to a guy just drafted) are left out.

Same with our own model's value in place of the market's: which one makes real trades add up? If it's the
market's, the other GM really does think in points and minutes.

needs data/raw/salary_pages/transactions_*.html (python src/scrape_transactions.py, on your own machine)

Output
  data/processed/trades.parquet       one row per team per trade with what went in and out
  data/processed/trade_value.json     pick prices in $ at next season's cap (trade_search.py uses them)

  python src/trade_value.py
"""

import glob
import json
import re

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, SEASONS, season_label
from market_value import CAP, GROWTH, LO, RAISE, features, predict, salary_series, stats_table
from scrape_salaries import TO_NBA
from war import compact, unsuffix

SEASON_START, SEASON_END = (10, 20), (4, 14)      # regular season, roughly


# ── parsing ───────────────────────────────────────────────────────────────────

def clean(p: str) -> str:
    t = re.sub(r"\(<a [^>]*>[^<]*</a> was later selected\)", "", p)
    t = re.sub(r'<a [^>]*href="/players/[^"]*"[^>]*>(.*?)</a>', r"[P:\1]", t)
    t = re.sub(r'<a [^>]*data-attr-(from|to)="(\w+)"[^>]*>.*?</a>', r"{\1:\2}", t)
    t = re.sub(r"<[^>]+>", "", t)
    # the notes after the sentence ("2027 2nd-rd pick is DEN own", "Boston received a trade exception")
    return re.split(r"\.\s{2,}", t)[0].strip().rstrip(".")


def items(text: str, season: int) -> dict:
    ones = [int(y) for y in re.findall(r"(\d{4}) 1st round draft pick", text)]
    rights = re.findall(r"draft rights to \[P:(.*?)\]", text)
    players = [n for n in re.findall(r"\[P:(.*?)\]", text) if n not in rights]
    return dict(players=players, rights=rights,
                n1_near=sum(y <= season + 2 for y in ones), n1_far=sum(y > season + 2 for y in ones),
                n2=len(re.findall(r"\d{4} 2nd round draft pick", text)),
                cash=int("cash" in text), swap=len(re.findall(r"swap", text)))


def parse_trades() -> list:
    trades = []
    for f in sorted(glob.glob(str(RAW_DIR / "salary_pages" / "transactions_*.html"))):
        page = int(re.search(r"(\d{4})\.html", f).group(1))
        season = page - 1                          # NBA_2024 = the 2023-24 season (July 2023 on)
        h = open(f, encoding="utf-8").read()
        for date, body in re.findall(r"<li><span>(.*?)</span>(.*?)</li>", h, re.S):
            for p in re.findall(r"<p>(.*?)</p>", body, re.S):
                if " traded " not in p:
                    continue
                t = re.sub(r"^In a \d+-team trade, ", "", clean(p))
                legs = []
                for seg in re.split(r";\s*(?:and\s+)?", t):
                    m = re.search(r"\{from:(\w+)\} traded (.*?) to the \{to:(\w+)\}(?: for (.*))?$", seg.strip())
                    if not m:
                        continue
                    a, b = TO_NBA.get(m.group(1), m.group(1)), TO_NBA.get(m.group(3), m.group(3))
                    legs.append((a, b, items(m.group(2), season)))
                    if m.group(4):
                        legs.append((b, a, items(m.group(4), season)))
                if legs:
                    trades.append(dict(date=pd.to_datetime(date, errors="coerce"), season=season, legs=legs, text=t))
    return trades


# ── value at the time of the trade ────────────────────────────────────────────

class Valuer:
    """market value (and our model's value) of a player's remaining contract at a date"""

    def __init__(self):
        self.m = json.loads((PROCESSED_DIR / "market_model.json").read_text())
        self.m = dict(cols=self.m["cols"], mu=np.array(self.m["mu"]), sd=np.array(self.m["sd"]),
                      beta=np.array(self.m["beta"]), sigma=self.m["sigma"])
        self.stats = stats_table()
        sal = salary_series().sort_values(["player_id", "season"])
        prev, prev_s = sal.groupby("player_id")["salary"].shift(1), sal.groupby("player_id")["season"].shift(1)
        r = sal["salary"] / prev
        sal["new"] = prev.isna() | (prev_s != sal["season"] - 1) | (r < RAISE[0]) | (r > RAISE[1])
        sal["deal"] = sal.groupby("player_id")["new"].cumsum()
        self.sal = sal.set_index(["player_id", "season"])
        # names -> ids, per season (the same name can be two players over 15 years)
        p = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["SEASON", "PLAYER_ID", "PLAYER_NAME"]]
                       for y in SEASONS])
        self.names = {}
        for s, pid, n in zip(p["SEASON"], p["PLAYER_ID"], p["PLAYER_NAME"]):
            for k in {compact(n), unsuffix(compact(n))}:
                self.names.setdefault((k, s), set()).add(pid)
        self.ours = self._ours()

    def _ours(self):
        """our model: projected WAR as of s-1 (war.py) * $ per WAR, aged, for the value comparison"""
        from surplus import aging_fn
        w = pd.read_parquet(PROCESSED_DIR / "war.parquet")
        prm = json.loads((PROCESSED_DIR / "war_params.json").read_text())
        self.age_fn, self.prm = aging_fn(), prm
        return w.assign(season=w["season"] + 1).set_index(["player_id", "season"])

    def pid(self, name: str, season: int):
        for s in (season, season - 1, season + 1):
            for k in (compact(name), unsuffix(compact(name))):
                hit = self.names.get((k, s))
                if hit and len(hit) == 1:
                    return next(iter(hit))
        return None

    def contract(self, pid: int, season: int) -> pd.Series:
        """salary by season for the rest of the deal he's on in `season`"""
        if (pid, season) not in self.sal.index:
            return pd.Series(dtype=float)
        s = self.sal.loc[pid]
        deal = s.loc[season, "deal"]
        s = s[(s["deal"] == deal) & (s.index >= season)]
        return s["salary"]

    def value(self, pid: int, season: int, frac: float):
        """salary, market price, star term, our worth of his remaining contract, shares of the trade season's cap.
        None if he has no contract that season (a just-drafted guy's rights, a stash): can't price those"""
        c = self.contract(pid, season)
        if c.empty:
            return None
        f = features(self.stats, pd.DataFrame({"player_id": [pid] * len(c), "season": season}), skip_missed=True)
        k = c.index.to_numpy() - season
        f["age"] = f["age"] + k
        f["age2"] = (f["age"] - 28) ** 2
        # no stats in the 3 seasons before (overseas, G League): priced like a minimum guy
        share = np.nan_to_num(predict(self.m, f), nan=LO)
        cap_y = np.array([CAP.get(y, CAP[max(CAP)] * (1 + GROWTH) ** (y - max(CAP))) for y in c.index])
        w = np.where(k == 0, frac, 1.0) * cap_y / CAP[season]
        sal_share = c.to_numpy() / cap_y
        out = dict(S=float(np.sum(w * sal_share)), MV=float(np.sum(w * share)), W=0.0,
                   # market surplus: market price - salary, nobody is paid under the minimum (market_value.py)
                   mkt=float(np.sum(w * (share - np.maximum(sal_share, LO)))), ours=0.0)
        if (pid, season) in self.ours.index:
            o = self.ours.loc[(pid, season)]
            prm = self.prm
            kk = prm["pace"].get(str(season - 1), list(prm["pace"].values())[-1]) / 100 / prm["ppw"]
            dpw_share = 7.7e6 / CAP[max(SEASONS)]      # our $ per WAR as a share of the cap ($7.7M in 2025-26)
            raw = o["val"] + self.age_fn(o["age_next"] + k) - self.age_fn(o["age_next"])
            war = np.clip((prm["a"] * raw + prm["c"] - prm["repl"]) * o["proj_min"] * kk, 0, None)
            out["W"] = float(np.sum(w * war * dpw_share))
        out["ours"] = out["W"] - out["S"]
        return out


def season_fraction(date: pd.Timestamp, season: int) -> float:
    if pd.isna(date):
        return 1.0
    start = pd.Timestamp(season, *SEASON_START)
    end = pd.Timestamp(season + 1, *SEASON_END)
    if date <= start:
        return 1.0
    return float(np.clip((end - date).days / (end - start).days, 0, 1))


# ── fit ───────────────────────────────────────────────────────────────────────

def huber(X, y, delta=None, iters=50):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ b
        d = delta or 1.345 * np.median(np.abs(r)) / 0.6745
        w = np.where(np.abs(r) <= d, 1.0, d / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(w)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return b, w


PICK_COLS = ["n1_near", "n1_far", "n2", "cash", "swap"]


def fit(t: pd.DataFrame, ycol: str, boot: int = 500) -> dict:
    """value in - value out ~ picks out - picks in, least squares (the big trades carry the pick prices, a
    robust fit would mute exactly those), bootstrap over trades"""
    X, y = t[[f"d_{c}" for c in PICK_COLS]].to_numpy(float), t[ycol].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    rng = np.random.default_rng(7)
    groups = [np.flatnonzero(t["trade"].to_numpy() == i) for i in t["trade"].unique()]
    bs = np.array([np.linalg.lstsq(X[idx], y[idx], rcond=None)[0] for idx in
                   (np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
                    for _ in range(boot))])
    return dict(beta=dict(zip(PICK_COLS, b)), lo=dict(zip(PICK_COLS, np.percentile(bs, 2.5, 0))),
                hi=dict(zip(PICK_COLS, np.percentile(bs, 97.5, 0))),
                mae=float(np.mean(np.abs(res))), med=float(np.median(np.abs(res))))


if __name__ == "__main__":
    trades = parse_trades()
    V = Valuer()
    rows, miss, tot = [], 0, 0
    for i, tr in enumerate(trades):
        s = tr["season"]
        if s < min(SEASONS) + 3 or s > max(SEASONS) + 1:
            continue
        frac = season_fraction(tr["date"], s)
        teams = sorted({a for a, _, _ in tr["legs"]} | {b for _, b, _ in tr["legs"]})
        net = {t_: dict(mkt=0.0, ours=0.0, S=0.0, n_in=0, n_out=0, **{c: 0 for c in PICK_COLS}) for t_ in teams}
        ok = True
        for a, b, it in tr["legs"]:
            # draft-night deals for this year's picks are about specific slots, left out
            if it["rights"] and not pd.isna(tr["date"]) and tr["date"].month in (6, 7):
                ok = False
            for n in it["players"]:
                tot += 1
                pid = V.pid(n, s)
                v = V.value(pid, s, frac) if pid is not None else None
                if v is None:
                    # a player we can't price (not matched, or just drafted): the whole trade goes, a 0 for him
                    # would look like the picks bought nothing
                    miss += 1
                    ok = False
                    continue
                for k in ("mkt", "ours", "S"):
                    net[b][k] += v[k]
                    net[a][k] -= v[k]
                net[b]["n_in"] += 1
                net[a]["n_out"] += 1
            for c in PICK_COLS:
                # d_x = what this team gave minus what it got
                net[a][c] += it[c]
                net[b][c] -= it[c]
        if not ok:
            continue
        # k teams: k-1 equations (the last one is the others summed)
        for t_ in teams[:-1]:
            r = net[t_]
            rows.append(dict(trade=i, season=s, date=tr["date"], team=t_, n_teams=len(teams), n_in=r["n_in"],
                             n_out=r["n_out"], mkt=r["mkt"], ours=r["ours"], d_S=r["S"],
                             **{f"d_{c}": r[c] for c in PICK_COLS}, text=tr["text"][:300]))
    t = pd.DataFrame(rows)
    print(f"{t['trade'].nunique()} trades ({season_label(t['season'].min())} to {season_label(t['season'].max())}), "
          f"{len(t)} team equations. {miss} of {tot} players couldn't be priced (their trades left out)")
    t.to_parquet(PROCESSED_DIR / "trades.parquet", index=False)

    cap_next = CAP[max(SEASONS) + 1]
    fits = {}
    for ycol, label in (("mkt", "the market's price"), ("ours", "our model")):
        r = fits[ycol] = fit(t, ycol)
        print(f"\nplayers valued by {label}: trades miss by {r['mae'] * 100:.1f}% of cap on average "
              f"(median {r['med'] * 100:.1f}%)")
        for c in PICK_COLS:
            print(f"  {c:8s} {r['beta'][c] * 100:6.2f}% of cap  [{r['lo'][c] * 100:6.2f}, {r['hi'][c] * 100:6.2f}]"
                  f"  = ${r['beta'][c] * cap_next / 1e6:5.1f}M at the {season_label(max(SEASONS) + 1)} cap")
    b, f = fits["mkt"]["beta"], fits["mkt"]
    # PHI's 1st is 2033 = far away. a 2nd is a 2nd
    tv = dict(first=float(b["n1_far"] * cap_next), first_near=float(b["n1_near"] * cap_next),
              second=float(max(b["n2"], 0) * cap_next), cash=float(b["cash"] * cap_next),
              ci_first=[float(f["lo"]["n1_far"] * cap_next), float(f["hi"]["n1_far"] * cap_next)],
              ci_second=[float(f["lo"]["n2"] * cap_next), float(f["hi"]["n2"] * cap_next)],
              mae={k: v["mae"] for k, v in fits.items()}, n_trades=int(t["trade"].nunique()))
    (PROCESSED_DIR / "trade_value.json").write_text(json.dumps(tv, indent=1))
    print(f"\n-> trade_value.json: far 1st ${tv['first'] / 1e6:.1f}M, near 1st ${tv['first_near'] / 1e6:.1f}M, "
          f"2nd ${tv['second'] / 1e6:.1f}M")
