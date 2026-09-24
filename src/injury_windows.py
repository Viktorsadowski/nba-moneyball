#!/usr/bin/env python3
"""
Step 9c: injuries measured in games instead of seasons.

injury_types.py compares the injury season with the next season. That mixes things up: hurt in October,
back in January -> "next season" is really his second season back. hurt in March -> "next season" starts
right after he's back. So here everything is counted in games he played around the injury:

  pre window    last 82 games he played before the injury (within 3 years), his baseline
  rust window   first 20 games after he's back
  post window   first 82 games after he's back (within 2 years), rust included
  back          did he play 41+ games in the 2 years after the injury started

Controls: the same windows around points in healthy stretches (a "pivot" game, no 10+ game injury there),
same rules, so shrinkage and regression to the mean hit both groups the same way.

Impact per window comes from one ridge over all 16 seasons of stints:
  fit 1   pre windows get their own columns, all other games go to normal player-season columns
  fit 2   rust + rest of the post window get their own columns, fitted as a CHANGE from the same guy's
          pre estimate (his pre value is taken out of y as an offset). so a 20-game window shrinks
          towards "no change" instead of towards 0 = average player

Then like injury_types.py: change minus aging, residualised on pre level and age, per injury type
vs the controls, bootstrap CI + empirical-Bayes shrinkage.

Windows never overlap within a player: a new injury inside a post window cuts that post window there,
and the new injury's pre window only uses games after the previous injury's return. Controls sit 82+
games after the last event. Injuries and controls get their own fits, so an injury's pre window can
still use games that are also in a control's window.

Output
  data/processed/injury_events.parquet    one row per injury / control, windows + estimates
  data/processed/injury_windows.parquet   effects per type
  figures/injury_windows.png

  python src/injury_windows.py
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from availability import build_name_index, compact, is_injury, note_rank, resolve, team_id, unsuffix
from config import PROCESSED_DIR, RAW_DIR, ROOT, SEASONS
from injury_types import MIN_GAMES, aging_fn, effects, injury_type

PRE, POST, RUST = 82, 82, 20
PRE_DAYS, POST_DAYS = 3 * 365, 730
MIN_PRE_GAMES, MIN_PRE_MIN = 41, 500     # established before the injury
MIN_POST_GAMES = 41                      # enough of a post window to say something
BACK_GAMES, BACK_DAYS = 41, 730
LAMBDA = 3000                            # what the CV picked for most seasons in rapm.py
H = [f"h{i}" for i in range(1, 6)]
A = [f"a{i}" for i in range(1, 6)]
RNG = np.random.default_rng(5)


# ── games ─────────────────────────────────────────────────────────────────────

def load_games():
    rosters = {y: pd.read_parquet(RAW_DIR / f"roster_{y}.parquet") for y in SEASONS}
    scheds = {y: pd.read_parquet(RAW_DIR / f"team_games_{y}.parquet") for y in SEASONS}
    apps = pd.concat([rosters[y][["GAME_ID", "TEAM_ID", "PLAYER_ID", "MIN"]]
                      .merge(scheds[y][["GAME_ID", "TEAM_ID", "GAME_DATE"]], on=["GAME_ID", "TEAM_ID"])
                      .assign(season=y) for y in SEASONS], ignore_index=True)
    apps.columns = ["game_id", "team_id", "pid", "min", "date", "season"]
    apps = apps.drop_duplicates(["pid", "game_id"]).sort_values(["pid", "date"]).reset_index(drop=True)
    sched = pd.concat(scheds.values(), ignore_index=True)
    return rosters, scheds, apps, sched


# ── injury spells, with the date he played again ──────────────────────────────

def report_missed(year, roster, sched, rosters) -> pd.DataFrame:
    # same matching as availability.report_injuries, but keeps every missed game (need the dates here)
    rep = pd.read_parquet(RAW_DIR / f"injury_report_rows_{year}.parquet")
    rep = rep[rep["reason"].str.startswith("Injury/Illness", na=False)].dropna(subset=["date"])
    rep = rep.sort_values("report").drop_duplicates(["date", "team", "player"], keep="last")

    from nba_api.stats.static import teams
    nick = {compact(t["nickname"]): t["id"] for t in teams.get_teams()}
    rep["team_id"] = [next((i for n, i in nick.items() if compact(t).endswith(n)), None) for t in rep["team"]]

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

    g = sched[["GAME_ID", "TEAM_ID", "GAME_DATE"]].rename(columns={"TEAM_ID": "team_id", "GAME_DATE": "date"})
    m = rep.dropna(subset=["pid"]).merge(g, on=["team_id", "date"], how="inner")
    played = set(zip(roster["GAME_ID"], roster["PLAYER_ID"]))
    m = m[np.array([(gid, p) not in played for gid, p in zip(m["GAME_ID"], m["pid"])], dtype=bool)]
    return m.assign(pid=m["pid"].astype(int))[["pid", "team_id", "date", "reason"]]


def all_spells(rosters, scheds, apps, sched) -> pd.DataFrame:
    played = {p: g["date"].to_numpy() for p, g in apps.groupby("pid")}
    team_dates = {t: np.sort(g["GAME_DATE"].unique()) for t, g in sched.groupby("TEAM_ID")}

    def ret_of(pid, start):
        d = played.get(pid)
        if d is None:
            return pd.NaT
        i = np.searchsorted(d, np.datetime64(start), side="right")
        return pd.Timestamp(d[i]) if i < len(d) else pd.NaT

    # PST log, 2010-11 to 2019-20. every "out" row on its own, the absence runs to his next game played.
    # (not build_spells: that folds an earlier DNP without a logged return and a later torn ACL into one
    # spell, which then ends at his next game and the ACL is gone)
    idx = build_name_index(rosters)
    rows = []
    for y in SEASONS:
        f = RAW_DIR / f"injuries_{y}.parquet"
        if not f.exists() or (RAW_DIR / f"injury_report_rows_{y}.parquet").exists():
            continue
        for r in pd.read_parquet(f).fillna("").itertuples(index=False):
            if r.Relinquished and is_injury(r.Notes):
                tid = team_id(r.Team, y)
                rows.append((resolve(r.Relinquished, y, tid, idx), tid, pd.Timestamp(r.Date), r.Notes, y))
    pst = pd.DataFrame(rows, columns=["pid", "team_id", "start", "note", "season"]).dropna(subset=["pid", "start"])
    pst["pid"] = pst["pid"].astype(int)
    pst["ret"] = [ret_of(p, s) for p, s in zip(pst["pid"], pst["start"])]
    last_team = apps.groupby("pid")["team_id"].last()
    missed = []
    for r in pst.itertuples(index=False):
        t = r.team_id if pd.notna(r.team_id) else last_team.get(r.pid)
        # his team's games from going out to playing again, also across the summer.
        # never played again = to the end of that season
        stop = r.ret if pd.notna(r.ret) else pd.Timestamp(f"{r.season + 1}-07-31")
        td = team_dates.get(t, np.array([], dtype="datetime64[ns]"))
        missed.append(int(np.searchsorted(td, np.datetime64(stop)) - np.searchsorted(td, np.datetime64(r.start))))
    pst["missed"] = missed
    pst["src"] = "pst"

    # official reports, 2021-22 on. consecutive missed games with no game played in between = one spell,
    # that also runs across the summer
    m = pd.concat([report_missed(y, rosters[y], scheds[y], rosters) for y in SEASONS
                   if (RAW_DIR / f"injury_report_rows_{y}.parquet").exists()], ignore_index=True)
    m = m.sort_values(["pid", "date"])
    m["n_played"] = [np.searchsorted(played.get(p, np.array([], dtype="datetime64[ns]")), np.datetime64(d),
                                     side="right") for p, d in zip(m["pid"], m["date"])]
    rep = m.groupby(["pid", "n_played"]).agg(start=("date", "first"), last=("date", "last"),
                                             missed=("date", "size"), team_id=("team_id", "first"),
                                             note=("reason", lambda x: x.value_counts().index[0])).reset_index()
    rep["ret"] = [ret_of(p, s) for p, s in zip(rep["pid"], rep["last"])]
    rep["src"] = "report"

    sp = pd.concat([pst[["pid", "start", "ret", "missed", "note", "src"]],
                    rep[["pid", "start", "ret", "missed", "note", "src"]]], ignore_index=True)
    # same return = same absence (PST relists guys: DTD -> out -> out for season). keep the first start and
    # the biggest count. note: the most serious one, so "strained calf (DTD)" in May + "torn Achilles" in
    # June is an Achilles
    sp["ret_key"] = sp["ret"].fillna(pd.Timestamp("2100-01-01"))
    sp["rank"] = [note_rank(n) for n in sp["note"]]
    note = sp.sort_values("rank").drop_duplicates(["pid", "ret_key"]).set_index(["pid", "ret_key"])["note"]
    sp = sp.sort_values(["pid", "start"])
    sp = sp.groupby(["pid", "ret_key"]).agg(start=("start", "first"), ret=("ret", "first"), missed=("missed", "max"),
                                            src=("src", "first"))
    sp["note"] = note
    return sp.reset_index().drop(columns="ret_key")


# ── windows ───────────────────────────────────────────────────────────────────

def build_events(apps: pd.DataFrame, spells: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """walk each player's games in order: injuries where they happened, controls in the healthy stretches."""
    q = spells[spells["missed"] >= MIN_GAMES]
    q_by = {p: g.sort_values("start") for p, g in q.groupby("pid")}
    ev = []
    for pid, g in apps.groupby("pid", sort=False):
        dates = g["date"].to_numpy()
        mins = g["min"].to_numpy()
        n = len(dates)
        prev = 0            # injury pre windows only use games from here on (= previous injury's return)
        last = {}           # last accepted injury / control, their post can still get cut

        def add(kind, s_pos, r_pos, start, extra):
            nonlocal prev
            # an injury cuts both open post windows, a pivot only the previous control's (no-op, it's 82+ later)
            for k in (("injury", "control") if kind == "injury" else ("control",)):
                if k in last:
                    last[k]["post_hi"] = min(last[k]["post_hi"], s_pos)
            # controls sit 82+ games after the last event anyway. injuries and controls get fitted
            # separately, so an injury's pre window may run through a control's windows
            lo = max(prev, s_pos - PRE)
            # pre within 3 years of the start
            lo = max(lo, int(np.searchsorted(dates, np.datetime64(start - pd.Timedelta(days=PRE_DAYS)))))
            hi_date = dates[r_pos] + np.timedelta64(POST_DAYS, "D") if r_pos < n else None
            post_hi = min(r_pos + POST, int(np.searchsorted(dates, hi_date, side="right"))) if hi_date is not None else r_pos
            e = dict(pid=pid, kind=kind, start=start, pre_lo=lo, pre_hi=s_pos, post_lo=r_pos, post_hi=post_hi,
                     pre_games=s_pos - lo, pre_min=float(mins[lo:s_pos].sum()), **extra)
            # games in the 2 years after it started (the injury game itself isn't played, the pivot is)
            end = np.datetime64(start + pd.Timedelta(days=BACK_DAYS))
            e["games_2y"] = int(np.searchsorted(dates, end) - s_pos)
            e["censored"] = start + pd.Timedelta(days=BACK_DAYS) > data_end
            ok = e["pre_games"] >= MIN_PRE_GAMES and e["pre_min"] >= MIN_PRE_MIN
            if ok:
                ev.append(e)
                last[kind] = e
            # every 10+ game injury resets the pre windows, accepted or not. a rejected one still cuts the
            # previous post window (above), otherwise that post would run through a second long injury
            if kind == "injury":
                prev = r_pos
            return ok

        def next_pivot(from_pos):
            return from_pos + PRE + int(RNG.integers(0, PRE // 2))

        pivot = next_pivot(0)
        for s in (q_by[pid].itertuples(index=False) if pid in q_by else []):
            s_pos = int(np.searchsorted(dates, np.datetime64(s.start), side="left"))
            r_pos = int(np.searchsorted(dates, np.datetime64(s.start), side="right"))
            while pivot < s_pos:
                add("control", pivot, pivot, pd.Timestamp(dates[pivot]), dict(note="", missed=0, ret=pd.Timestamp(dates[pivot])))
                pivot = next_pivot(pivot)
            add("injury", s_pos, r_pos, s.start, dict(note=s.note, missed=s.missed, ret=s.ret))
            pivot = next_pivot(r_pos)
        while pivot < n:
            add("control", pivot, pivot, pd.Timestamp(dates[pivot]), dict(note="", missed=0, ret=pd.Timestamp(dates[pivot])))
            pivot = next_pivot(pivot)
    e = pd.DataFrame(ev)
    e["post_games"] = e["post_hi"] - e["post_lo"]
    return e.reset_index(drop=True)


# ── ridge with window columns ─────────────────────────────────────────────────

def load_stints() -> pd.DataFrame:
    cols = ["season", "game_id", *H, *A, "pts_h", "pts_a", "poss"]
    st = pd.concat([pd.read_parquet(PROCESSED_DIR / f"stints_{y}.parquet", columns=cols) for y in SEASONS],
                   ignore_index=True)
    return st[st["poss"] > 0].reset_index(drop=True)


def window_keys(ev: pd.DataFrame, apps: pd.DataFrame, parts: dict) -> pd.Series:
    """(pid, game) combo -> column key for the games inside the chosen window parts."""
    gids = {p: g["game_id"].to_numpy() for p, g in apps.groupby("pid")}
    combos, keys = [], []
    for i, r in zip(ev.index, ev.itertuples(index=False)):
        g = gids[r.pid]
        for part, (lo, hi) in parts.items():
            a, b = lo(r), hi(r)
            if b <= a:
                continue
            c = r.pid * 10 ** 10 + g[a:b].astype(np.int64)
            combos.append(c)
            keys.append(np.full(len(c), -(i * 3 + part) - 1, dtype=np.int64))
    return pd.Series(np.concatenate(keys), index=np.concatenate(combos))


def fit(st: pd.DataFrame, override: pd.Series, offset_o=None, offset_d=None):
    """one ridge over all seasons. player-season columns, except the (pid, game) combos in override.
    offsets (per override key) are taken out of y, so those columns come out as a change from them."""
    P = st[H + A].to_numpy().astype(np.int64)
    n = len(st)
    combo = P * 10 ** 10 + st["game_id"].astype(np.int64).to_numpy()[:, None]
    keys = P * 10000 + st["season"].to_numpy()[:, None]
    pos = override.index.get_indexer(combo.ravel())
    hit = pos >= 0
    keys = keys.ravel()
    keys[hit] = override.to_numpy()[pos[hit]]
    col, uniq = pd.factorize(keys)
    col = col.reshape(n, 10)
    nc = len(uniq)

    def side(off, dfn, pts):
        r = np.repeat(np.arange(n), 5)
        rows = np.concatenate([r, r, np.arange(n)])
        c = np.concatenate([col[:, off].ravel(), col[:, dfn].ravel() + nc, np.full(n, 2 * nc)])
        v = np.concatenate([np.ones(10 * n), np.full(n, 1.0 if pts == "pts_h" else 0.0)])
        X = sparse.csr_matrix((v, (rows, c)), shape=(n, 2 * nc + 1))
        y = 100 * st[pts].to_numpy() / st["poss"].to_numpy()
        if offset_o is not None:
            # y - sum(pre O of the window guys on offense) + sum(pre D of the ones on defense)
            oo = pd.Series(uniq).map(offset_o).fillna(0).to_numpy()
            dd = pd.Series(uniq).map(offset_d).fillna(0).to_numpy()
            y = y - oo[col[:, off]].sum(1) + dd[col[:, dfn]].sum(1)
        return X, y

    Xh, yh = side(slice(0, 5), slice(5, 10), "pts_h")
    Xa, ya = side(slice(5, 10), slice(0, 5), "pts_a")
    X = sparse.vstack([Xh, Xa]).tocsr()
    y = np.concatenate([yh, ya])
    w = np.concatenate([st["poss"].to_numpy()] * 2)
    m = Ridge(alpha=LAMBDA).fit(X, y, sample_weight=w)
    poss = X[:, :nc].T @ w
    return pd.DataFrame({"key": uniq, "o": m.coef_[:nc], "d": -m.coef_[nc:2 * nc], "poss": poss})


# ── analysis ──────────────────────────────────────────────────────────────────

def resid(x: pd.DataFrame, col: str, w: str | None, sq=False) -> pd.Series:
    # take out pre level and age (fitted on injuries + controls together), keep the overall mean
    ok = x[col].notna()
    Z = [np.ones(ok.sum()), x.loc[ok, "pre"], x.loc[ok, "age"]] + ([x.loc[ok, "age"] ** 2] if sq else [])
    Z = np.column_stack(Z)
    ww = np.sqrt(x.loc[ok, w].to_numpy()) if w else np.ones(ok.sum())
    b = np.linalg.lstsq(Z * ww[:, None], x.loc[ok, col].to_numpy() * ww, rcond=None)[0]
    out = pd.Series(np.nan, index=x.index)
    out[ok] = x.loc[ok, col] - Z @ b + np.average(x.loc[ok, col], weights=ww ** 2)
    return out


def plot(back, rust, post) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
    order = back.sort_values("shrunk")["group"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(surface)
    for ax, t, title, scale, fmt in (
        (axes[0], back, f"Played {BACK_GAMES}+ games within 2 years", 100, "{:+.0f} pts"),
        (axes[1], rust, f"First {RUST} games back, change vs before", 1, "{:+.2f}"),
        (axes[2], post, f"First {POST} games back, change vs before", 1, "{:+.2f}"),
    ):
        ax.set_facecolor(surface)
        t = t.set_index("group").reindex(order)
        y = np.arange(len(order))
        ax.hlines(y, t["lo"] * scale, t["hi"] * scale, color=muted, lw=1.2, zorder=2)
        ax.scatter(t["raw"] * scale, y, s=26, facecolor=surface, edgecolor=muted, lw=1.2, zorder=3, label="raw")
        ax.scatter(t["shrunk"] * scale, y, s=46, color=blue, zorder=4, label="shrunk (believe this one)")
        for yi, (v, n) in enumerate(zip(t["shrunk"], t["n"])):
            if pd.notna(v):
                ax.annotate(f"{fmt.format(v * scale)}  n={int(n)}", (ax.get_xlim()[1], yi), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=muted,
                            annotation_clip=False)
        ax.axvline(0, color=ink, lw=0.8)
        ax.set_yticks(y, order)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.grid(axis="x", color=grid, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8.5, left=False)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("percentage points vs controls", color=muted, fontsize=8.5)
    for ax in axes[1:]:
        ax.set_xlabel("RAPM change vs controls, points per 100", color=muted, fontsize=8.5)
    # top right next to the title, every panel is full somewhere
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper right", ncol=2,
               labelcolor=ink, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f"Injuries in games, not seasons ({MIN_GAMES}+ games lost, vs healthy stretches of the same length)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "injury_windows.png", facecolor=surface)
    print(f"\nplot -> {out / 'injury_windows.png'}")


if __name__ == "__main__":
    rosters, scheds, apps, sched = load_games()
    data_end = apps["date"].max()
    spells = all_spells(rosters, scheds, apps, sched)
    q = spells[spells["missed"] >= MIN_GAMES]
    print(f"{len(q):,} absences of {MIN_GAMES}+ games ({(q['src'] == 'pst').sum():,} PST, "
          f"{(q['src'] == 'report').sum():,} reports), {q['ret'].isna().sum()} never played again")

    ev = build_events(apps, spells, data_end)
    inj = ev["kind"] == "injury"
    print(f"events with an established pre window: {inj.sum():,} injuries, {(~inj).sum():,} controls")

    # ages: rough birthday from the season ages (AGE in players_ is his age that season)
    pl = pd.concat([pd.read_parquet(RAW_DIR / f"players_{y}.parquet")[["PLAYER_ID", "AGE"]].assign(y=y) for y in SEASONS])
    birth = (pd.to_datetime((pl["y"] + 1).astype(str) + "-02-01") - pd.to_timedelta(pl["AGE"] * 365.25, unit="D")) \
        .groupby(pl["PLAYER_ID"]).mean()
    dates = {p: g["date"].to_numpy() for p, g in apps.groupby("pid")}

    def mid_age(r, lo, hi):
        if hi <= lo:
            return np.nan
        d = dates[r.pid][lo:hi]
        return (pd.Timestamp(d[len(d) // 2]) - birth.get(r.pid, pd.NaT)).days / 365.25
    ev["age"] = [(r.start - birth.get(r.pid, pd.NaT)).days / 365.25 for r in ev.itertuples()]
    ev["age_pre"] = [mid_age(r, r.pre_lo, r.pre_hi) for r in ev.itertuples()]
    ev["age_rust"] = [mid_age(r, r.post_lo, min(r.post_hi, r.post_lo + RUST)) for r in ev.itertuples()]
    ev["age_post"] = [mid_age(r, r.post_lo, r.post_hi) for r in ev.itertuples()]

    # injuries and controls in separate fits (their windows can overlap), 4 ridges in total
    st = load_stints()
    print(f"ridge over {len(st):,} stints, lambda {LAMBDA}, injuries and controls fitted separately")
    for kind in ("injury", "control"):
        sub = ev[ev["kind"] == kind]
        i = sub.index.to_numpy()

        # fit 1: pre windows
        c1 = fit(st, window_keys(sub, apps, {0: (lambda r: r.pre_lo, lambda r: r.pre_hi)})).set_index("key")
        k_pre = -(i * 3 + 0) - 1
        ev.loc[i, "pre_o"] = c1["o"].reindex(k_pre).to_numpy()
        ev.loc[i, "pre_d"] = c1["d"].reindex(k_pre).to_numpy()

        # fit 2: rust + rest of the post window, as a change from pre
        ov2 = window_keys(sub, apps, {1: (lambda r: r.post_lo, lambda r: min(r.post_hi, r.post_lo + RUST)),
                                      2: (lambda r: r.post_lo + RUST, lambda r: r.post_hi)})
        off_o, off_d = {}, {}
        for part in (1, 2):
            k = -(i * 3 + part) - 1
            off_o.update(dict(zip(k, ev.loc[i, "pre_o"].fillna(0))))
            off_d.update(dict(zip(k, ev.loc[i, "pre_d"].fillna(0))))
        c2 = fit(st, ov2, off_o, off_d).set_index("key")
        for part, name in ((1, "rust"), (2, "rest")):
            k = -(i * 3 + part) - 1
            ev.loc[i, f"chg_{name}"] = (c2["o"] + c2["d"]).reindex(k).to_numpy()
            ev.loc[i, f"poss_{name}"] = c2["poss"].reindex(k).fillna(0).to_numpy()
    ev["pre"] = ev["pre_o"] + ev["pre_d"]
    pr, ps = ev["poss_rust"], ev["poss_rest"]
    ev["chg_post"] = (ev["chg_rust"].fillna(0) * pr + ev["chg_rest"].fillna(0) * ps) / (pr + ps).replace(0, np.nan)
    ev["poss_post"] = pr + ps

    # aging between the windows (the injured guys sat out, so their gap is longer)
    age_fn = aging_fn()
    ev["rust"] = ev["chg_rust"] - (age_fn(ev["age_rust"]) - age_fn(ev["age_pre"]))
    ev["post"] = ev["chg_post"] - (age_fn(ev["age_post"]) - age_fn(ev["age_pre"]))
    ev.loc[ev["post_games"] < RUST, "rust"] = np.nan
    ev.loc[ev["post_games"] < MIN_POST_GAMES, "post"] = np.nan
    ev["group"] = np.where(ev["kind"] == "control", "No injury", ev["note"].map(injury_type))
    ev.to_parquet(PROCESSED_DIR / "injury_events.parquet", index=False)

    # ── effects ──
    x = ev.dropna(subset=["pre", "age"]).copy()
    x["back_raw"] = (x["games_2y"] >= BACK_GAMES).astype(float)
    x.loc[x["censored"], "back_raw"] = np.nan
    x["back"] = resid(x, "back_raw", None, sq=True)
    x["rust_r"] = resid(x, "rust", "poss_rust")
    x["post_r"] = resid(x, "post", "poss_post")
    back = effects(x, "back", None)
    rust = effects(x, "rust_r", "poss_rust")
    post = effects(x, "post_r", "poss_post")

    ctrl = x["group"] == "No injury"
    print(f"\ncontrols: {x.loc[ctrl, 'back_raw'].mean():.0%} played {BACK_GAMES}+ games in the next 2 years")
    # pooled, and by how long he was out
    x["length"] = pd.cut(x["missed"], [0, 9, 19, 39, 81, 999], labels=["control", "10-19", "20-39", "40-81", "82+"])
    print("\nALL INJURIES by games missed, vs controls (adjusted for pre level + age)")
    for L, sub in x.groupby("length", observed=True):
        if L == "control":
            continue
        c = x[ctrl]
        b = sub["back"].mean() - c["back"].mean()
        r = np.average(sub["rust_r"].dropna(), weights=sub.loc[sub["rust_r"].notna(), "poss_rust"]) - \
            np.average(c["rust_r"].dropna(), weights=c.loc[c["rust_r"].notna(), "poss_rust"])
        p = np.average(sub["post_r"].dropna(), weights=sub.loc[sub["post_r"].notna(), "poss_post"]) - \
            np.average(c["post_r"].dropna(), weights=c.loc[c["post_r"].notna(), "poss_post"])
        print(f"  out {L:>5} games  n={len(sub):>4}   back {b * 100:+5.1f} pts   first {RUST} {r:+.2f}   "
              f"first {POST} {p:+.2f}")

    print(f"\nBACK ({BACK_GAMES}+ games within 2 years of the injury), pct points vs controls")
    print(f"  types really differ by about +-{back.attrs['tau']:.0%}")
    print(back.assign(**{c: back[c] * 100 for c in ("raw", "lo", "hi", "shrunk")})
          [["group", "n", "raw", "lo", "hi", "shrunk"]].round(1).to_string(index=False))
    for t, lab in ((rust, f"FIRST {RUST} GAMES BACK"), (post, f"FIRST {POST} GAMES BACK")):
        print(f"\n{lab}: RAPM change vs pre window, relative to controls (per 100)")
        print(f"  types really differ by about +-{t.attrs['tau']:.2f}")
        print(t[["group", "n", "raw", "lo", "hi", "shrunk"]].round(2).to_string(index=False))

    out = back.merge(rust, on="group", how="outer", suffixes=("_back", "_rust")).merge(
        post.rename(columns={c: f"{c}_post" for c in post.columns if c != "group"}), on="group", how="outer")
    out.to_parquet(PROCESSED_DIR / "injury_windows.parquet", index=False)
    plot(back, rust, post)
