#!/usr/bin/env python3
"""
Step 9b: are some injury TYPES worse for a player's future than others?

Every player-season with 10+ games lost to injury gets a type from its main injury note
(Achilles, ACL, other knee, foot, ankle, back/core, hip, leg muscles, upper body, head, illness).
Compared with a control group of player-seasons with under 5 injury games, on two things:

  back on the floor   share who play 500+ minutes the next season. an injury can end careers
                      without ever showing up in RAPM, so this one comes first
  quality             next season RAPM (and the season after) vs our age-adjusted projection, for the
                      ones who did come back. residualised on value level and age, like in injury.py

Small groups (40-60 Achilles/ACL cases) are noisy, so every type's effect also gets shrunk towards
the average type effect (empirical Bayes: the noisier the estimate, the harder the pull). The
shrunk number is the one to believe, the raw one shows what the data alone says.

Output
  data/processed/injury_types.parquet
  figures/injury_types.png

  python src/injury_types.py
"""

import re

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ROOT, SEASONS
from injury import load_panel

MIN_GAMES = 10
CONTROL_MAX = 5
ESTABLISHED = 500
RNG = np.random.default_rng(11)

# first match wins, so the specific ones go first (an "ACL tear in left knee" is ACL, not knee)
TYPES = [
    ("Achilles", r"achilles"),
    ("ACL", r"acl|cruciate"),
    ("Knee, other", r"knee|menisc|patell|mcl"),
    ("Foot", r"foot|toe|plantar|metatars|heel|navicular"),
    ("Ankle", r"ankle"),
    ("Back / core", r"back|lumbar|spin|disc|neck|oblique|abdom|hernia|core"),
    ("Hip", r"hip|pelvi|glute"),
    ("Leg muscles", r"hamstring|groin|calf|adductor|quad|thigh|shin|tibia|fibula|leg"),
    ("Upper body", r"hand|wrist|finger|thumb|shoulder|elbow|rotator|tricep|bicep|arm|rib|chest|pector|clavic"),
    ("Head", r"concussion|head|face|facial|nose|eye|orbital|jaw"),
    ("Illness", r"illness|covid|flu|virus|sick|infection|pneumonia"),
]


def injury_type(note: str) -> str:
    # report notes start with "Injury/Illness", which contains "illness". cut that off first, otherwise
    # every report injury that matches nothing above (tailbone, lung, UCL...) lands in Illness
    note = re.sub(r"^\s*injury\s*/\s*illness\s*-?", "", str(note), flags=re.I)
    s = re.sub(r"[^a-z]", "", note.lower())
    for name, pat in TYPES:
        if re.search(pat, s):
            return name
    return "Other"


def aging_fn():
    c = pd.read_parquet(PROCESSED_DIR / "aging_curve.parquet")
    return np.poly1d(np.polyfit(c["age"], c["curve"], 2))


def build(d: pd.DataFrame) -> pd.DataFrame:
    age_fn = aging_fn()
    d = d.copy()
    d.attrs = {}  # load_panel stashes a Series in attrs, pandas chokes on that when concatenating
    by = {s: g.set_index("player_id") for s, g in d.groupby("season")}
    rows = []
    for s in SEASONS[1:-1]:
        cur, prev = by[s], by.get(s - 1)
        if cur["inj_games"].isna().all():
            continue  # 2020-21, no injury data
        n1, n2 = by.get(s + 1), by.get(s + 2)
        x = cur[["inj_games", "inj_note", "age", "val", "min"]].copy()
        # established = real minutes the season before or the injury season itself
        x["min_prev"] = prev["min"].reindex(x.index).fillna(0) if prev is not None else 0
        x = x[(np.maximum(x["min"], x["min_prev"]) >= ESTABLISHED) & x["val"].notna() & x["inj_games"].notna()]
        x["min_next"] = n1["min"].reindex(x.index).fillna(0)
        x["rapm_next"] = n1["rapm"].reindex(x.index)
        if n2 is not None:
            x["min_next2"] = n2["min"].reindex(x.index).fillna(0)
            x["rapm_next2"] = n2["rapm"].reindex(x.index)
        else:
            x["min_next2"], x["rapm_next2"] = 0, np.nan
        x["season"] = s
        # val is already aged to next season, one more year for the season after
        x["val2"] = x["val"] + age_fn(x["age"] + 2) - age_fn(x["age"] + 1)
        rows.append(x.reset_index())
    x = pd.concat(rows, ignore_index=True)
    x["group"] = np.where(x["inj_games"] >= MIN_GAMES, x["inj_note"].map(injury_type),
                          np.where(x["inj_games"] < CONTROL_MAX, "No injury", None))
    return x[x["group"].notna()].copy()


def residualise(x: pd.DataFrame, rapm: str, val: str, mins: str) -> pd.Series:
    # projection error, with the level (shrunk projections) and age taken out, fitted on everyone
    ok = x[rapm].notna() & x[val].notna() & x["age"].notna() & (x[mins] >= ESTABLISHED)
    err = x.loc[ok, rapm] - x.loc[ok, val]
    X = np.column_stack([np.ones(ok.sum()), x.loc[ok, val], x.loc[ok, "age"]])
    w = np.sqrt(x.loc[ok, mins].clip(upper=3000))
    b = np.linalg.lstsq(X * w.to_numpy()[:, None], err.to_numpy() * w.to_numpy(), rcond=None)[0]
    out = pd.Series(np.nan, index=x.index)
    out[ok] = err - X @ b
    return out


def effects(x: pd.DataFrame, col: str, weight: str | None) -> pd.DataFrame:
    """mean of col per type minus the control group, bootstrap se, then empirical-Bayes shrinkage."""
    ctrl = x[x["group"] == "No injury"][col].dropna()
    ctrl_w = x.loc[ctrl.index, weight] if weight else pd.Series(1.0, index=ctrl.index)
    base = np.average(ctrl, weights=ctrl_w)
    rows = []
    for g, sub in x[x["group"] != "No injury"].groupby("group"):
        v = sub[col].dropna()
        if len(v) < 8:
            continue
        w = (sub.loc[v.index, weight] if weight else pd.Series(1.0, index=v.index)).to_numpy()
        vv = v.to_numpy()
        est = np.average(vv, weights=w) - base
        boots = []
        for _ in range(1000):
            i = RNG.integers(0, len(vv), len(vv))
            boots.append(np.average(vv[i], weights=w[i]) - base)
        rows.append(dict(group=g, n=len(v), raw=est, se=np.std(boots),
                         lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5)))
    t = pd.DataFrame(rows)
    # how much do types really differ: spread of the estimates minus what the noise alone explains
    mu = np.average(t["raw"], weights=1 / t["se"] ** 2)
    tau2 = max(0.0, np.var(t["raw"]) - np.mean(t["se"] ** 2))
    t["shrunk"] = mu + (t["raw"] - mu) * tau2 / (tau2 + t["se"] ** 2)
    t.attrs.update(base=base, tau=np.sqrt(tau2), mu=mu)
    return t.sort_values("shrunk")


def plot(ret: pd.DataFrame, q1: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
    # worst for getting back on the floor at the top
    order = ret.sort_values("shrunk")["group"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(surface)
    for ax, t, title, scale, fmt in (
        (axes[0], ret, "Back to 500+ minutes next season, vs no injury", 100, "{:+.0f} pts"),
        (axes[1], q1, "Next season RAPM vs projection, vs no injury (per 100)", 1, "{:+.2f}"),
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
        ax.tick_params(colors=muted, labelsize=8.5)
    axes[0].invert_yaxis()  # shared y, so once is enough
    axes[0].set_xlabel("percentage points", color=muted, fontsize=8.5)
    axes[1].set_xlabel("points per 100 possessions", color=muted, fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=ink)  # top row is empty there
    fig.suptitle("Which injuries hurt a player's future? (10+ games lost, vs seasons without injury)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "injury_types.png", facecolor=surface)
    print(f"\nplot -> {out / 'injury_types.png'}")


if __name__ == "__main__":
    d = load_panel()
    x = build(d)
    print(f"{len(x):,} established player-seasons: {(x['group'] != 'No injury').sum():,} with {MIN_GAMES}+ "
          f"injury games, {(x['group'] == 'No injury').sum():,} controls (<{CONTROL_MAX})")
    print(x["group"].value_counts().to_string())

    # back on the floor, with value level and age taken out: good young players get their minutes back
    # regardless, so without this a type that mostly hits stars would look harmless
    x["returned_raw"] = (x["min_next"] >= ESTABLISHED).astype(float)
    ok = x["val"].notna() & x["age"].notna() & (x["season"] < max(SEASONS))
    X = np.column_stack([np.ones(ok.sum()), x.loc[ok, "val"], x.loc[ok, "age"], x.loc[ok, "age"] ** 2])
    b = np.linalg.lstsq(X, x.loc[ok, "returned_raw"].to_numpy(), rcond=None)[0]
    x["returned"] = np.nan
    x.loc[ok, "returned"] = x.loc[ok, "returned_raw"] - X @ b + x.loc[ok, "returned_raw"].mean()
    ret = effects(x, "returned", None)

    x["err1"] = residualise(x, "rapm_next", "val", "min_next")
    x["err2"] = residualise(x, "rapm_next2", "val2", "min_next2")
    x["w1"] = x["min_next"].clip(upper=3000)
    x["w2"] = x["min_next2"].clip(upper=3000)
    q1 = effects(x, "err1", "w1")
    q2 = effects(x, "err2", "w2")

    base_raw = x.loc[(x["group"] == "No injury") & ok, "returned_raw"].mean()
    print(f"\nBACK ON THE FLOOR (500+ min next season, adjusted for value + age). no-injury group: {base_raw:.0%}")
    print(f"  types really differ by about +-{ret.attrs['tau']:.0%} (after removing noise)")
    print(ret.assign(raw=ret["raw"] * 100, shrunk=ret["shrunk"] * 100, lo=ret["lo"] * 100, hi=ret["hi"] * 100)
          [["group", "n", "raw", "lo", "hi", "shrunk"]].round(1).to_string(index=False))

    for t, lab in ((q1, "NEXT SEASON"), (q2, "TWO SEASONS LATER")):
        print(f"\nQUALITY {lab}: RAPM vs projection, relative to no injury (per 100), players who came back")
        print(f"  types really differ by about +-{t.attrs['tau']:.2f}")
        print(t[["group", "n", "raw", "lo", "hi", "shrunk"]].round(2).to_string(index=False))

    out = q1.merge(ret, on="group", suffixes=("_quality", "_return")).merge(
        q2[["group", "shrunk"]].rename(columns={"shrunk": "shrunk_quality2"}), on="group", how="left")
    out.to_parquet(PROCESSED_DIR / "injury_types.parquet", index=False)
    plot(ret, q1)
