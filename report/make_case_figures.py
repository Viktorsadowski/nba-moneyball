#!/usr/bin/env python3
"""
Figures for the PHI case (report/build_case.py). The analysis figures (market_value, arbitrage,
playoff_rotation) come from src/.

  report/figures/case_curve.png       what a win is worth to PHI: odds by projected wins
  report/figures/case_two_prices.png  every player: our value vs the market's price, PHI's moves marked
  report/figures/case_picks.png       what a pick turns into vs what trades pay for one
  report/figures/case_scenarios.png   odds per scenario

  python report/make_case_figures.py [scenarios csv tag, default mkt]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROCESSED_DIR  # noqa: E402
from labels import place_labels  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, MUTED, GRID, BLUE, ORANGE, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834", "#b9b7b0"
TAG = sys.argv[1] if len(sys.argv) > 1 else "mkt"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def curve(now_w: float, plan_w: float) -> None:
    c = pd.read_parquet(PROCESSED_DIR / "win_curve_PHI.parquet")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for k, col, lab in (("playoffs", MUTED, "make the playoffs"), ("finals", BLUE, "reach the Finals"),
                        ("title", ORANGE, "win the title")):
        ax.plot(c["wins"], c[k] * 100, color=col, lw=2, label=lab)
    for x, lab in ((now_w, "now"), (52, "target 1"), (60, "target 2"), (plan_w, "the plan")):
        ax.axvline(x, color=GRID if lab.startswith("target") else INK, lw=0.9, ls="--" if lab.startswith("target") else "-")
        # the plan sits close to target 2: whichever of the two is further left gets its label left of the line
        left = "the plan" if plan_w < 60 else "target 2"
        ax.text(x - 0.2 if lab == left else x + 0.2, 103, lab, fontsize=7.5, ha="right" if lab == left else "left",
                color=MUTED if lab.startswith("target") else INK)
    ax.set_xlabel("projected wins", color=MUTED, fontsize=8.5)
    ax.set_ylabel("chance, %", color=MUTED, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, lw=0.6)
    # top left is the only corner no curve runs through
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "case_curve.png", facecolor=SURFACE)
    plt.close(fig)


def two_prices(out_names: list, in_names: list) -> None:
    m = pd.read_parquet(PROCESSED_DIR / "market_value.parquet").dropna(subset=["surplus"])
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x, y = m["market_surplus"] / 1e6, m["surplus"] / 1e6
    ax.scatter(x, y, s=8, color=GREY, alpha=0.6, lw=0)
    lim = [-130, 330]
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    for names, col, lab in ((out_names, ORANGE, "PHI sends"), (in_names, BLUE, "PHI gets")):
        s = m[m["name"].isin(names)]
        ax.scatter(s["market_surplus"] / 1e6, s["surplus"] / 1e6, s=30, color=col, zorder=3, label=lab)
    note = ax.text(150, -110, "above the line: we like him more\nthan the market does", fontsize=7.5, color=MUTED)
    ax.set_xlabel("market value over his contract, $M (what other teams think he's worth minus his salary)",
                  color=MUTED, fontsize=8)
    ax.set_ylabel("our value over his contract, $M", color=MUTED, fontsize=8.5)
    ax.set_xlim(-120, 220)
    ax.set_ylim(-130, 330)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    # names last, when the layout is final. only the ones that matter get a name, the salary filler stays a dot,
    # but the names stay off those dots too, and off the even-value line
    moved = m[m["name"].isin(list(out_names) + list(in_names))]
    named = moved[moved["salary"] >= 7.5e6]
    rest = moved[moved["salary"] < 7.5e6]
    ll = np.linspace(lim[0], lim[1], 400)
    place_labels(ax, named["market_surplus"] / 1e6, named["surplus"] / 1e6, named["name"],
                 avoid_xy=np.column_stack([rest["market_surplus"] / 1e6, rest["surplus"] / 1e6]),
                 avoid_line=np.column_stack([ll, ll]), avoid_artists=[note, leg], marker_pt=3.0)
    fig.savefig(OUT / "case_two_prices.png", facecolor=SURFACE)
    plt.close(fig)


def picks() -> None:
    pk = pd.read_parquet(PROCESSED_DIR / "pick_market_value.parquet")
    tv = json.loads((PROCESSED_DIR / "trade_value.json").read_text())
    cap = 164.961e6
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.bar(pk["pick"], pk["mkt_share_4y_raw"] * cap / 1e6, color=GRID, width=0.8, label="what picks turned into, average by slot")
    ax.plot(pk["pick"], pk["mkt_4y"] / 1e6, color=BLUE, lw=2, label="smoothed")
    lo, hi = tv["ci_first"]
    ax.axhspan(lo / 1e6, hi / 1e6, xmin=0, xmax=0.5, color=ORANGE, alpha=0.12)
    ax.plot([1, 30], [tv["first"] / 1e6] * 2, color=ORANGE, lw=2, label="what trades pay for a future 1st (95% band)")
    ax.plot([31, 60], [tv["second"] / 1e6] * 2, color=ORANGE, lw=2, ls="--", label="what trades pay for a 2nd")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("draft pick", color=MUTED, fontsize=8.5)
    ax.set_ylabel("market value over the rookie deal, $M", color=MUTED, fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "case_picks.png", facecolor=SURFACE)
    plt.close(fig)


def scenarios(sc: pd.DataFrame, now: dict) -> None:
    rows = [("now", now)] + [(r["label"], r) for _, r in sc.iterrows()]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x = np.arange(len(rows))
    for j, (k, col) in enumerate((("playoffs", GREY), ("finals", BLUE), ("title", ORANGE))):
        v = [float(r[k]) * 100 for _, r in rows]
        ax.bar(x + (j - 1) * 0.26, v, width=0.26, color=col, label={"playoffs": "playoffs", "finals": "Finals",
                                                                      "title": "title"}[k])
        for xi, vi in zip(x, v):
            if k != "playoffs":
                ax.text(xi + (j - 1) * 0.26, vi + 1.2, f"{vi:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    # two lines, so the long names can't run into each other when the figure is printed small
    ax.set_xticklabels([n.replace(", ", ",\n") for n, _ in rows], fontsize=7.8, color=INK)
    ax.set_ylabel("chance, %", color=MUTED, fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylim(0, 122)                 # room for the legend above the 100% bars
    ax.set_yticks(range(0, 101, 20))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "case_scenarios.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sc = pd.read_csv(PROCESSED_DIR / f"trade_scenarios_{TAG}.csv")
    # the scenarios in the figure: best under the tax, the best single trade (both aprons), best under the 1st apron
    keep = ["max title, under tax", "one trade, under apron1", "one trade, under apron2", "max title, under apron1"]
    sc = sc[sc["tag"].isin(keep)].set_index("tag").loc[keep].reset_index()
    sc["label"] = ["best, tax line", "one trade, 1st apron", "one trade, 2nd apron", "best, 1st apron"]
    odds = pd.read_parquet(PROCESSED_DIR / "season_odds.parquet").set_index("team").loc["PHI"]
    now = dict(playoffs=odds["playoffs"], finals=odds["finals"], title=odds["title"], wins=odds["wins"])
    # the recommended plan = best under the 1st apron (the 2nd apron finds the same trades)
    best = sc[sc["tag"] == "max title, under apron1"].iloc[0]
    curve(now["wins"], best["wins"])
    plan = best["trades"]
    names = pd.read_parquet(PROCESSED_DIR / "market_value.parquet")["name"].dropna().unique()
    outs, ins = [], []
    for line in plan.split("\n"):
        sent, got = line.split(" for ")
        outs += [n for n in names if n in sent.split(": send ")[1]]
        ins += [n for n in names if n in got]
    two_prices(outs, ins)
    picks()
    scenarios(sc, now)
    print(f"case figures -> {OUT} (plan: {best['tag']})")
