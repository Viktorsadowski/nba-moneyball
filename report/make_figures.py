#!/usr/bin/env python3
"""
Figures that only the report needs (the analysis figures come from src/, figures/).

  report/figures/pipeline.png   how the pieces fit together
  report/figures/market.png     projected WAR vs salary next season, with the market price line

  python report/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROCESSED_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, MUTED, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREY = "#e3eefb", "#fdeadf", "#f0efeb"


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.5, 99)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fill, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge, lw=1))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.6, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=0.9,
                                     shrinkA=0, shrinkB=0))

    w, h = 16.5, 8.5
    # main row: talent
    top = 25
    xs = [0, 20.5, 41, 61.5, 82]
    main = [("Play-by-play", "16 seasons, free\nstats.nba.com mirror"),
            ("Stints", "565k stretches with the\nsame 10 on the floor"),
            ("RAPM", "ridge regression,\noffense + defense"),
            ("Value", "50/30/20 blend, 1000\nghost min, aging curve"),
            ("Wins (WAR)", "calibrated on team\nresults, vs replacement")]
    for x, (t, s) in zip(xs, main):
        box(x, top, w, h, t, s, LIGHT_BLUE, BLUE)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + w + 0.4, top + h / 2, b - 0.4, top + h / 2)

    # middle row: availability
    mid = 13
    box(xs[1], mid, w, h, "Injury data", "PST log 2010-20,\nofficial reports 2021-26", LIGHT_ORANGE, ORANGE)
    box(xs[2], mid, w, h, "Availability", "games missed, injury\nspells, types", LIGHT_ORANGE, ORANGE)
    box(xs[3], mid, w, h, "Injury risk", "expected games next\nseason (linear model)", LIGHT_ORANGE, ORANGE)
    arrow(xs[1] + w + 0.4, mid + h / 2, xs[2] - 0.4, mid + h / 2)
    arrow(xs[2] + w + 0.4, mid + h / 2, xs[3] - 0.4, mid + h / 2)
    arrow(xs[3] + w + 0.4, mid + h / 2 + 1.5, xs[4] + 2, top - 0.4)   # expected minutes into WAR

    # bottom row: money
    bot = 1
    box(xs[2], bot, w, h, "Salaries", "every season + current\ncontracts (bbref)", LIGHT_GREY, MUTED)
    box(xs[3], bot, w, h, "$ per WAR", "what the market pays\nfor a win, per season", LIGHT_GREY, MUTED)
    box(xs[4], bot, w, h, "Surplus", "wins x price - salary,\nover the contract", LIGHT_GREY, MUTED)
    arrow(xs[2] + w + 0.4, bot + h / 2, xs[3] - 0.4, bot + h / 2)
    arrow(xs[3] + w + 0.4, bot + h / 2, xs[4] - 0.4, bot + h / 2)
    arrow(xs[4] + w / 2, top - 0.4, xs[4] + w / 2, bot + h + 0.4)          # WAR down into surplus
    # side analyses, as a note
    ax.text(0.5, mid + h / 2, "Side questions on\nthe same data:\nMVP votes vs RAPM,\n3-pt shooters' pay,\ndefense in playoffs",
            fontsize=6.8, color=MUTED, va="center", linespacing=1.3)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "pipeline.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def market() -> None:
    s = pd.read_parquet(PROCESSED_DIR / "surplus.parquet")
    s = s[s["salary_next"].notna() & s["war_next"].notna()].copy()
    ok = s["war_next"] > 0.5
    dpw = ((s.loc[ok, "surplus_next"] + s.loc[ok, "salary_next"]) / s.loc[ok, "war_next"]).median()
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x, y = s["war_next"], s["salary_next"] / 1e6
    ax.scatter(x, y, s=12, color=MUTED, alpha=0.35, lw=0)
    xx = np.linspace(0, x.max() + 0.5, 50)
    ax.plot(xx, xx * dpw / 1e6, color=INK, lw=1)
    y_top = y.max() + 5
    ax.text(0.9 * y_top / (dpw / 1e6) + 0.25, 0.9 * y_top, f"market price: ${dpw / 1e6:.1f}M per win.\nbelow the line = underpaid",
            ha="left", va="center", fontsize=7.5, color=INK)
    under = s.nlargest(9, "surplus_next")
    over = s.nsmallest(9, "surplus_next")
    ax.scatter(under["war_next"], under["salary_next"] / 1e6, s=26, color=BLUE, zorder=3, label="most underpaid")
    ax.scatter(over["war_next"], over["salary_next"] / 1e6, s=26, color=ORANGE, zorder=3, label="most overpaid")
    placed = []
    for r in pd.concat([under, over]).sort_values("salary_next", ascending=False).itertuples():
        px, py = r.war_next, r.salary_next / 1e6
        for dx, dy in ((0.15, 0.6), (0.15, -1.9), (-1.9, 0.6), (-1.9, -1.9), (0.15, 2.4), (0.15, -3.6)):
            tx, ty = px + dx, py + dy
            if all(abs(tx - a) > 1.7 or abs(ty - b) > 1.7 for a, b in placed):
                break
        placed.append((tx, ty))
        ax.text(tx, ty, r.name, fontsize=6.8, color=INK)
    ax.set_xlabel("projected WAR 2026-27 (injury-adjusted)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("salary 2026-27, $M", color=MUTED, fontsize=8.5)
    ax.set_xlim(-0.3, x.max() + 1.6)
    ax.set_ylim(0, y_top)
    ax.grid(color=GRID, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.savefig(OUT / "market.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    market()
    print(f"figures -> {OUT}")
