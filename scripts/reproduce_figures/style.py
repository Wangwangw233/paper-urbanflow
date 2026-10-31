"""Shared plotting primitives, pruned to the public figure closure."""

from __future__ import annotations

import csv

import json

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

OUT = HERE

MOB = "#1f4e79"

ROAD = "#c0392b"

CROSS = "#7b3f9e"

FLEX = "#d97706"

BASE = "#4d4d4d"

NULL = "#8a8f94"

INK = "#0d2b45"

TEAL = "#0f766e"

LGRAY = "#d9d9d9"

BLACK = "#1a1a1a"

FG = "#111111"

AXCOL = "#555555"

FONT = "Arial"

SERVER_FONT = "DejaVu Sans"

LW = 1.2

def apply_theme():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, SERVER_FONT],
        "font.size": 7.6,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": AXCOL,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fancybox": False,
        "legend.frameon": False,
        "legend.fontsize": 6.6,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def panel_tag(ax, letter, pts=(-15.0, 2.5), anchor=(0.0, 1.0), fontsize=8.6):
    """Panel letter at a FIXED offset in points from a named corner.

    Axes-fraction offsets put (a) and (d) in visibly different places once the
    panels have different widths; a points offset lines the column up.
    """
    ax.annotate(f"({letter})", xy=anchor, xycoords="axes fraction",
                xytext=pts, textcoords="offset points", fontsize=fontsize,
                fontweight="bold", va="bottom", ha="left", color=FG,
                annotation_clip=False)

def tidy(ax, grid=False, hide=("top", "right")):
    for s in hide:
        if s in ax.spines:
            ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        if s in ax.spines:
            ax.spines[s].set_color(AXCOL)
    ax.tick_params(colors=AXCOL)
    if grid:
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.5, ls=":", zorder=0)
        ax.set_axisbelow(True)

def save(fig, name):
    out = []
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=400)
        out.append(p)
    return out

def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)

def csvrows(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))

def f(x):
    return float(x)

def scale_bar(ax, xedges, yedges, km=2.0, color=FG):
    """A single length scale, so the two mode panels share one ruler."""
    x0 = (xedges[0] + 0.06 * (xedges[-1] - xedges[0])) / 1e3
    y0 = (yedges[0] + 0.06 * (yedges[-1] - yedges[0])) / 1e3
    ax.plot([x0, x0 + km], [y0, y0], color=color, lw=1.4,
            solid_capstyle="butt", zorder=6)
    ax.text(x0 + km / 2, y0, f"{km:g} km", ha="center", va="bottom",
            fontsize=6.2, color=color, zorder=6)

def to_rgba(color):
    from matplotlib.colors import to_rgba as _t
    return _t(color)
