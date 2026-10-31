r"""Shared scaffolding for the merged-manuscript figures (Fig. 3-5 + SI).

ONE style module for the whole figure set, so the new panels cannot drift from
Fig. 2.  `paper/singlecolumn/figures/style.py` is imported rather than copied:
`fig2_composite.py` draws through it, so anything that goes through the same
module is style-identical by construction, whereas a re-implementation would
match only until one of the two was edited.

WHAT THIS MODULE ADDS over `style.py`:

  * a different OUTPUT directory.  `style.OUT` points at the directory the
    module itself lives in; the merged manuscript keeps its figures beside the
    `.tex` files it includes them from, so `S.OUT` is re-pointed here.
  * the frozen-round ROOT, located by marker rather than by counting parents
    (the scripts are copied between editions at different depths -- see
    `_common._locate`'s note).
  * readers for `empirical_transfer_round/results/`, which is where every
    number these figures draw is frozen.

DISCIPLINE (inherited): every drawn number is READ from a frozen product or
recomputed deterministically from frozen matrices -- nothing is re-fitted,
re-tuned, or typed in as a literal.  A panel that cannot reproduce its frozen
value raises instead of drawing.
"""
from __future__ import annotations

import pathlib
import sys
import os

import numpy as np

HERE = pathlib.Path(__file__).resolve()
FIGS = HERE.parent


def _locate(marker: str) -> pathlib.Path:
    """Walk up until `marker` exists, and return that directory.

    Counting parents is not safe: this directory sits at
    `paper/merged_manuscript/figures`, two levels deeper than
    `paper/singlecolumn/figures` sits beside its own root marker.  A wrong root
    does not fail loudly -- it yields a plausible sibling path and raises
    FileNotFoundError much later.
    """
    for base in (FIGS, *FIGS.parents):
        if (base / marker).exists():
            return base
    raise RuntimeError("cannot locate %r above %s" % (marker, FIGS))


REPO_ROOT = FIGS.parents[2]
ROOT = pathlib.Path(os.environ.get("URBANFLOW_ROOT", _locate("empirical_transfer_round")))
ET = ROOT / "empirical_transfer_round"
ET_RES = ET / "results"
ET_ART = ET / "artifacts"

STYLE_DIR = REPO_ROOT / "paper" / "singlecolumn" / "figures"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

import style as S  # noqa: E402

S.apply_theme()
OUT_MAIN = pathlib.Path(os.environ.get("URBANFLOW_FIGURE_OUT", FIGS))
OUT_SUPP = pathlib.Path(os.environ.get("URBANFLOW_SUPP_FIGURE_OUT", FIGS.parent / "figures_supp"))
S.OUT = OUT_MAIN

# palette, re-exported so panels read semantics from one place
MOB, ROAD, CROSS, BASE, NULL = S.MOB, S.ROAD, S.CROSS, S.BASE, S.NULL
INK, TEAL, LGRAY, BLACK, FG, AXCOL = S.INK, S.TEAL, S.LGRAY, S.BLACK, S.FG, S.AXCOL

CITIES = ("Porto", "SF")
CITY_COLOR = {"Porto": MOB, "SF": ROAD}


def save(fig, name, dpi=600):
    """Write vector PDF and high-resolution PNG to the appropriate figure set."""
    out = []
    folder = OUT_SUPP if name.startswith("figS") else OUT_MAIN
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = folder / ("%s.%s" % (name, ext))
        fig.savefig(p, dpi=dpi)
        out.append(p)
    return out


def jload(path):
    return S.jload(path)


def csvrows(path):
    return S.csvrows(path)


def res(name):
    """A frozen result file from the round's `results/` directory."""
    return ET_RES / name


def assert_close(got, want, tol, what):
    if not abs(float(got) - float(want)) <= tol:
        raise AssertionError("%s: got %r, frozen %r (tol %g)"
                             % (what, got, want, tol))


def fmt(x, nd=3):
    return ("%%.%df" % nd) % x


def shade(ax, hset, lo=0.0, hi=1.0, color="#eeeeee", alpha=0.55, zorder=0):
    """Very light vertical band marking the primary harmonics, as in Fig. 2/3/4."""
    for h in hset:
        ax.axvspan(h - 0.5, h + 0.5, color=color, alpha=alpha, lw=0,
                   zorder=zorder)


def light_axes(ax, grid=True):
    S.tidy(ax, grid=grid)
    ax.tick_params(labelsize=6.9)


def km_scalebar(ax, ge, km=1.0):
    S.scale_bar(ax, ge["xedges"], ge["yedges"], km=km)


__all__ = ["S", "ROOT", "ET", "ET_RES", "ET_ART", "FIGS", "MOB", "ROAD", "CROSS",
           "BASE", "NULL", "INK", "TEAL", "LGRAY", "BLACK", "FG", "AXCOL",
           "CITIES", "CITY_COLOR", "save", "jload", "csvrows", "res",
           "assert_close", "fmt", "shade", "light_axes", "km_scalebar"]
