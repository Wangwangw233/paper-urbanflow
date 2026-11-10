"""Split the combined anisotropy/medium figure into two readable main figures.

The map panel uses the same frozen arrays and drawing code as
``fig5_redesign.py``.  The structural panel uses its real-data panel data
(``fig4_panels.json``); no distribution is synthesized.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig5_redesign as R  # noqa: E402


def maps():
    z = R.F5.load()
    R.F5.checks(z)
    bm = R.F5.basemaps()
    blk = {c: R.block(z, c) for c in R.CITY3}
    asp = {c: (blk[c]["x1"] - blk[c]["x0"]) / (blk[c]["y1"] - blk[c]["y0"])
           for c in R.CITY3}
    maph = 2.45
    wmap = {c: maph * asp[c] for c in R.CITY3}
    gap = 0.12
    total = sum(wmap.values()) + 2 * gap
    xp = (R.W - total) / 2.0
    xs = {"Porto": xp, "SF": xp + wmap["Porto"] + gap,
          "Chengdu": xp + wmap["Porto"] + wmap["SF"] + 2 * gap}
    fig = R.plt.figure(figsize=(R.W, 3.35))
    aa = lambda x, y, w, h: fig.add_axes([x / R.W, y / 3.35, w / R.W, h / 3.35])  # noqa
    for city in R.CITY3:
        ax = aa(xs[city], 0.58, wmap[city], maph)
        R.draw_map(ax, z, bm, city, wmap[city], "B")
        R.km_bar(ax, blk[city])
        R.tag(ax, "abc"[R.CITY3.index(city)])
        fig.text((xs[city] + wmap[city] / 2) / R.W,
                 (0.58 + maph + 0.04) / 3.35, R.NAME[city],
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                 color=R.COL[city])
    cax = aa(0.94, 0.12, 2.0, 0.05)
    cax.imshow(R.np.linspace(0, 1, 256)[None, :], origin="lower",
               aspect="auto", cmap=R.CMAP, extent=[0, 1, 0, 1],
               vmin=0, vmax=1)
    cax.set_yticks([])
    cax.set_xticks([0.0, 0.5, 1.0])
    cax.set_xticklabels(["0", "0.5", "1"], fontsize=6.5)
    cax.tick_params(axis="x", colors=R.FS.AXCOL, length=2.0, pad=1.0)
    for sp in cax.spines.values():
        sp.set_color("#9aa4ad")
        sp.set_linewidth(0.5)
    fig.text(0.94 / R.W, 0.205 / 3.35, "local response anisotropy  $A_C(c)$",
             fontsize=6.6, color=R.FS.FG, ha="left", va="bottom")
    la = aa(3.25, 0.08, 3.4, 0.15)
    la.set_xlim(0, 1)
    la.set_ylim(0, 1)
    la.axis("off")
    la.plot([0.02, 0.10], [0.78, 0.78], color=R.FS.INK, lw=1.0,
            solid_capstyle="round")
    la.text(0.14, 0.78, "principal axis (orientation only)", fontsize=6.4,
            va="center", color="#3a3f44")
    la.plot([0.06], [0.22], marker="s", ms=2.5, mfc="none",
            mec=R.NORESP_EDGE, mew=0.6)
    la.text(0.14, 0.22, "zero anisotropy", fontsize=6.4, va="center",
            color="#3a3f44")
    return fig


def structural():
    pm = R.load_panels()
    fig = R.plt.figure(figsize=(R.W, 3.15))
    axd = fig.add_axes([0.085, 0.18, 0.405, 0.68])
    axe = fig.add_axes([0.565, 0.18, 0.405, 0.68])
    R.panel_d(axd, pm)
    R.panel_e(axe, pm)
    axd.annotate("(a)", xy=(0.0, 1.02), xycoords="axes fraction",
                 xytext=(-18, 4), textcoords="offset points",
                 fontsize=8, fontweight="bold", va="bottom", ha="left",
                 color=R.FS.FG, annotation_clip=False)
    axe.annotate("(b)", xy=(0.0, 1.02), xycoords="axes fraction",
                 xytext=(-18, 4), textcoords="offset points",
                 fontsize=8, fontweight="bold", va="bottom", ha="left",
                 color=R.FS.FG, annotation_clip=False)
    fig.text(0.2875, 0.935, "Road-medium structural specificity",
             ha="center", fontsize=7.4, fontweight="bold", color=R.FS.INK)
    fig.text(0.7675, 0.935, "Road-prior contribution",
             ha="center", fontsize=7.4, fontweight="bold", color=R.FS.INK)
    fig.text(0.2875, 0.893, "real road vs matched spatial statistics",
             ha="center", fontsize=6.5, color="#4f555b", style="italic")
    fig.text(0.7675, 0.875, "absolute $\\Delta R^2$ over learned-only",
             ha="center", fontsize=6.5, color="#4f555b", style="italic")
    return fig


if __name__ == "__main__":
    for fig, name in ((maps(), "fig04_anisotropy_maps"),
                      (structural(), "fig05_structural_tests")):
        for p in R.save(fig, name):
            print("wrote", p)
        R.plt.close(fig)
