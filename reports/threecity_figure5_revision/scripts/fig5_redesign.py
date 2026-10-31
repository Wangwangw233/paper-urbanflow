r"""Build main Fig. 5 and Supplementary Fig. S8 from frozen products.

The three maps show response anisotropy and principal-axis orientation over
OpenStreetMap reference geometry. Panel (d) compares the road-induced subspace
with rank-matched random subspaces of ker D; panel (e) shows the paired-date
mobility-minus-road contrast at matched capacity. No model is refitted.

Run from this directory:
    python fig5_redesign.py preview
    python fig5_redesign.py B
    python fig5_redesign.py supp
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve()
sys.dont_write_bytecode = True
REPO_ROOT = HERE.parents[3]
REV = REPO_ROOT / 'reports' / 'threecity_figure5_revision'
ROOT = pathlib.Path(os.environ.get('URBANFLOW_ROOT', REPO_ROOT))
FIGS = REPO_ROOT / "paper" / "merged_manuscript" / "figures"
SUPP_DIR = REPO_ROOT / "paper" / "merged_manuscript" / "figures_supp"
for p in (FIGS, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import figstyle as FS                       # noqa: E402
import fig5_threecity as F5                 # noqa: E402  frozen readers
import threecity_stats as TS                # noqa: E402  (+ Chengdu)
import fig5_ablation as A5                  # noqa: E402  (constants only)

OUT = (pathlib.Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
       if "--out" in sys.argv else pathlib.Path(os.environ.get("URBANFLOW_FIGURE_OUT", REV / "figures")))
CITY3 = F5.CITY3
COL = F5.COL
NAME = F5.NAME
LONG = {"Porto": "Porto", "SF": "San Francisco", "Chengdu": "Chengdu"}

# ---- geometry, in inches ---------------------------------------------------
W, H = 7.00, 5.28
MAPH = 2.60
GAPM = 0.12
Y_MAP = 2.14
Y_CB, CBH = 1.905, 0.050
Y_PC, PH = 0.60, 0.98
XLBL = 0.72
WD = 2.68
XE = XLBL + WD + 0.42
WE = W - XE - 0.06
YLIM3 = (-0.58, 2.74)

# ---- anisotropy ramp -------------------------------------------------------
# Single-hue sequential (purple) chosen so it shares no hue with the three
# city colours: blue (Porto), orange-brown (SF), green (Chengdu).  The dark end
# is deep enough for the segment glyphs in candidate B to stay visible on white.
CMAP = LinearSegmentedColormap.from_list(
    "ac_violet", ["#d9cfe5", "#b09ccb", "#7f63a8", "#3f1176"])
NORM = Normalize(0.0, 1.0)

# ---- basemap ---------------------------------------------------------------
WATER = "#e3edf4"
ROAD_MINOR = "#6f777e"
ROAD_MAJOR = "#3c444b"
ZERO_FACE = "#ffffff"
ZERO_SLASH = "#b4bac0"
NORESP_EDGE = "#a7adb2"

GLYPH_HALF = 0.045                         # inches
GLYPH_PITCH = 0.155                        # inches
GLYPH_AMIN = 0.02                          # same threshold in all cities
KM_BAR = 2.0


# ---------------------------------------------------------------------------
# geometry and basemap
# ---------------------------------------------------------------------------
def block(z, city):
    """The FULL filled lattice, at the city's true aspect ratio."""
    x0, x1, y0, y1, nx, ny, dx, dy, rs, cs = F5.core(z, city)
    A = F5.grid(z, city, "A_C")[rs, cs]
    TH = F5.grid(z, city, "theta_C")[rs, cs]
    return dict(x0=x0, x1=x1, y0=y0, y1=y1, nx=nx, ny=ny, dx=dx, dy=dy,
                A=A, TH=TH, zr=(A == 0.0))


def _affine(b, xlim, ylim):
    ex = np.asarray(b["extent"], np.float64)
    sx = (xlim[1] - xlim[0]) / (ex[1] - ex[0])
    sy = (ylim[1] - ylim[0]) / (ex[3] - ex[2])
    ox, oy = xlim[0] - ex[0] * sx, ylim[0] - ex[2] * sy
    return lambda p: np.column_stack([p[:, 0] * sx + ox, p[:, 1] * sy + oy])


def basemap_to_axes(z, bm, city):
    """Map projected metres to the full kilometre lattice before axis cropping."""
    full_x = tuple(float(v) for v in z[f"{city}_xlim"])
    full_y = tuple(float(v) for v in z[f"{city}_ylim"])
    to_ax = _affine(bm[city], full_x, full_y)
    ex = np.asarray(bm[city]["extent"], np.float64)
    corners = to_ax(np.array([[ex[0], ex[2]], [ex[1], ex[3]]]))
    np.testing.assert_allclose(corners[:, 0], full_x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(corners[:, 1], full_y, rtol=0, atol=1e-10)
    scale = np.array([(full_x[1] - full_x[0]) / (ex[1] - ex[0]),
                      (full_y[1] - full_y[0]) / (ex[3] - ex[2])])
    np.testing.assert_allclose(scale, [0.001, 0.001], rtol=0, atol=1e-12)
    return to_ax


def _segs(b, tag, to_ax):
    pts = np.asarray(b[f"{tag}_pts"], np.float64)
    if pts.size == 0:
        return []
    st = np.asarray(b[f"{tag}_start"], np.int64)
    ct = np.asarray(b[f"{tag}_count"], np.int64)
    return [to_ax(pts[s:s + c]) for s, c in zip(st, ct) if c > 1]


def draw_roads(ax, b, to_ax, tag, color, lw, alpha, z):
    segs = _segs(b, tag, to_ax)
    if segs:
        ax.add_collection(LineCollection(
            segs, colors=color, linewidths=lw, alpha=alpha, zorder=z,
            capstyle="round", antialiased=True))


def draw_water(ax, b, to_ax):
    segs = _segs(b, "water", to_ax)
    if segs:
        ax.add_collection(PolyCollection(segs, closed=True, facecolors=WATER,
                                         edgecolors="none", zorder=0.4))


def glyphs(blk, w_in):
    """Fixed-pitch principal-axis segments; length and colour carry no arrow.

    The threshold and the pitch are identical in the three cities: the pitch is
    set in POINTS OF THE PAGE, so a tall narrow panel is not sampled more
    sparsely than a square one.
    """
    cell_in = w_in / blk["nx"]
    stride = max(1, int(round(GLYPH_PITCH / cell_in)))
    ii, jj = np.meshgrid(np.arange(blk["ny"]), np.arange(blk["nx"]),
                         indexing="ij")
    keep = (ii % stride == 0) & (jj % stride == 0) & (blk["A"] > GLYPH_AMIN)
    cx = blk["x0"] + (jj[keep] + 0.5) * blk["dx"]
    cy = blk["y0"] + (ii[keep] + 0.5) * blk["dy"]
    half = GLYPH_HALF * (blk["x1"] - blk["x0"]) / w_in
    th = blk["TH"][keep]
    u, v = half * np.cos(th), half * np.sin(th)
    seg = np.stack([np.stack([cx - u, cy - v], 1),
                    np.stack([cx + u, cy + v], 1)], 1)
    return seg, blk["A"][keep], stride


def draw_map(ax, z, bm, city, w_in, mode):
    blk = block(z, city)
    to_ax = basemap_to_axes(z, bm, city)
    ext = [blk["x0"], blk["x1"], blk["y0"], blk["y1"]]
    draw_water(ax, bm[city], to_ax)

    if mode == "A":
        f = blk["A"].copy()
        f[blk["zr"]] = np.nan
        ax.imshow(f, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=1,
                  interpolation="nearest", aspect="auto", zorder=1)
        if blk["zr"].any():
            x0, y0, dx, dy = blk["x0"], blk["y0"], blk["dx"], blk["dy"]
            ii, jj = np.nonzero(blk["zr"])
            rects = [[(x0 + j * dx, y0 + i * dy), (x0 + (j + 1) * dx, y0 + i * dy),
                      (x0 + (j + 1) * dx, y0 + (i + 1) * dy), (x0 + j * dx, y0 + (i + 1) * dy)]
                     for i, j in zip(ii, jj)]
            ax.add_collection(PolyCollection(rects, facecolors=ZERO_FACE,
                                             edgecolors="none", zorder=2.0))
            h = 0.30 * min(dx, dy)
            cx, cy = x0 + (jj + 0.5) * dx, y0 + (ii + 0.5) * dy
            ax.add_collection(LineCollection(
                np.stack([np.stack([cx - h, cy - h], 1),
                          np.stack([cx + h, cy + h], 1)], 1),
                colors=ZERO_SLASH, linewidths=0.4, zorder=2.05, capstyle="butt"))
        # only the major-road skeleton is laid over the field; the minor
        # network would re-create the texture that made the old maps noisy
        draw_roads(ax, bm[city], to_ax, "major", ROAD_MAJOR, 0.40, 0.55, 2.6)
        seg, val, stride = glyphs(blk, w_in)
        lc = LineCollection(seg, colors=FS.INK, linewidths=0.5, alpha=0.95,
                            zorder=4, capstyle="round")
        lc.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white",
                                           alpha=0.55)])
        ax.add_collection(lc)
        noresp = int(blk["zr"].sum())
    else:
        draw_roads(ax, bm[city], to_ax, "minor", ROAD_MINOR, 0.18, 0.30, 1.0)
        draw_roads(ax, bm[city], to_ax, "major", ROAD_MAJOR, 0.32, 0.85, 1.3)
        seg, val, stride = glyphs(blk, w_in)
        lc = LineCollection(seg, colors=CMAP(NORM(val)), linewidths=1.05,
                            zorder=3.0, capstyle="round")
        lc.set_path_effects([pe.withStroke(linewidth=1.9, foreground="white",
                                           alpha=0.60)])
        ax.add_collection(lc)
        if blk["zr"].any():
            ii, jj = np.nonzero(blk["zr"])
            ax.scatter(blk["x0"] + (jj + 0.5) * blk["dx"],
                       blk["y0"] + (ii + 0.5) * blk["dy"],
                       marker="s", s=2.8, facecolors="none",
                       edgecolors=NORESP_EDGE, linewidths=0.50, zorder=2.5)
        noresp = int(blk["zr"].sum())

    ax.set_xlim(blk["x0"], blk["x1"])
    ax.set_ylim(blk["y0"], blk["y1"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#5a6167")
        sp.set_linewidth(0.55)
    return stride, int(seg.shape[0]), noresp


def tag(ax, letter):
    ax.annotate("(%s)" % letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(3.0, -2.0), textcoords="offset points", fontsize=7.8,
                fontweight="bold", va="top", ha="left", color=FS.FG, zorder=20,
                annotation_clip=False,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def km_bar(ax, blk):
    x0 = blk["x0"] + 0.06 * (blk["x1"] - blk["x0"])
    y0 = blk["y0"] + 0.06 * (blk["y1"] - blk["y0"])
    ax.plot([x0, x0 + KM_BAR], [y0, y0], color=FS.FG, lw=1.1,
            solid_capstyle="butt", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.text(x0 + KM_BAR / 2, y0 + 0.035 * (blk["y1"] - blk["y0"]),
            f"{KM_BAR:g} km", ha="center", va="bottom", fontsize=6.5,
            color=FS.FG, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


# ---------------------------------------------------------------------------
# bottom row
# ---------------------------------------------------------------------------
def panel_d(ax, nl):
    """dE = E_null - E_road on one axis: the gap is the effect size."""
    ax.axvline(0.0, color="#4f555b", lw=0.75, ls=(0, (3.0, 2.2)), zorder=1)
    for k, city in enumerate(CITY3):
        i = 2 - k                       # Porto on top, as in the map row
        d = nl[city]["de"]
        med = float(np.median(d))
        q01, q99 = np.percentile(d, [1, 99])
        ax.plot([med, 0.0], [i, i], color="#8d949a", lw=0.5,
                ls=(0, (1.0, 1.8)), zorder=1.5)
        ax.plot([d.min(), d.max()], [i, i], color="#c9ced3", lw=0.8,
                solid_capstyle="round", zorder=2)
        ax.plot([q01, q99], [i, i], color="#aeb5bb", lw=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([med], [i], "o", ms=3.4, color=COL[city], mec="white",
                mew=0.6, zorder=5)
        ax.plot([0.0], [i], "o", ms=3.0, mfc="white", mec="#3f464c", mew=0.75,
                zorder=6)
    ax.set_ylim(*YLIM3)
    ax.set_xlim(-0.018, 0.345)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([LONG[c] for c in CITY3], fontsize=6.6)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3"], fontsize=6.6)
    ax.set_xlabel("Excess unexplained energy,  $\\Delta E$",
                  fontsize=7.0, labelpad=2.0)
    ax.text(0.004, 2.53, "real road ($\\Delta E=0$)", fontsize=6.5,
            color="#3f464c", ha="left", va="center")
    FS.light_axes(ax, grid=False)


def panel_e(ax, sel):
    """One contrast only: mobility-inferred minus road at matched capacity."""
    ax.axvline(0.0, color="#4f555b", lw=0.75, ls=(0, (3.0, 2.2)), zorder=1)
    for k, city in enumerate(CITY3):
        i = 2 - k
        v = sel[city]
        lo, hi = v["gap_ci"]
        ax.plot([lo, hi], [i, i], color=COL[city], lw=1.5,
                solid_capstyle="butt", zorder=3)
        for g in (lo, hi):
            ax.plot([g, g], [i - 0.075, i + 0.075], color=COL[city], lw=0.8,
                    zorder=3)
        ax.plot([v["gap"]], [i], "o", ms=3.4, color=COL[city], mec="white",
                mew=0.6, zorder=5)
        ax.text(hi + 0.004, i, "%+.3f" % v["gap"], fontsize=6.6,
                color="#3a3f44", va="center", ha="left")
    ax.set_ylim(*YLIM3)
    ax.set_xlim(-0.012, 0.136)
    # row labels are printed once, in (d); both axes share the same row order
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_xticks([0.0, 0.05, 0.10])
    ax.set_xticklabels(["0", "0.05", "0.10"], fontsize=6.6)
    ax.set_xlabel("Gain in conditional $R^2$:  mobility-inferred $-$ road",
                  fontsize=7.0, labelpad=2.0)
    FS.light_axes(ax, grid=False)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def build(mode):
    z = F5.load()
    F5.checks(z)                              # frozen anchors, or raise
    nl, sel = TS.nulls3(), TS.sweep3()
    bm = F5.basemaps()

    blk = {c: block(z, c) for c in CITY3}
    asp = {c: (blk[c]["x1"] - blk[c]["x0"]) / (blk[c]["y1"] - blk[c]["y0"])
           for c in CITY3}
    wmap = {c: MAPH * asp[c] for c in CITY3}
    total = sum(wmap.values()) + 2 * GAPM
    assert total <= W - 0.05, total
    xp = (W - total) / 2.0
    xs = {"Porto": xp, "SF": xp + wmap["Porto"] + GAPM,
          "Chengdu": xp + wmap["Porto"] + wmap["SF"] + 2 * GAPM}

    fig = plt.figure(figsize=(W, H))
    aa = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa

    info = {}
    for city in CITY3:
        ax = aa(xs[city], Y_MAP, wmap[city], MAPH)
        stride, ng, noresp = draw_map(ax, z, bm, city, wmap[city], mode)
        km_bar(ax, blk[city])
        tag(ax, "abc"[CITY3.index(city)])
        fig.text((xs[city] + wmap[city] / 2) / W, (Y_MAP + MAPH + 0.035) / H,
                 NAME[city], ha="center", va="bottom", fontsize=7.2,
                 fontweight="bold", color=COL[city])
        info[city] = dict(width_in=wmap[city], aspect=asp[city],
                          stride=stride, glyphs=ng, no_response=noresp,
                          cells=int(blk[city]["A"].size))
        print("%-8s %.2f x %.2f in  stride %d  %d segments  %d no-response"
              % (city, wmap[city], MAPH, stride, ng, noresp))

    # colourbar + the two encodings that are not a colour value
    CBX, CBW = 1.34, 1.95
    cax = aa(CBX, Y_CB, CBW, CBH)
    cax.imshow(np.linspace(0, 1, 256)[None, :], origin="lower", aspect="auto",
               cmap=CMAP, extent=[0, 1, 0, 1], vmin=0, vmax=1)
    cax.set_yticks([])
    cax.set_xticks([0.0, 0.5, 1.0])
    cax.set_xticklabels(["0", "0.5", "1"], fontsize=6.5)
    cax.tick_params(axis="x", colors=FS.AXCOL, length=2.0, pad=1.0)
    for sp in cax.spines.values():
        sp.set_color("#9aa4ad")
        sp.set_linewidth(0.5)
    fig.text(CBX / W, (Y_CB + CBH + 0.045) / H,
             "local response anisotropy  $A_C(c)$", fontsize=6.6,
             color=FS.FG, ha="left", va="bottom")

    lx = CBX + CBW + 0.30
    la = aa(lx, Y_CB - 0.075, 1.55, CBH + 0.15)
    la.set_xlim(0, 1)
    la.set_ylim(0, 1)
    la.axis("off")
    la.plot([0.02, 0.10], [0.78, 0.78], color=FS.INK, lw=1.0,
            solid_capstyle="round")
    la.text(0.14, 0.78, "principal axis\n(orientation only)", fontsize=6.5,
            va="center", color="#3a3f44", linespacing=1.05)
    la.plot([0.06], [0.22], marker="s", ms=2.5, mfc="none",
            mec=NORESP_EDGE, mew=0.6)
    la.text(0.14, 0.22, "zero anisotropy", fontsize=6.5, va="center",
            color="#3a3f44")

    # ---- bottom row -------------------------------------------------------
    axd = aa(XLBL, Y_PC, WD, PH)
    panel_d(axd, nl)
    axd.annotate("(d)", xy=(0.0, 1.02), xycoords="axes fraction",
                 xytext=(-16.0, 4.0), textcoords="offset points",
                 fontsize=7.8, fontweight="bold", va="bottom", ha="left",
                 color=FS.FG, annotation_clip=False)
    axe = aa(XE, Y_PC, WE, PH)
    panel_e(axe, sel)
    axe.annotate("(e)", xy=(0.0, 1.02), xycoords="axes fraction",
                 xytext=(-16.0, 4.0), textcoords="offset points",
                 fontsize=7.8, fontweight="bold", va="bottom", ha="left",
                 color=FS.FG, annotation_clip=False)

    for x0, w0, t, sub in (
            (XLBL, WD, "Road specificity",
             "real road vs rank-matched random subspaces"),
            (XE, WE, "Matched-capacity media",
             "paired mobility-inferred minus road contrast")):
        fig.text((x0 + w0 / 2) / W, (Y_PC + PH + 0.095) / H, t,
                 fontsize=7.2, fontweight="bold", color=FS.INK,
                 ha="center", va="bottom")
        fig.text((x0 + w0 / 2) / W, (Y_PC + PH + 0.030) / H, sub,
                 fontsize=6.5, color="#4f555b", ha="center", va="bottom",
                 style="italic")

    info["mode"] = mode
    info["layout_in"] = dict(W=W, H=H, map_h=MAPH, map_gap=GAPM,
                             y_map=Y_MAP, panel_d=[XLBL, Y_PC, WD, PH],
                             panel_e=[XE, Y_PC, WE, PH])
    info["glyph_amin"] = GLYPH_AMIN
    info["glyph_pitch_in"] = GLYPH_PITCH
    info["de_median"] = {c: float(np.median(nl[c]["de"])) for c in CITY3}
    info["gap"] = {c: float(sel[c]["gap"]) for c in CITY3}
    info["gap_ci"] = {c: [float(v) for v in sel[c]["gap_ci"]] for c in CITY3}
    return fig, info


# ---------------------------------------------------------------------------
# Supplementary Fig. S8: the three media behind panel (e)
# ---------------------------------------------------------------------------
def supp_media():
    z = F5.load()
    F5.checks(z)
    sel = TS.sweep3()
    fig = plt.figure(figsize=(3.0, 1.92))
    ax = fig.add_axes([0.32, 0.20, 0.65, 0.60])
    ax.axvline(0.0, color="#4f555b", lw=0.75, ls=(0, (3.0, 2.2)), zorder=1)
    for k, city in enumerate(CITY3):
        i = 2 - k                            # Porto on top, as in Fig. 5
        v = sel[city]
        ax.plot([v["road"], v["mob"]], [i, i], color="#e3e7ea", lw=1.8,
                zorder=1.4, solid_capstyle="round")
        ax.plot([v["shuf"]], [i], "v", ms=4.2, color=A5.SHUF_COLOR, mec="w",
                mew=0.5, zorder=4)
        ax.plot([v["road"]], [i], "s", ms=4.0, color=COL[city], mec="w",
                mew=0.5, zorder=5)
        ax.plot([v["mob"]], [i], "o", ms=4.2, color=COL[city], mec="w",
                mew=0.5, zorder=5)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(-0.075, 0.115)
    ax.set_yticks(range(len(CITY3)))
    ax.set_yticklabels([LONG[c] for c in CITY3], fontsize=6.0)
    ax.set_xticks([-0.05, 0.0, 0.05, 0.10])
    ax.set_xticklabels(["$-0.05$", "0", "0.05", "0.10"], fontsize=5.8)
    ax.set_xlabel("gain in conditional $R^2$ over a constant medium",
                  fontsize=5.9, labelpad=2.0)
    key = [Line2D([], [], ls="none", marker="v", ms=4.0,
                  color=A5.SHUF_COLOR, label="permuted mobility medium"),
           Line2D([], [], ls="none", marker="s", ms=3.8, color="#555555",
                  label="road medium"),
           Line2D([], [], ls="none", marker="o", ms=4.0, color="#555555",
                  label="mobility-inferred")]
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([LONG[c] for c in CITY3], fontsize=6.0)
    fig.legend(handles=key, loc="upper center", bbox_to_anchor=(0.63, 0.985),
               ncol=3, fontsize=4.7, frameon=False, handletextpad=0.35,
               columnspacing=0.9, borderpad=0.05, handlelength=1.2)
    FS.light_axes(ax, grid=False)
    return fig


# ---------------------------------------------------------------------------
def save(fig, name, dpi=600, preview_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    specs = [("png", 150, "_preview")] if preview_only else [
        ("pdf", dpi, ""), ("png", dpi, ""), ("png", 150, "_preview")]
    for ext, d, tag_ in specs:
        p = OUT / ("%s%s.%s" % (name, tag_, ext))
        tmp = OUT / ("%s%s.%s.new" % (name, tag_, ext))
        metadata = None
        if ext == "pdf":
            metadata = {
                "Title": "Road specificity and matched-capacity media" if name.startswith("fig05") else "Three media at matched capacity",
                "Author": "UrbanFlow manuscript authors",
                "Subject": "Urban mobility constitutive response analysis",
                "Creator": "UrbanFlow frozen-product figure pipeline",
            }
        fig.savefig(tmp, dpi=d, format=ext, bbox_inches="tight", pad_inches=0.04,
                    **({"metadata": metadata} if metadata else {}))
        try:
            os.replace(tmp, p)
        except PermissionError:
            print("  %s open elsewhere; left as %s" % (p.name, tmp.name))
            p = tmp
        out.append(p)
    return out


def write_manifest(info):
    ins = [F5.FO4, F5.CD, F5.CD_SUM, A5.TOPO, A5.RM, A5.PW_CSV,
           TS.CD_A, TS.CD_F] + [F5.DATA_REV / "results" / n for n in F5.BM_FILE.values()]
    rec = dict(info)
    rec["inputs_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in ins}
    p = OUT / "fig5_redesign_manifest.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if what == "supp":
        for p in save(supp_media(), "figS08_matched_capacity"):
            print("wrote", p)
        raise SystemExit(0)
    if what == "preview":
        fig, info = build("A")
        save(fig, "fig5_candidate_A", preview_only=True)
        plt.close(fig)
        fig, info = build("B")
        save(fig, "fig5_candidate_B", preview_only=True)
        plt.close(fig)
        print("wrote previews for candidates A and B")
        raise SystemExit(0)
    assert what in ("A", "B"), what
    fig, info = build(what)
    for p in save(fig, "fig04_anisotropy_road_tests"):
        print("wrote", p)
    print("wrote", write_manifest(info))
