r"""Figure 5, three-city revision -- anisotropy -> specificity -> insufficiency.

FIVE PANELS (2026-10-30 revision of the six-panel version).  The figure now
carries one argument, in the order it is read:

    (a) (b) (c)  local response anisotropy A_C(c) of the identified operator,
                 Porto / San Francisco / Chengdu, with sparse principal-axis
                 segments over the street base.
    (d)          ROAD SPECIFICITY.  The effect of the real road arrangement
                 against 999 rank-matched random arrangements of the same
                 space: dE = E_null - E_road, one distribution per city, with
                 the real road basis at zero.  This is a test of the
                 ARRANGEMENT (does its organisation carry routing structure?),
                 not of whether roads predict the response.
    (e)          CONSTITUTIVE RESPONSE.  Destroyed arrangement / road medium /
                 mobility-inferred medium at matched capacity, against a
                 constant reference; the right-hand axis carries the only
                 interval the archive supports, the paired-date 95% interval
                 on mobility - road.  This is a test of SUFFICIENCY.

(d) and (e) are complementary, not contradictory: the road arrangement is not
interchangeable, and it is not the whole response either.  The anisotropy
distribution (the old panel d) is now Supplementary Fig. S7 -- it duplicated
the maps, and it is a description, not one of the two tests.

WHAT CHANGED FROM `nc_fig5_anisotropy`.  That figure drew two maps, the
anisotropy distribution and a singular-value spectrum.

CITY SCOPE.  All five panels carry all three cities.  (d) and (e) originally did
not: Chengdu had no same-standard rank-matched null test and no equal-capacity
medium ablation.  Both were computed for this revision with the SAME estimators
(`cd_partA_null.py`: 999 seeds, identical pool and null construction, with
Porto re-run in the same job as the control; `cd_partF_sweep.py`: the frozen
sweep body, with Porto and San Francisco reproduced cell-for-cell against the
frozen CSVs before Chengdu ran).  What still differs for Chengdu is stated in
`threecity_stats.py` and in the README: one mobility channel (`default`, no
occupancy-defined variant) and 4 held-out dates, so its paired interval
resamples 4 dates.

THE MAPS CARRY A REAL BASEMAP OVER FULL-STRENGTH CELLS.  The paper's own
Figure 3 draws its fields over a pale street map.  The previous version put the
streets UNDER the cells at 58% cell opacity, which muted the colour scale and
let the streets compete with the response axes.  `scripts/
cd_basemap_export.py` exports, per city and clipped to the same 250 m study
window, the OSM street network from `project6/data/gis/<city>/processed/
roads.gpkg` (major roads and the minor network drawn separately, simplified at
12 m) and water polygons where the archive has them.  The cells are now drawn
AT FULL STRENGTH and the network is drawn OVER them as a faint texture, so the
colour carries the anisotropy and the streets only locate it.

ONE GROUND EXTENT FOR ALL THREE MAPS.  The anisotropy is defined on the
INTERIOR of each lattice -- the outermost row and column are not in `cells`
(Porto 30x30 of 32x32, SF 30x86 of 32x88, Chengdu 58x58 of 60x60).  The panels
now draw the same 7.5 x 7.5 km block, taken at the CENTRE of the filled
lattice: for Porto that is the whole lattice, for San Francisco 30 of its 86
rows, for Chengdu 30 of its 58 rows and columns.  A fixed rule (the centre, not
the busiest or the prettiest block) keeps the three maps at one scale and one
cell size, so they can be compared directly; the locator inset in the two
cropped panels says which part of the study area is shown.  The statistics in
(d)-(e) and the no-response counts are computed on the FULL lattice, not on the
drawn block.

Because the glyph is now a uniform-length orientation mark (length no longer
encodes A_C -- the colour does), glyph spacing and length are fixed in POINTS
OF THE PAGE, so every panel is sampled at the same visual density.

NOTHING IS FITTED AND NO NUMBER IS TYPED IN.  Porto/SF come from
`full_operator_spatial_anisotropy/results/fo4_fig5_data.npz`; Chengdu from
`chengdu_anisotropy.npz` (frame D, the primary frame; frame P is carried in
the file and asserted against the summary as a sensitivity check).  The two
ablation readers are imported from `paper/merged_manuscript/figures/
fig5_ablation.py`, which re-asserts every frozen anchor before returning, so a
moved number raises instead of being drawn.  Importing it is read-only: this
module never writes into `paper/`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve()
sys.dont_write_bytecode = True          # read `paper/` without caching into it
REPO_ROOT = HERE.parents[3]
REV = REPO_ROOT / 'reports' / 'threecity_figure5_revision'                     # reports/threecity_figure5_revision
ROOT = pathlib.Path(os.environ.get('URBANFLOW_ROOT', REPO_ROOT))                    # the frozen-product root
OUT = pathlib.Path(os.environ.get("URBANFLOW_FIGURE_OUT", REV / "figures"))
FIGS = REPO_ROOT / "paper" / "merged_manuscript" / "figures"
for p in (FIGS,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE.parent))       # threecity_stats.py lives beside us

import figstyle as FS                     # noqa: E402  (theme + helpers)
import fig5_ablation as A5                # noqa: E402  (frozen readers only)
import threecity_stats as TS              # noqa: E402  (+ Chengdu, same readers)

S = FS.S

FO4 = ROOT / "full_operator_spatial_anisotropy" / "results" / "fo4_fig5_data.npz"
DATA_REV = ROOT / "reports" / "threecity_figure5_revision"
CD = DATA_REV / "results" / "chengdu_anisotropy.npz"
CD_SUM = DATA_REV / "results" / "chengdu_anisotropy_summary.json"
BM_FILE = {"Porto": "basemap_Porto.npz", "SF": "basemap_SF.npz",
           "Chengdu": "basemap_Chengdu.npz"}

# ---- palette: the Nature-Cities three-city palette -------------------------
# The nc_* series colours Porto/SF/Chengdu as blue/orange/green throughout
# (`redraw_urbanflow.py`); Figure 5 keeps that so the three cities read the
# same across the figure set.  `FS.CITY_COLOR` is the two-city manuscript
# palette and is not used here.
COL = {"Porto": "#315f85", "SF": "#a76539", "Chengdu": "#49735b"}
CITY3 = ("Porto", "SF", "Chengdu")
NAME = {"Porto": "Porto", "SF": "San Francisco", "Chengdu": "Chengdu"}
#  The row labels of (e)/(f) use the short form: "San Francisco" is wider than
#  the gap between the panels and was printing inside (d).
SHORT = {"Porto": "Porto", "SF": "SF", "Chengdu": "Chengdu"}

# ---- layout, in inches -----------------------------------------------------
W, H = 7.00, 5.05
MH = 2.12                                 # all three windows are 7.5 km squares
GAPM = 0.18                               # between maps
Y_R1 = 2.66                               # bottom of the map block
Y_MAPTOP = Y_R1 + MH
CB_GAP, CB_H = 0.09, 0.05
Y_CB = Y_R1 - CB_GAP - CB_H
Y_PC, PH = 0.38, 1.62                     # bottom row (three cities tall)
YLIM3 = (-0.62, 3.25)                     # shared by (d) and (e), three rows
#  The gap is sized for the ROW LABELS of (d)/(e): "Chengdu" at 6 pt is
#  ~0.30 in, and in the two-city version it was printing on top of panel (d).
PC_XL, PC_GAP = 0.62, 0.42
PC_W = {"d": 2.78, "e": 3.13}
XD = PC_XL
XE = XD + PC_W["d"] + PC_GAP
#  (e) is two axes: the three media on the common gain axis, and the paired
#  mobility-road interval on its own axis to the right.  The interval is NOT a
#  second scale of the same quantity -- it is a different quantity -- so the
#  two axes are side by side, each with its own label, never on one axis.
E_WL, E_MID = 1.82, 0.13
E_WR = PC_W["e"] - E_WL - E_MID

CMAP = LinearSegmentedColormap.from_list(
    "ac_seq", plt.get_cmap("Blues")(np.linspace(0.16, 1.0, 256)))
CMAP.set_bad((1.0, 1.0, 1.0, 0.0))        # outside the lattice: basemap only

# ---- basemap and cell strength --------------------------------------------
#  Data first, streets second.  The cells are opaque so the colour scale is
#  read at its own contrast; the network is drawn over them as a faint
#  mid-grey texture, which locates the response without competing with it.
ALPHA = 1.00                              # data-cell opacity
WATER_C = "#dcebf5"
ROAD_MINOR_C = "#8f979e"
ROAD_MAJOR_C = "#7c848b"
ROAD_MINOR_A = 0.16                       # the texture's own weight
ROAD_MAJOR_A = 0.30
ZERO_FACE = "#ffffff"
ZERO_SLASH_C = "#b3b9bf"                  # "no directional response" glyph

GLYPH_HALF_IN = 0.042                     # uniform half-length, in inches
GLYPH_PITCH_IN = 0.150                    # glyph spacing, in inches
KM_BAR = 2.0
WIN_KM = 7.5                              # the common drawn window


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    z = dict(np.load(FO4))
    c = dict(np.load(CD))
    #  The frozen `zero_resp` columns test the response trace for an EXACT
    #  zero.  That test misses the cells whose trace is numerically zero but
    #  not bit-zero (S_C ~ 1e-29 in Porto, 1e-24 in SF, 1e-20 in Chengdu); those
    #  carry A_C = 0 as well and were being drawn in the palest blue, i.e. as
    #  "almost perfectly isotropic".  This figure shades every A_C = 0 cell,
    #  which is the same statement the frozen column tries to make -- and is
    #  the count the Chengdu export reports as `zero_response_cells` (165).
    for city in ("Porto", "SF"):
        z[f"{city}_zero_exact"] = np.asarray(z[f"{city}_zero_resp"], bool)
    z.update(c)
    z["Chengdu_zero_exact"] = (np.asarray(c["Chengdu_S_C"], float) == 0.0)
    for city in CITY3:
        z[f"{city}_zero_resp"] = (np.asarray(z[f"{city}_A_C"], float) == 0.0)
    return z


def grid(z, city, key):
    nx, ny = int(z[f"{city}_nx"]), int(z[f"{city}_ny"])
    cells = np.asarray(z[f"{city}_cells"], int)
    out = np.full(nx * ny, np.nan)
    out[cells] = np.asarray(z[f"{city}_{key}"], float)
    return out.reshape(ny, nx)


def window(z, city):
    """The drawn block: WIN_KM x WIN_KM at the CENTRE of the filled lattice.

    One ground extent for all three maps, so they are read at one scale.  Porto
    fills its lattice exactly; San Francisco and Chengdu are cropped to a
    centred 30x30 block of 86 and 58 rows, and the locator inset marks where
    that block sits.  The rule is fixed in advance (the centre), so the block
    is not chosen for how it looks.
    """
    x0, x1, y0, y1, nxw, nyw, dx, dy, rs, cs = core(z, city)
    n = int(round(WIN_KM / dx))
    assert abs(n * dx - WIN_KM) < 1e-9, (city, dx)
    assert n <= nxw and n <= nyw, (city, n, nxw, nyw)
    r0 = rs.start + (nyw - n) // 2
    c0 = cs.start + (nxw - n) // 2
    return (x0 + (c0 - cs.start) * dx,
            x0 + (c0 - cs.start + n) * dx,
            y0 + (r0 - rs.start) * dy,
            y0 + (r0 - rs.start + n) * dy,
            n, n, dx, dy, slice(r0, r0 + n), slice(c0, c0 + n))


def core(z, city):
    """The tight box around the cells that actually exist.

    The anisotropy is undefined on the lattice's outer ring, so the full
    `xlim`/`ylim` box carries one dead cell on each side.  Cropping the axes to
    the filled block keeps the frame honest (the panel shows exactly the cells
    it draws) and lets the street map sit under the data instead of in a border.
    """
    nx, ny = int(z[f"{city}_nx"]), int(z[f"{city}_ny"])
    xlim = np.asarray(z[f"{city}_xlim"], float)
    ylim = np.asarray(z[f"{city}_ylim"], float)
    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny
    f = grid(z, city, "A_C")
    rr = np.where(~np.isnan(f).all(1))[0]
    cc = np.where(~np.isnan(f).all(0))[0]
    assert np.array_equal(rr, np.arange(rr[0], rr[-1] + 1)), city
    assert np.array_equal(cc, np.arange(cc[0], cc[-1] + 1)), city
    return (xlim[0] + cc[0] * dx, xlim[0] + (cc[-1] + 1) * dx,
            ylim[0] + rr[0] * dy, ylim[0] + (rr[-1] + 1) * dy,
            int(cc.size), int(rr.size), float(dx), float(dy),
            slice(int(rr[0]), int(rr[-1]) + 1), slice(int(cc[0]), int(cc[-1]) + 1))


def glyph_segments(z, city, w_in, A, TH, win):
    """Uniform-length principal-axis segments, sampled on a fixed page pitch.

    Length no longer encodes A_C (the cell colour does), so every segment is
    the same length: the glyph says WHICH WAY, the colour says HOW MUCH.  A
    cell with a vanishing response has no meaningful axis, so the marks are
    restricted to A > 0.02 and subsampled on a stride that keeps the density
    the same in all three panels.
    """
    x0, x1, y0, y1, nxi, nyi, dx, dy, rs, cs = window(z, city)
    A = A[rs, cs]
    TH = TH[rs, cs]
    cell_in = min(w_in / nxi, MH / nyi)
    stride = max(1, int(round(GLYPH_PITCH_IN / cell_in)))
    ii, jj = np.meshgrid(np.arange(nyi), np.arange(nxi), indexing="ij")
    keep = (ii % stride == 0) & (jj % stride == 0) & (A > 0.02)
    cx = x0 + (jj[keep] + 0.5) * dx
    cy = y0 + (ii[keep] + 0.5) * dy
    h = GLYPH_HALF_IN * (x1 - x0) / w_in       # uniform, in data units
    u, v = h * np.cos(TH[keep]), h * np.sin(TH[keep])
    seg = np.stack([np.stack([cx - u, cy - v], 1),
                    np.stack([cx + u, cy + v], 1)], 1)
    return seg, stride


def basemaps():
    out = {}
    for city, name in BM_FILE.items():
        p = DATA_REV / "results" / name
        if not p.exists():
            raise SystemExit("missing %s (run scripts/cd_basemap_export.py on "
                             "host 164 and pull it)" % p)
        out[city] = dict(np.load(p))
    return out


def _pieces(b, tag, to_ax):
    """(pts, start, count) packed polylines -> a list of (k,2) arrays.

    The basemap is exported in the city's PROJECTED METRES; the anisotropy
    arrays carry their window in KILOMETRES (Porto x 527.5-535.5).  Drawing one
    in the other's units puts every street off-canvas, so the window is moved
    onto the axes' own affine frame instead of assuming a scale.
    """
    pts = np.asarray(b[f"{tag}_pts"], np.float64)
    if pts.size == 0:
        return []
    st = np.asarray(b[f"{tag}_start"], np.int64)
    ct = np.asarray(b[f"{tag}_count"], np.int64)
    return [to_ax(pts[s:s + c]) for s, c in zip(st, ct)]


def draw_basemap(ax, b, xlim, ylim):
    """Water under the cells, the street network over them as a texture."""
    ex = np.asarray(b["extent"], np.float64)
    sx = (xlim[1] - xlim[0]) / (ex[1] - ex[0])
    sy = (ylim[1] - ylim[0]) / (ex[3] - ex[2])
    ox, oy = xlim[0] - ex[0] * sx, ylim[0] - ex[2] * sy

    def to_ax(p):
        return np.column_stack([p[:, 0] * sx + ox, p[:, 1] * sy + oy])

    water = _pieces(b, "water", to_ax)
    if water:
        #  above the white no-response fill (1.80), below the streets: cells
        #  without a response over the bay stay recognisable as water
        ax.add_collection(PolyCollection(water, closed=True, facecolors=WATER_C,
                                         edgecolors="none", zorder=1.90))
    #  The network is drawn ABOVE the opaque cells at low alpha: it locates the
    #  response on the street grid without the cells having to be faded.
    for tag, col, lw, alpha in (("minor", ROAD_MINOR_C, 0.22, ROAD_MINOR_A),
                                ("major", ROAD_MAJOR_C, 0.45, ROAD_MAJOR_A)):
        segs = _pieces(b, tag, to_ax)
        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw,
                                         alpha=alpha, zorder=2.60,
                                         capstyle="round", antialiased=True))
    return dict(minor=int(np.asarray(b["minor_n"])),
                major=int(np.asarray(b["major_n"])),
                water=int(np.asarray(b["water_n"])),
                scale=[float(sx), float(sy)])


def _squares(x0, y0, dx, dy, mask):
    """Cell rectangles as polygon vertices, for the no-response fill."""
    ii, jj = np.nonzero(mask)
    out = []
    for i, j in zip(ii, jj):
        xa, ya = x0 + j * dx, y0 + i * dy
        out.append([(xa, ya), (xa + dx, ya), (xa + dx, ya + dy), (xa, ya + dy)])
    return out


def draw_city(ax, z, bm, city, w_in):
    x0, x1, y0, y1, _, _, dx, dy, rs, cs = window(z, city)
    ext = [x0, x1, y0, y1]
    #  The basemap is placed on the FULL lattice window (its own extent), not on
    #  the cropped one: putting an 8 km window on a 7.5 km box would scale every
    #  street by 1.067 and drift against the cells it is meant to underlie.
    full_x = tuple(float(v) for v in z[f"{city}_xlim"])
    full_y = tuple(float(v) for v in z[f"{city}_ylim"])
    nbm = draw_basemap(ax, bm[city], full_x, full_y)
    A = grid(z, city, "A_C")
    TH = grid(z, city, "theta_C")
    #  No-response cells are white in the data layer (not the palest blue), so
    #  "no directional response" cannot be read as "almost isotropic".
    zr = A[rs, cs] == 0.0
    f = A[rs, cs].copy()
    f[zr] = np.nan
    ax.imshow(f, origin="lower", extent=ext, cmap=CMAP, vmin=0.0, vmax=1.0,
              interpolation="nearest", aspect="equal", zorder=1, alpha=ALPHA)
    if zr.any():
        ax.add_collection(PolyCollection(_squares(x0, y0, dx, dy, zr),
                                         facecolors=ZERO_FACE,
                                         edgecolors="none", zorder=1.80))
        # one light diagonal per missing cell: a texture, not a colour, so it
        # cannot be confused with a low value of A_C
        h = 0.34 * min(dx, dy)
        ii, jj = np.nonzero(zr)
        cx, cy = x0 + (jj + 0.5) * dx, y0 + (ii + 0.5) * dy
        ax.add_collection(LineCollection(
            np.stack([np.stack([cx - h, cy - h], 1),
                      np.stack([cx + h, cy + h], 1)], 1),
            colors=ZERO_SLASH_C, linewidths=0.45, zorder=1.85,
            capstyle="butt"))
    #  the streets go over the missing-cell fill as well, so the network stays
    #  continuous where there is no response
    seg, stride = glyph_segments(z, city, w_in, A, TH, (x0, x1, y0, y1))
    lc = LineCollection(seg, colors=FS.INK, linewidths=0.55, alpha=0.95,
                        zorder=4, capstyle="round")
    lc.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white",
                                       alpha=0.45)])
    ax.add_collection(lc)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#9aa4ad")
        sp.set_linewidth(0.5)
    return stride, seg.shape[0], nbm


def locator(fig, ax, z, city):
    """Miniature of the full study lattice with the drawn block outlined.

    Only for the two cities where the drawn block is a sub-window; Porto's
    block IS its lattice, so a locator there would say nothing.  The extent is
    drawn as a MID-grey field and the block as an OUTLINED rectangle with a
    faint fill: a solid saturated patch at this size reads as a stray colour
    block rather than as a location, and the pale grey it used to sit on was
    invisible at print size.
    """
    if city == "Porto":
        return None
    xl = np.asarray(z[f"{city}_xlim"], float)
    yl = np.asarray(z[f"{city}_ylim"], float)
    w0, w1, h0, h1 = window(z, city)[:4]
    asp = (xl[1] - xl[0]) / (yl[1] - yl[0])
    h_in = 0.30
    w_in = h_in * asp
    if w_in > 0.52:
        w_in, h_in = 0.52, 0.52 / asp
    pos = ax.get_position()
    lax = fig.add_axes([pos.x1 - (w_in + 0.05) / W, pos.y0 + 0.05 / H,
                        w_in / W, h_in / H])
    lax.set_facecolor("white")
    lax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#d9dfe4",
                                edgecolor="#7c868e", lw=0.5, zorder=1))
    fx0 = (w0 - xl[0]) / (xl[1] - xl[0])
    fy0 = (h0 - yl[0]) / (yl[1] - yl[0])
    fw = (w1 - w0) / (xl[1] - xl[0])
    fh = (h1 - h0) / (yl[1] - yl[0])
    lax.add_patch(plt.Rectangle((fx0, fy0), fw, fh, facecolor=COL[city],
                                alpha=0.42, edgecolor=COL[city], lw=1.1,
                                zorder=2))
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_xticks([])
    lax.set_yticks([])
    for sp in lax.spines.values():
        sp.set_visible(False)
    return lax


def km_bar(ax, z, city):
    xlo, xhi, ylo, yhi = window(z, city)[:4]
    x0 = xlo + 0.06 * (xhi - xlo)
    y0 = ylo + 0.05 * (yhi - ylo)
    ax.plot([x0, x0 + KM_BAR], [y0, y0], color=FS.FG, lw=1.2,
            solid_capstyle="butt", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.text(x0 + KM_BAR / 2, y0 + 0.045 * (yhi - ylo), f"{KM_BAR:g} km",
            ha="center", va="bottom", fontsize=5.0, color=FS.FG, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


# ---------------------------------------------------------------------------
# shared scalars
# ---------------------------------------------------------------------------
def orientation_similarity(z, city):
    """Anisotropy-weighted agreement between response and structural axes.

    This is the main text's "anisotropy-weighted orientation similarity": the
    mean of cos(2 dtheta) over cells where both tensors are live, weighted by
    A_C * A_U.  cos(2 dtheta) rather than cos(dtheta) because a principal axis
    is unoriented -- theta and theta + pi are the same axis.  The definition is
    calibrated on the published pair (0.79 in Porto and in San Francisco).
    """
    AC = np.asarray(z[f"{city}_A_C"], float)
    AU = np.asarray(z[f"{city}_A_U"], float)
    thC = np.asarray(z[f"{city}_theta_C"], float)
    thU = np.asarray(z[f"{city}_theta_U"], float)
    m = (AC > 0) & (AU > 0)
    w = AC[m] * AU[m]
    return float(np.sum(w * np.cos(2.0 * (thC[m] - thU[m]))) / np.sum(w))


def ecdf(ax, A, color, ls, lw=1.15):
    """Empirical CDF, run to 1 at x = 1.

    A is bounded by 1, so the curve reaches (1, 1) by definition; the earlier
    version stopped at the largest observed A (0.95 in Porto), which read as a
    CDF that had not been normalised.
    """
    x = np.sort(np.asarray(A, float))
    y = np.arange(1, x.size + 1) / x.size
    ax.step(np.concatenate([[0.0], x, [1.0]]),
            np.concatenate([[0.0], y, [1.0]]),
            where="post", color=color, ls=ls, lw=lw, solid_capstyle="butt")


# ---------------------------------------------------------------------------
# (d) specificity -- the road arrangement is not interchangeable
# ---------------------------------------------------------------------------
AMP = A5.AMP


def panel_d(ax, nl):
    """The effect of the real road arrangement, on ONE axis.

    dE = E_null - E_road for 999 rank-matched random arrangements of the same
    space, drawn as a distribution per city with the real road basis at zero.
    The previous version put the road reference in a narrow broken-axis piece
    beside the null distribution, so the two looked like different energies on
    one axis.  Here the DIFFERENCE is the axis, and the empty ground between
    zero and each distribution is the effect it measures.
    """
    ax.axvline(0.0, color="#5c6166", lw=0.8, ls=(0, (3.2, 2.2)), zorder=1)
    lo, hi = -0.030, max(float(nl[c]["de"].max()) for c in CITY3) + 0.024
    gx = np.linspace(lo, hi, 400)
    for i, city in enumerate(CITY3):
        col = COL[city]
        d = nl[city]["de"]
        dens = A5.density(d, gx)
        dens = dens / dens.max()
        ax.fill_between(gx, i - dens * AMP, i + dens * AMP, color=S.NULL,
                        alpha=0.80, lw=0.5, edgecolor="#6f7377", zorder=2)
        ax.plot([d.min(), d.max()], [i, i], "-", color="#8a9096", lw=0.7,
                zorder=3)
        ax.plot([float(np.median(d))], [i], "o", ms=4.2, color=col, mec="w",
                mew=0.7, zorder=5)
    ax.set_ylim(*YLIM3)
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(CITY3)))
    ax.set_yticklabels([SHORT[c] for c in CITY3])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
    ax.set_xlabel(r"excess uncaptured energy  "
                  r"$\Delta E=E_{\rm null}-E_{\rm road}$"
                  "\n(all 999 random arrangements lie to the right)",
                  fontsize=5.6, labelpad=1.8, linespacing=1.5)
    ax.text(0.004, YLIM3[1] - 0.16, "real road\narrangement", fontsize=4.9,
            color="#5c6166", ha="left", va="center", linespacing=1.15)
    FS.light_axes(ax, grid=False)


# ---------------------------------------------------------------------------
# (e) constitutive response -- three media at matched capacity
# ---------------------------------------------------------------------------
def panel_e(axl, axr, sel):
    """Media positions on the left, the paired interval on the right.

    The three media share one axis (gain over a constant medium at matched
    capacity).  Only the mobility - road contrast has a resampling interval in
    the archive, and an interval on a DIFFERENCE belongs on an axis of
    differences, so it gets its own small axis instead of a whisker that would
    be read as an interval on the mobility point alone.
    """
    axl.axvline(0.0, color="#9aa0a6", lw=0.8, ls=(0, (3.2, 2.2)), zorder=1)
    #  the right axis starts at zero: its left spine IS the "no difference"
    #  reference, so no second dashed rule is drawn on it
    allv = [sel[c][k] for c in CITY3 for k in A5.DUMBBELL]
    lo = min(min(allv), -0.02) - 0.012
    hi = max(max(allv), 0.02) + 0.012
    for i, city in enumerate(CITY3):
        col = COL[city]
        v = sel[city]
        # the connecting line is a reading aid, not a result: pale and thin
        axl.plot([v["shuf"], v["mob"]], [i, i], "-", color="#e6eaee", lw=1.3,
                 zorder=1.4, solid_capstyle="round")
        axl.plot([v["shuf"]], [i], marker="v", ls="none", ms=5.0,
                 color=A5.SHUF_COLOR, mec="w", mew=0.6, zorder=5)
        axl.plot([v["road"]], [i], marker="s", ls="none", ms=4.6,
                 color=col, mec="w", mew=0.6, zorder=5)
        axl.plot([v["mob"]], [i], marker="o", ls="none", ms=5.0,
                 color=col, mec="w", mew=0.6, zorder=5)
        g0, g1 = v["gap_ci"]
        axr.plot([g0, g1], [i, i], color=col, lw=1.8, solid_capstyle="butt",
                 zorder=3)
        for g in (g0, g1):
            axr.plot([g, g], [i - 0.09, i + 0.09], color=col, lw=0.9, zorder=3)
        axr.plot([v["gap"]], [i], "o", ms=3.2, mfc="w", mec=col, mew=1.1,
                 zorder=5)
    for ax in (axl, axr):
        ax.set_ylim(*YLIM3)
        FS.light_axes(ax, grid=False)
    axl.set_yticks(range(len(CITY3)))
    axl.set_yticklabels([SHORT[c] for c in CITY3])
    axl.set_xlim(lo, hi)
    axl.set_xticks([-0.05, 0.0, 0.05])
    axl.set_xticklabels(["$-0.05$", "0", "0.05"])
    axl.set_xlabel("gain in conditional $R^2$ over\na constant medium",
                   fontsize=5.6, labelpad=1.8, linespacing=1.5)
    axr.set_xlim(0.0, max(sel[c]["gap_ci"][1] for c in CITY3) + 0.014)
    axr.set_xticks([0.0, 0.05, 0.10])
    axr.set_xticklabels(["0", "0.05", "0.10"])
    axr.set_xlabel("mobility $-$ road\n95% interval", fontsize=5.6,
                   labelpad=1.8, linespacing=1.5)
    axr.tick_params(axis="y", which="both", left=False, labelleft=False)
    key = [plt.Line2D([], [], ls="none", marker=mk, ms=4.8, color=c, mec="w",
                      mew=0.6, label=lab)
           for mk, c, lab in (("v", A5.SHUF_COLOR, "destroyed arrangement"),
                              ("s", FS.FG, "road medium"),
                              ("o", FS.FG, "mobility-inferred"))]
    axl.legend(handles=key, fontsize=4.9, loc="upper left", borderpad=0.16,
               borderaxespad=0.08, handletextpad=0.40, labelspacing=0.16,
               frameon=True, facecolor="white", edgecolor="none",
               framealpha=1.0)


# ---------------------------------------------------------------------------
# Supplementary Fig. S7: the anisotropy distribution, moved out of Fig. 5
# ---------------------------------------------------------------------------
def supp_cdf(z):
    """The same measure on the road support (dashed) and the response (solid).

    Moved here because it describes the maps rather than testing anything.
    Both curves are computed on the SAME cells with the SAME definition,
    A = (l1-l2)/(l1+l2); what differs is the weight on the road-space rows
    (equal for A_U, the identified operator's coefficient energy for A_C).
    """
    fig = plt.figure(figsize=(3.0, 2.05))
    ax = fig.add_axes([0.185, 0.215, 0.775, 0.715])
    for city in CITY3:
        col = COL[city]
        AU = np.asarray(z[f"{city}_A_U"], float)
        AC = np.asarray(z[f"{city}_A_C"], float)
        ecdf(ax, AU, col, (0, (1.7, 1.7)), lw=1.0)
        ecdf(ax, AC, col, "-", lw=1.2)
        for A, filled in ((AU, False), (AC, True)):
            ax.plot([float(np.median(A))], [0.5], "o", ms=3.2, zorder=6,
                    mfc=col if filled else "white", mec=col,
                    mew=0.8 if filled else 0.7)
        ax.text(float(np.median(AU)) + 0.012, 0.44, SHORT[city], color=col,
                fontsize=5.0, ha="left", va="top", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.0,
                                            foreground="white")])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"local anisotropy  $A$", fontsize=6.2, labelpad=1.4)
    ax.set_ylabel("fraction of cells", fontsize=6.2, labelpad=1.4)
    FS.light_axes(ax, grid=False)
    key = [plt.Line2D([], [], color="#777777", lw=1.2,
                      label="response $A_C$"),
           plt.Line2D([], [], color="#777777", lw=1.1, ls=(0, (1.7, 1.7)),
                      label="road support $A_U$")]
    ax.legend(handles=key, loc="lower right", fontsize=4.8, frameon=False,
              handlelength=1.3, labelspacing=0.16, borderpad=0.10,
              handletextpad=0.40, borderaxespad=0.16)
    return fig


# ---------------------------------------------------------------------------
def checks(z):
    """Re-derive every cross-city statistic and assert it against the summary."""
    summ = json.loads(CD_SUM.read_text(encoding="utf-8"))
    D = summ["frames"][summ["primary_frame"]]
    FS.assert_close(float(np.median(z["Chengdu_A_C"])), D["median_A_C"], 1e-9,
                    "Chengdu median A_C")
    FS.assert_close(float(np.median(z["Chengdu_A_U"])), D["median_A_U"], 1e-9,
                    "Chengdu median A_U")
    FS.assert_close(float(np.median(z["Chengdu_A_C_P"])),
                    summ["frames"][summ["sensitivity_frame"]]["median_A_C"],
                    1e-9, "Chengdu median A_C (frame P)")
    n0 = int(np.asarray(z["Chengdu_zero_resp"]).sum())
    assert n0 == D["zero_response_cells"], (n0, D["zero_response_cells"])
    #  the exact-zero columns must be a subset of the shaded mask, and the
    #  cells they miss must really be numerically dead
    for city in CITY3:
        ex = np.asarray(z[f"{city}_zero_exact"], bool)
        zm = np.asarray(z[f"{city}_zero_resp"], bool)
        assert np.all(zm[ex]), city
        print("%-8s no response: %d shaded, %d of them exact zeros"
              % (city, int(zm.sum()), int(ex.sum())))
    for city, want in (("Porto", 0.6348), ("SF", 0.6514)):
        FS.assert_close(float(np.median(z[f"{city}_A_C"])), want, 2e-3,
                        "%s median A_C" % city)
    FS.assert_close(float(np.median(z["Porto_A_U"])), 0.1440, 2e-3,
                    "Porto median A_U")
    FS.assert_close(float(np.median(z["SF_A_U"])), 0.0573, 2e-3,
                    "SF median A_U")
    # every cell whose A_C is zero has no road support either, in all three
    for city in CITY3:
        A = np.asarray(z[f"{city}_A_C"], float)
        U = np.asarray(z[f"{city}_A_U"], float)
        assert np.allclose(A == 0.0, U == 0.0), city
    #  the published orientation similarity is the calibration for this metric:
    #  if the two archived cities stop returning 0.79, the Chengdu number drawn
    #  from the same formula cannot be quoted anywhere.
    sim = {c: orientation_similarity(z, c) for c in CITY3}
    for c in ("Porto", "SF"):
        FS.assert_close(sim[c], 0.79, 2e-2, "%s axis similarity" % c)
    return ({c: (float(np.median(z[f"{c}_A_C"])),
                  float(np.median(z[f"{c}_A_U"]))) for c in CITY3}, sim)


def layout(z):
    """Panel geometry, in inches.  All three windows are 7.5 km squares."""
    w_in = {}
    for c in CITY3:
        x0, x1, y0, y1 = window(z, c)[:4]
        w_in[c] = MH * (x1 - x0) / (y1 - y0)
    group = sum(w_in.values()) + 2 * GAPM
    xp = (W - group) / 2.0
    xs = xp + w_in["Porto"] + GAPM
    xc = xs + w_in["SF"] + GAPM
    return w_in, xp, xs, xc


def main():
    z = load()
    med, sim = checks(z)
    nl, sel = TS.nulls3(), TS.sweep3()
    bm = basemaps()
    W_IN, XP, XS, XC = layout(z)

    fig = plt.figure(figsize=(W, H))
    aa = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa

    # ---- top row: three maps on one ground extent, one colour scale --------
    for city, x, w in (("Porto", XP, W_IN["Porto"]), ("SF", XS, W_IN["SF"]),
                       ("Chengdu", XC, W_IN["Chengdu"])):
        ax = aa(x, Y_R1, w, MH)
        stride, n, nbm = draw_city(ax, z, bm, city, w)
        print("%-8s map %.2f x %.2f in  stride %d  %d glyphs  basemap "
              "%d major / %d minor / %d water"
              % (city, w, MH, stride, n, nbm["major"], nbm["minor"],
                 nbm["water"]))
        km_bar(ax, z, city)
        locator(fig, ax, z, city)
        fig.text((x + w / 2) / W, (Y_MAPTOP + 0.035) / H, NAME[city],
                 ha="center", va="bottom", fontsize=7.4, fontweight="bold",
                 color=COL[city])
        S.panel_tag(ax, "abc"[CITY3.index(city)], pts=(-6.0, 2.0),
                    fontsize=8.4)

    # one colour bar for all three maps, plus the no-response swatch
    CBW = 2.20
    cax = aa((W - CBW) / 2, Y_CB, CBW, CB_H)
    cax.imshow(np.linspace(0, 1, 256)[None, :], origin="lower", aspect="auto",
               cmap=CMAP, extent=[0, 1, 0, 1], vmin=0, vmax=1, alpha=ALPHA)
    cax.set_yticks([])
    cax.set_xticks([0.0, 0.5, 1.0])
    cax.set_xticklabels(["0", "0.5", "1"], fontsize=5.0)
    cax.tick_params(axis="x", colors=FS.AXCOL, length=2.0, pad=1.0)
    for sp in cax.spines.values():
        sp.set_color("#9aa4ad")
        sp.set_linewidth(0.5)
    fig.text((W - CBW) / 2 / W, (Y_CB - 0.205) / H,
             r"local response anisotropy  $A_C(c)$", fontsize=5.8,
             color=FS.FG, ha="left", va="bottom")
    sw = aa((W + CBW) / 2 + 0.16, Y_CB - 0.008, 0.18, CB_H + 0.016)
    sw.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=ZERO_FACE,
                               edgecolor="#9aa4ad", lw=0.5))
    sw.plot([0.14, 0.86], [0.12, 0.88], color=ZERO_SLASH_C, lw=0.8)
    sw.set_xlim(0, 1)
    sw.set_ylim(0, 1)
    sw.set_xticks([])
    sw.set_yticks([])
    for sp in sw.spines.values():
        sp.set_visible(False)
    fig.text(((W + CBW) / 2 + 0.40) / W, (Y_CB + CB_H / 2) / H,
             "no directional\nresponse", fontsize=5.0, color="#5c6166",
             ha="left", va="center", linespacing=1.15)

    # ---- bottom row: the two tests ----------------------------------------
    axd = aa(XD, Y_PC, PC_W["d"], PH)
    panel_d(axd, nl)
    S.panel_tag(axd, "d", pts=(-15.0, 3.0))

    axe1 = aa(XE, Y_PC, E_WL, PH)
    axe2 = aa(XE + E_WL + E_MID, Y_PC, E_WR, PH)
    panel_e(axe1, axe2, sel)
    S.panel_tag(axe1, "e", pts=(-15.0, 3.0))

    for x, w, t, sub in ((XD, PC_W["d"], "Road specificity",
                          "real vs disrupted road organization"),
                         (XE, PC_W["e"], "Constitutive response",
                          "GIS road prior vs mobility-inferred medium")):
        fig.text((x + w / 2) / W, (Y_PC + PH + 0.075) / H, t, fontsize=7.6,
                 fontweight="bold", color=FS.INK, ha="center", va="bottom")
        fig.text((x + w / 2) / W, (Y_PC + PH + 0.012) / H, sub, fontsize=5.6,
                 color="#5c6166", ha="center", va="bottom", style="italic")

    for c in CITY3:
        print("%-8s median A_C %.4f   median A_U %.4f   axis similarity %.3f"
              % ((c,) + med[c] + (sim[c],)))
    return fig


def manifest(z):
    """sha256 of every input, plus the drawn statistics, beside the figure."""
    import hashlib
    ins = [FO4, CD, CD_SUM, A5.TOPO, A5.RM, A5.PW_CSV, TS.CD_A, TS.CD_F] + \
        [DATA_REV / "results" / n for n in BM_FILE.values()]
    rec = {"inputs_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in ins},
           "cell_alpha": ALPHA,
           "axis_similarity": {c: round(orientation_similarity(z, c), 6)
                               for c in CITY3},
           "maps": {}}
    for c in CITY3:
        A = np.asarray(z[f"{c}_A_C"], float)
        U = np.asarray(z[f"{c}_A_U"], float)
        ok = A > 0
        w = window(z, c)
        rec["maps"][c] = dict(
            nx=int(z[f"{c}_nx"]), ny=int(z[f"{c}_ny"]),
            xlim=[float(v) for v in z[f"{c}_xlim"]],
            ylim=[float(v) for v in z[f"{c}_ylim"]],
            drawn_window_km=[float(w[1] - w[0]), float(w[3] - w[2])],
            drawn_window_rows=[int(w[8].start), int(w[8].stop)],
            drawn_window_cols=[int(w[9].start), int(w[9].stop)],
            cells=int(A.size), no_response=int((~ok).sum()),
            no_response_in_window=int(
                (grid(z, c, "A_C")[w[8], w[9]] == 0).sum()),
            median_A_C=float(np.median(A)), median_A_U=float(np.median(U)),
            median_A_C_supported=float(np.median(A[ok])),
            median_A_U_supported=float(np.median(U[ok])),
            fraction_AC_gt_AU=float((A[ok] > U[ok]).mean()))
    p = pathlib.Path(os.environ.get("URBANFLOW_RESULT_OUT", REV / "results")) / "fig5_threecity_manifest.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


def save(fig, name="nc_fig5_anisotropy_threecity", dpi=600):
    """Render beside the module, then swap in -- a PDF held open in a viewer
    cannot be replaced on Windows, and that must not cost a whole re-run."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, d, tag in (("pdf", dpi, ""), ("png", dpi, ""),
                        ("png", 150, "_preview")):
        p = OUT / ("%s%s.%s" % (name, tag, ext))
        tmp = OUT / ("%s%s.%s.new" % (name, tag, ext))
        fig.savefig(tmp, dpi=d, format=ext)
        try:
            os.replace(tmp, p)
        except PermissionError:
            print("  %s is open elsewhere; left this run in %s"
                  % (p.name, tmp.name))
            p = tmp
        out.append(p)
    return out


if __name__ == "__main__":
    if "--supp-only" in sys.argv:
        for p in save(supp_cdf(load()), "figS07_anisotropy_distributions"):
            print("wrote", p)
        raise SystemExit(0)
    fig = main()
    for p in save(fig):
        print("wrote", p)
    fs = supp_cdf(load())
    for p in save(fs, "figS07_anisotropy_distributions"):
        print("wrote", p)
    print("wrote", manifest(load()))
