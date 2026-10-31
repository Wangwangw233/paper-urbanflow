r"""Figure 2 draft -- the routing field's spatiotemporal spectrum as a 3D surface.

WHAT IT DRAWS.  One panel per city.  x = temporal harmonic h = 0..12 of the
daily clock, y = spatial mode rank k of that harmonic, z = p_{h,k}, the share of
the city's whole routing power carried by mode k at harmonic h, so that
sum_{h,k} p_{h,k} = 1 over the full mode range.  The three panels share one
viewing angle, one axis range, one z scale, one colour scale and one colorbar;
only the data differ.

WHY THE Z AXIS IS LOGARITHMIC.  The shares span about four and a half decades in
every city, and the largest bin is the h = 0 one at 0.64.  On a linear z axis the
whole clock-dependent spectrum would be a flat sheet one pixel above the floor,
so the figure could not show the decay with harmonic order or the concentration
in the first modes, which are the two things it exists to show.  A log z axis
changes no value; it makes the small ones visible.  The linear rendering is
produced beside this one by `--linear` for comparison, and the diagnostics file
prints the same shares as numbers.

WHY THE BARS AND NOT A SURFACE.  `h` and `k` are discrete ranks, not continuous
variables, and there are cells with no meaning between them.  Bars draw one
object per cell and interpolate nothing; a smooth surface over this grid would
invent shape between the bins.

WHERE THE DATA COMES FROM.  `_f2_build.py` writes `_f2_shares.npz` and is the
only producer; it recomputes the frozen spatial decomposition and asserts it
against `spectral_power.csv` on all 132 frozen rows before writing.  This file
reads that npz and nothing else, and re-checks the normalisation itself.
"""
from __future__ import annotations

import argparse
import io
import pathlib
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402

SHARES = HERE / "_f2_shares.npz"
CITIES = ("Porto", "San Francisco", "Rome")
KEYS = {"Porto": "Porto", "San Francisco": "SF", "Rome": "Rome"}
H_MAX = 12
K_MAX = 9          # the largest mode rank every fold of every city supplies:
                   # San Francisco has nine training dates per fold, so its
                   # harmonic snapshot matrix has nine singular values.  Porto
                   # and Rome have more (151 and 16) and are cut to the same
                   # range so the three panels share one y axis.

# one continuous non-rainbow scale, ramped in the paper's own blue
CMAP = LinearSegmentedColormap.from_list(
    "paper_blue", ["#f4f7fa", "#b9cbdb", "#6f96b6", "#356391", "#1f4e79",
                   "#123a5c", "#0a2032"])

# THE DISPLAY FLOOR IS THE DATA FLOOR, NOT A ROUND NUMBER BELOW IT.  The smallest
# of the 351 drawn shares is 3.70e-4 (Porto), i.e. log10 = -3.43, so a floor at
# -4.6 spent 1.2 decades of axis height on cells that do not exist -- which is
# what made the near-floor plateau read as a heavy slab rather than as a floor.
# A floor of -3.5 sits just under the smallest real cell, so it shortens the
# plateau on screen and clips NOTHING.  Cells at or below it are drawn as
# zero-height tiles, so the below-threshold branch exists and is exercised only
# if a future build pushes a share under -3.5; `main()` asserts the clipping is
# inactive today and prints the margin, so a silent clip cannot creep in.
Z_FLOOR = -3.5
BAR_H = 0.60        # extent along h
BAR_K = 0.40        # extent along k, reduced from 0.60 so that k >= 2 is not
                    # occluded by the k = 1 column in front of it
# Bar edge strokes, as (lead colour, rest colour, lead lw, rest lw) in points.
# Ranks 1 and 2 are stroked harder than the rest so the low-dimensional reading
# the text claims can be taken off the picture; the rest stroke is deliberately
# just above the pale face colours, which is what separates adjacent cells.
EDGE = ("#33536f", "#7d97ad", 0.16, 0.09)
# Negative padding pulls each numeral back towards its own axis line: mplot3d
# parks a 3D tick label well clear of the axis, and in this row that gap was
# wider than the numerals themselves.  Chosen by sweeping -5/-8/-11/-14 and
# looking: at -5 the gap still reads as slack, at -11 the numerals touch the axis
# line and at -14 they sit on it, so -7 is the last value that is tight without
# crossing.  These are POINTS, so they do not scale with the panel -- the
# composite's panels are shorter than the standalone figure's, which makes the
# same padding relatively tighter there, and this value is set for the shorter
# panel.
TICK_PAD_X = -7.0   # the h numerals, under the h axis
TICK_PAD_Y = -6.0   # the k numerals, along the k seam
ELEV = 36.0         # viewing elevation, in degrees; shared by all three panels
AZIM = -58.0        # azimuth, in degrees; shared by all three panels
BOX_ZOOM = 1.06     # magnifies the 3D box inside its axes rectangle.  At zoom 1
                    # the projected box fills about 86 % of its rectangle in
                    # width, and across three panels that slack reads as
                    # inter-panel whitespace.  Measured: at 1.06 the content
                    # reaches x 0.420..2.315 in a rectangle spanning 0.415..2.320,
                    # i.e. it fills the rectangle without spilling into the
                    # neighbouring panel's numerals (at 1.18 the k numerals of one
                    # panel land on the h numerals of the next).  One number,
                    # three panels.

# In-panel city label.  This was abbreviated to "SF" on the assumption that a
# three-across 3D row had no width for the full name; measured, "San Francisco"
# at the panels' 8.4 pt occupies 0.62 in of a 2.35 in panel, so the assumption
# was wrong and the abbreviation was only costing the paper one city spelled two
# ways -- "San Francisco" in a caption and legend, "SF" on the canvas.  The
# figure set now uses the full name on every panel and the abbreviation in every
# legend, which is why this is the one string the whole set reads it from.
SHORT = {"Porto": "Porto", "San Francisco": "San Francisco", "Rome": "Rome"}

# The panel layout is fixed here rather than inline so that the axis key, which
# is placed in figure coordinates, cannot drift away from the panels it explains.
FIG_W, FIG_H = 7.16, 2.38
PANEL_X = (0.058, 0.349, 0.640)
PANEL_W = 0.266
PANEL_Y, PANEL_H = 0.042, 0.916

# The shared axis key: `at` is the key's lower-left in panel (a)'s own rectangle
# (0..1 from that rectangle's lower-left), `box` is its size in INCHES.  Placed
# by measurement, not by eye -- `axis_key` refuses to draw over ink.  It sits in
# panel (a)'s lower-left because that is the one band of the panel the drawing
# never reaches: below the h numerals is page.
KEY_AT = (0.000, 0.008)   # panel (a)'s lower-left corner
# The box is 1.00 in and the triad's origin sits at 0.44 in of it, not at the
# 0.28 in a centred key would want -- see the placement note inside `axis_key`.
KEY_BOX = (1.00, 0.34)
KEY_FS = 4.9
# The name the axis key gives the z axis.  It is a module constant rather than a
# literal inside `axis_key` so that a second spectrum drawn with these same
# conventions can name its own quantity (Figure 3's source excitation spectrum
# is `q_{h,k}`) by setting this attribute, instead of forking the key -- a fork
# would be the same furniture drawn twice and the two copies would drift the
# first time either was touched.  The default is unchanged, so Figure 2's own
# output is byte-identical.
Z_LABEL = "$p_{h,k}$  share"
K_PROBE_DPI = 200   # the dpi the "is this rectangle blank" check renders at


def load(city):
    z = np.load(SHARES)
    k = KEYS[city]
    pk = z["%s__pk" % k]                      # (13, nd) within-harmonic shares
    P = z["%s__P" % k]                        # (13,) harmonic powers
    joint = pk[:, :K_MAX] * (P / P.sum())[:, None]
    return joint, pk, P


def painter_order(ax, nh, nk, zlo):
    """Far-to-near painting rank of every bar, keyed by its FOOTPRINT centre.

    `Axes3D` orders collections by `Poly3DCollection.do_3d_projection()`, which
    returns `np.min(tzs)` -- the depth of the collection's NEAREST vertex.  At
    elev > 0 the camera is above the floor, so the nearest vertex of a vertical
    column is its ROOF.  A tall bar therefore reports the depth of its top and is
    painted as though it stood in front of any shorter bar between it and the
    camera: a bar at the back is drawn over the bars in front of it.  Measured on
    panel (a): with the automatic order, 8 785 px (0.8 % of the panel) differ from
    the correct order and the difference is visible as pale roofs cut off by the
    dark spike behind them.

    Keying on the footprint is the correct painter's key for a set of columns
    standing on one floor -- a bar is drawn before every bar whose base is nearer
    the camera -- and it is stable: the rank depends on the relative depth of the
    floor points, which no later resize or aspect change reorders.

    Call this AFTER the limits, the view and `frame()` are set, because the
    projection it reads is the one those three define.
    """
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    cells = [(h, k) for h in range(nh) for k in range(nk)]
    depth = {c: float(proj3d.proj_transform(c[0], c[1] + 1.0, zlo, M)[2])
             for c in cells}
    return {c: i for i, c in enumerate(sorted(cells, key=lambda c: -depth[c]))}


def _zoffset(ax):
    """The first zorder above the axes furniture, as `Axes3D.draw` computes it.

    The automatic path raises the collections above the axes so that a bar is
    never hidden by a grid line.  Hand-ordering them means taking that over, so
    the same floor is used here rather than a literal.
    """
    return max(axis.get_zorder() for axis in ax._axis_map.values()) + 1


def bars(ax, joint, cmap, norm, zlo, edge=EDGE):
    """One bar per (h, k) cell.  No interpolation: both axes are ranks.

    The two leading ranks carry a marginally stronger edge than the rest.  The
    text after this figure says the routing spectrum is low-dimensional, so the
    reader wants to pick out k = 1 and k = 2 at a glance; a slightly darker
    stroke does that without drawing a frame or an arrow over the data, which
    would read as steering the reader rather than labelling the axis.

    The zorder is the FOOTPRINT rank from `painter_order`, not mplot3d's own --
    see that function for the defect this avoids.  `computed_zorder` is turned off
    because the automatic path would overwrite it.

    `edge` is `(lead colour, rest colour, lead lw, rest lw)` points and defaults
    to this figure's own values, so passing nothing keeps the composite exactly as
    it was.  It is a parameter because the stroke that separates bars in an
    `h x k` grid is a function of the BARS' on-paper size: the 0.09 pt rest stroke
    is legible in this figure's shallow panels and vanishes once the same grid is
    drawn larger, where the pale faces then merge into one slab.  A figure that
    draws these bars at another scale has to be able to say so.
    """
    z = np.log10(np.clip(joint, 10.0 ** Z_FLOOR, None))
    nh, nk = joint.shape
    rank = painter_order(ax, nh, nk, zlo)
    zo = _zoffset(ax)
    ax.computed_zorder = False
    ec_lead, ec_rest, lw_lead, lw_rest = edge
    for h in range(nh):
        for k in range(nk):
            hgt = z[h, k] - zlo
            if z[h, k] <= zlo:          # at or under the floor: no bar height
                hgt = 0.0
            lead = k < 2                # ranks 1 and 2
            ax.bar3d(h - BAR_H / 2, k + 1.0 - BAR_K / 2, zlo, BAR_H, BAR_K, hgt,
                     color=cmap(norm(z[h, k])), shade=False,
                     edgecolor=ec_lead if lead else ec_rest,
                     linewidth=lw_lead if lead else lw_rest,
                     zorder=zo + rank[(h, k)])
    return z


def full_canvas_png(fig, dpi):
    """Render the figure at its FULL canvas size, as float RGB.

    `style.apply_theme` sets `savefig.bbox = "tight"`, which crops the saved
    image to its contents -- for this row that is 6.94 x 2.06 in out of a
    7.16 x 2.38 in canvas.  Any inch -> pixel arithmetic done on the tight image
    is therefore both SHIFTED (by the crop) and SCALED (by the wrong ratio), and
    it fails silently: the slice is still valid, just somewhere else.  Everything
    that measures the page must ask for the full canvas explicitly.
    """
    from matplotlib.transforms import Bbox
    W, H = fig.get_size_inches()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches=Bbox.from_bounds(0.0, 0.0, W, H))
    buf.seek(0)
    return plt.imread(buf)[..., :3]


def axis_key(fig, ax, rect, at=KEY_AT, box=KEY_BOX, fs=KEY_FS, zlabel=None):
    """ONE axis key for the three panels: which way each axis runs.

    This replaces four pieces of furniture that said two things -- a
    "temporal harmonic h" strip under each panel and a rotated "spatial mode rank
    k" beside the first one.  Three copies of a name is not three times the
    information, and the strips were the tallest empty band in the figure.

    The arrows are drawn along the directions THIS panel's projection actually
    gives the three data axes: each direction is measured by projecting a unit
    step through `get_proj()` and normalising, so the key cannot contradict the
    picture it explains.  And because the key axes' own data coordinates are
    INCHES, a measured direction can be used unrotated.

    The key is only allowed to sit on blank paper.  Before anything is drawn the
    figure is rendered and the key's rectangle is measured against the page
    background; more than 2 % ink raises.  That check is what lets the key live
    inside a panel instead of in a reserved strip, and it is why the position is
    a measured number rather than a taste.
    """
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    # `rect` arrives in figure fractions, like every other rectangle in this row;
    # the key is placed in INCHES because its size must not change with the canvas
    x0, y0, rw, rh = rect[0] * W, rect[1] * H, rect[2] * W, rect[3] * H
    kw, kh = box
    kx, ky = x0 + at[0] * rw, y0 + at[1] * rh

    # -- the key must land on empty page ------------------------------------
    # Measured on the page the file will actually contain, not on the live canvas:
    # with the Tk backend the canvas buffer is the WINDOW's, at whatever size the
    # window manager gave it, so index arithmetic on it silently produced an empty
    # slice and a NaN.  `savefig` to a buffer is the same renderer the deliverable
    # goes through, at a dpi chosen here rather than inherited.
    img = full_canvas_png(fig, K_PROBE_DPI)
    ih, iw = img.shape[:2]
    page = img[2, 2]
    r0 = int(round((1.0 - (ky + kh) / H) * ih))
    r1 = int(round((1.0 - ky / H) * ih))
    c0 = int(round(kx / W * iw))
    c1 = int(round((kx + kw) / W * iw))
    assert r1 > r0 and c1 > c0, ("the key rectangle fell off the page",
                                 (kx, ky, kw, kh), (ih, iw))
    ink = float((np.abs(img[r0:r1, c0:c1] - page).max(axis=2) > 0.03).mean())
    print("axis key   rect %.3f..%.3f x %.3f..%.3f in   ink under it %.2f%%"
          % (kx, kx + kw, ky, ky + kh, 100.0 * ink))
    assert ink <= 0.02, ("the axis key would be drawn over ink", ink)

    ka = fig.add_axes([kx / W, ky / H, kw / W, kh / H])
    ka.set_axis_off()
    ka.set_xlim(0.0, kw)          # the key's data coordinates ARE inches, so a
    ka.set_ylim(0.0, kh)          # measured direction needs no rescaling

    def p(h, k, z):
        from mpl_toolkits.mplot3d import proj3d
        xx, yy = proj3d.proj_transform(h, k, z, ax.get_proj())[:2]
        q = fig.transFigure.inverted().transform(ax.transData.transform((xx, yy)))
        return float(q[0]) * W, float(q[1]) * H

    o = p(0.0, 1.0, Z_FLOOR)

    def unit(q):
        v = (q[0] - o[0], q[1] - o[1])
        n = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return v[0] / n, v[1] / n

    uh = unit(p(1.0, 1.0, Z_FLOOR))      # one step of h, along the floor
    uk = unit(p(0.0, 2.0, Z_FLOOR))      # one step of k
    uz = unit(p(0.0, 1.0, 0.0))          # the floor, up to the top of the box

    # The origin and the three lengths are the only free numbers here; the
    # DIRECTIONS are measured above.  They are set so that every label clears its
    # own arrow inside the box, and so that the box itself fits the empty band
    # under the panel's h numerals -- see the placement note on KEY_AT.
    ox, oy = 0.440, 0.150
    LH, LK, LZ = 0.220, 0.185, 0.135
    for u, L in ((uh, LH), (uk, LK), (uz, LZ)):
        ka.annotate("", xy=(ox + u[0] * L, oy + u[1] * L), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color=S.AXCOL, lw=0.65,
                                    mutation_scale=4.2, shrinkA=0, shrinkB=0))

    # Short names only, "as abbreviated as possible" per the specification: the
    # caption carries the full sentence.
    #
    # The k name is the hard one, and it is why the origin sits at 0.44 in rather
    # than at the 0.28 in a centred triad.  Two constraints:
    #
    #   ABOVE the k arrow.  The arrow leaves the origin going down-left, so it
    #   never rises above the origin's own y = 0.15; a label whose lower edge
    #   clears 0.15 cannot cross it at ANY x.  That is what (0.030, 0.200) is --
    #   the fix that moved an earlier version down into the arrow's path.
    #
    #   LEFT of the z arrow.  This is the binding one.  The z arrow climbs
    #   straight up from the origin, so it cuts every horizontal band above 0.15
    #   at the origin's x.  "$k$  mode rank" is about 0.37 in wide and starts at
    #   the box's left edge, so it ends near 0.40: the origin has to sit to the
    #   RIGHT of that, and no further right than the h name can follow.  At 0.44
    #   the z arrow misses the k name by 0.037 in, and the h name, measured from
    #   its own arrow tip, still ends inside the box.
    ka.text(ox + uh[0] * LH + 0.010, oy + uh[1] * LH + 0.006,
            "$h$  harmonic", fontsize=fs, color=S.FG, ha="left", va="center")
    ka.text(0.030, 0.200,
            "$k$  mode rank", fontsize=fs, color=S.FG, ha="left", va="center")
    ka.text(ox + uz[0] * LZ + 0.014, oy + uz[1] * LZ - 0.010,
            Z_LABEL if zlabel is None else zlabel, fontsize=fs, color=S.FG,
            ha="left", va="center")


def topview():
    """The same 351 shares as flat (h, k) heatmaps.  DIAGNOSTIC ONLY.

    A 3D bar chart is a projection, and a projection can hide a cell behind a
    taller one.  This draws the identical array with no perspective at all, so
    any concentration the 3D view fails to show is visible here; run it beside
    the 3D figure and compare cell by cell.  It shares the cmap, the log norm and
    the floor with the 3D drawing, so a difference between the two is geometry
    and never colour.
    """
    norm = mpl.colors.Normalize(vmin=Z_FLOOR, vmax=0.0)
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for ax, city in zip(axs, CITIES):
        joint, pk, P = load(city)
        z = np.log10(np.clip(joint, 10.0 ** Z_FLOOR, None)).T   # (k, h)
        im = ax.imshow(z, origin="lower", aspect="auto", cmap=CMAP, norm=norm,
                       extent=(-0.5, H_MAX + 0.5, 0.5, K_MAX + 0.5),
                       interpolation="nearest")
        ax.set_xticks(range(0, H_MAX + 1))
        ax.set_xticklabels([str(h) for h in range(0, H_MAX + 1)], fontsize=5.8)
        ax.set_yticks(range(1, K_MAX + 1))
        ax.set_yticklabels([str(k) for k in range(1, K_MAX + 1)], fontsize=5.8)
        ax.set_title(city, fontsize=8.4)
        ax.set_xlabel("temporal harmonic $h$", fontsize=7.0)
        S.tidy(ax)
    axs[0].set_ylabel("spatial mode rank $k$", fontsize=7.0)
    # same cmap, same log norm and same floor as the 3D drawing, so a difference
    # between the two views is geometry and can never be colour
    cb = fig.colorbar(im, ax=axs, fraction=0.022, pad=0.015)
    cb.set_label("routing power share $p_{h,k}$ (log)", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.6)
    cb.set_ticks([-3.5, -3, -2, -1, 0])
    cb.set_ticklabels([r"$10^{-3.5}$", r"$10^{-3}$", r"$10^{-2}$",
                       r"$10^{-1}$", r"$1$"])
    cb.outline.set_linewidth(0.3)
    cb.outline.set_edgecolor(S.AXCOL)
    out = HERE / "fig2d_spectrum3d_topview"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=340)
    print("wrote %s.pdf / .png (diagnostic, not for the paper)" % out.name)


def frame(ax, zoom=1.0):
    """Thin axes, no back walls, no box: the paper's own 2D furniture in 3D.

    `zoom` magnifies the box inside its axes rectangle.  At zoom 1 the projected
    box fills only about three quarters of the rectangle, and across three panels
    that slack reads as inter-panel whitespace; enlarging the box closes it
    without moving the panels or changing any angle.
    """
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("none")
        axis.line.set_color(S.AXCOL)
        axis.line.set_linewidth(0.55)
        axis._axinfo["grid"].update(color="#e6eaee", linewidth=0.35,
                                    linestyle="-")
        axis._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.14)
    ax.set_box_aspect((1.5, 1.0, 0.82), zoom=zoom)


def spectra_row(fig, rects, cbar_rect, linear=False, zoom=None, key=True,
                tags=None):
    """Draw the three-city (h, k) spectra into `fig` at caller-given rectangles.

    Lifted out of `main()` so Figure 2 can place this row and the geographic row
    beneath it in ONE figure.  The alternative -- a second copy of the drawing
    code in the composite -- would be the same panel twice, and the two copies
    would drift apart the first time either one is touched.

    `rects` are `(x0, y0, w, h)` in figure fractions, one per city in `CITIES`
    order; `cbar_rect` is the colorbar, also in figure fractions.  Everything the
    three panels SHARE -- viewing angle, axis ranges, z scale, colour scale, the
    display floor -- stays fixed inside this function, because those are
    properties of the comparison and not of the page.  The one thing the caller
    owns is where the row sits.

    `tags`, if given, is one label per panel drawn inside the picture at its
    top-left ("(a)"); the city name is drawn inside at the top-right either way.
    Both are `text2D` in axes coordinates, so they cost no figure height.

    The axes are named ONCE, by `axis_key` in the first panel; there is no
    per-panel axis-name strip, because three copies of a name are not three times
    the information and the strip cost the figure a band of height it needed.
    """
    norm = mpl.colors.Normalize(vmin=Z_FLOOR, vmax=0.0)
    norm_lin = mpl.colors.Normalize(vmin=0.0, vmax=0.68)

    axes = []
    for i, city in enumerate(CITIES):
        joint, pk, P = load(city)
        # the normalisation the figure claims, checked rather than assumed.
        # `pk` partitions within each harmonic; `joint` is that share times the
        # harmonic's share of the whole city power, so it sums to 1 only over the
        # FULL mode range.  Truncating at k = 9 keeps the fraction printed below.
        assert abs(pk.sum(axis=1) - 1.0).max() < 1e-9, (
            city, "mode shares do not partition within a harmonic")
        assert (joint >= 0).all() and np.isfinite(joint).all()
        assert joint.sum() <= 1.0 + 1e-12, (city, "the shares exceed one")
        # the display floor must not be clipping a real cell, or the figure is
        # silently dropping power that the diagnostics still carry
        assert joint.min() > 10.0 ** Z_FLOOR, (
            city, "a drawn share is under the display floor", joint.min())
        print("%-14s lowest drawn cell %.2e (log10 %.2f, floor %.1f, "
              "margin %.2f decades)"
              % (city, joint.min(), np.log10(joint.min()), Z_FLOOR,
                 np.log10(joint.min()) - Z_FLOOR))
        print("%-14s kept %.4f of the routing power inside k <= %d "
              "(per h: %.3f .. %.3f)"
              % (city, joint.sum(), K_MAX, pk[:, :K_MAX].sum(axis=1).min(),
                 pk[:, :K_MAX].sum(axis=1).max()))

        ax = fig.add_axes(list(rects[i]), projection="3d")
        axes.append(ax)

        # -- geometry first, bars second -----------------------------------
        # The bars are painted in a hand-computed order (`painter_order`), and
        # that order is a property of the projection -- so the limits, the view
        # and the box aspect have to be final before a single bar is added.
        ax.set_xlim(-0.7, H_MAX + 0.7)
        # THE k AXIS IS DRAWN BACK-TO-FRONT, and that is a layout choice, not a
        # style.  At this azimuth the h = 0 column is already the far end of h,
        # but k = 1 was the NEAR end of k -- so the tallest bar in the figure
        # (h = 0, k = 1, the persistent bin's leading mode) stood at the front
        # corner and walled off the rows behind it.  Inverting the axis puts
        # k = 1 at the back: the tall bars rise behind the short ones and the
        # short ones are still drawn in front of them, which is what the manual
        # painting order in `bars` now guarantees.  The tick values are
        # unchanged, so the axis still reads 1..9 -- it is read from the far end
        # inward rather than the near end outward.
        ax.set_ylim(K_MAX + 0.5, 0.5)
        if linear:
            ax.set_zlim(0.0, 0.68)
            ax.set_zticks([0.0, 0.2, 0.4, 0.6])
        else:
            ax.set_zlim(Z_FLOOR, 0.0)
            # Decade ticks stay, so the horizontal grid planes mark the powers of
            # ten; the NUMBERS live on the shared colorbar and nowhere else.
            # mplot3d clips a z tick label at the text width it measures before
            # the exponent is laid out -- it does this for mathtext, for Unicode
            # superscripts and for plain digits alike (measured, not assumed), so
            # a labelled z axis here would print a truncated scale.  The colorbar
            # is a 2D axis and renders the same labels correctly, and it encodes
            # the same quantity as the height, so there is nothing to duplicate.
            ax.set_zticks([-3.5, -3, -2, -1, 0])
        # EVERY THIRD TICK.  Both axes are ranks and the reader knows they step by
        # one, so thirteen numbers on h and nine on k are nine and six numbers
        # more than the axis needs -- and in a three-across row those numerals
        # crowd the neighbouring panel.  h: 0, 3, 6, 9, 12 (which keeps both ends
        # and the h = 0 that the caption names).  k: 1, 4, 7, the stride that keeps
        # the leading rank labelled without wasting the seam.
        ax.set_xticks(range(0, H_MAX + 1, 3))
        ax.set_xticklabels([str(h) for h in range(0, H_MAX + 1, 3)],
                           fontsize=5.8)
        ax.set_yticks((1, 4, 7))
        ax.set_yticklabels(["1", "4", "7"], fontsize=5.4, color=S.AXCOL)
        # the leading rank is the one the text singles out; the rest of the axis
        # is one uniform set of numbers
        ax.get_yticklabels()[0].set_color(S.FG)
        ax.get_yticklabels()[0].set_fontweight("semibold")
        ax.set_zticklabels([])
        # negative padding pulls each numeral back towards its own axis line.
        # mplot3d parks 3D tick labels well clear of the axis by default, which in
        # this figure left a gap wider than the numerals themselves; the numbers
        # are read off the axis, not off the panel edge.
        ax.tick_params(axis="x", pad=TICK_PAD_X)
        ax.tick_params(axis="y", pad=TICK_PAD_Y)
        # 24 deg buries the k >= 2 rows behind the k = 1 column; 36 deg looks
        # over it while keeping the h direction readable as a direction.
        ax.view_init(elev=ELEV, azim=AZIM)
        frame(ax, BOX_ZOOM if zoom is None else zoom)

        # -- now the bars ---------------------------------------------------
        if linear:
            # The comparison variant.  Height AND colour both go linear here, so
            # the caption's "height and colour are the same number" stays true;
            # mixing a linear height with a log colour would make the two panels
            # disagree about the same cell.
            rank = painter_order(ax, joint.shape[0], joint.shape[1], 0.0)
            zo = _zoffset(ax)
            ax.computed_zorder = False
            for h_ in range(joint.shape[0]):
                for k_ in range(joint.shape[1]):
                    ax.bar3d(h_ - 0.30, k_ + 0.70, 0.0, 0.60, 0.60,
                             joint[h_, k_], color=CMAP(norm_lin(joint[h_, k_])),
                             shade=False, edgecolor="#5d7d9b", linewidth=0.10,
                             zorder=zo + rank[(h_, k_)])
        else:
            bars(ax, joint, CMAP, norm, Z_FLOOR)

        # -- the panel's own two words, inside the picture --------------------
        # `(a)` top-left and the city top-right, both inside the axes rectangle
        # rather than as a title above it: the corners of the 3D box above the
        # bar field are empty air in every one of the three panels, so the labels
        # cost the figure no height at all.  As a title they had to sit clear of
        # the box, which is a band across the full width that draws nothing else.
        if tags is not None:
            ax.text2D(0.020, 0.985, tags[i], transform=ax.transAxes, ha="left",
                      va="top", fontsize=8.4, color=S.FG)
        ax.text2D(0.980, 0.985, SHORT[city], transform=ax.transAxes, ha="right",
                  va="top", fontsize=8.4, color=S.FG)

    if key:
        axis_key(fig, axes[0], rects[0])

    # Narrow and subordinate: the bar is the display, the colorbar only names
    # the scale, so it is the thinnest a readable colorbar gets and its ticks are
    # two points smaller than the axis ticks it sits beside.
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_lin if linear else norm,
                                            cmap=CMAP),
                      cax=fig.add_axes(list(cbar_rect)))
    cb.set_label("routing power share $p_{h,k}$", fontsize=6.2, labelpad=2.0)
    cb.ax.tick_params(labelsize=5.4, length=1.6, width=0.4)
    if linear:
        cb.set_ticks([0.0, 0.2, 0.4, 0.6])
    else:
        cb.set_ticks([-3.5, -3, -2, -1, 0])
        cb.set_ticklabels([r"$10^{-3.5}$", r"$10^{-3}$", r"$10^{-2}$",
                           r"$10^{-1}$", r"$1$"])
    cb.outline.set_linewidth(0.3)
    cb.outline.set_edgecolor(S.AXCOL)
    return axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true",
                    help="draw z linearly instead of in log10, for comparison")
    ap.add_argument("--topview", action="store_true",
                    help="diagnostic: the same shares as a plain (h, k) heatmap, "
                         "flat, so the 3D projection can be checked against it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    S.apply_theme()

    if args.topview:
        return topview()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    rects = [(PANEL_X[i], PANEL_Y, PANEL_W, PANEL_H) for i in range(len(CITIES))]
    spectra_row(fig, rects, (0.958, 0.330, 0.008, 0.400), linear=args.linear,
                tags=("(a)", "(b)", "(c)"))

    name = args.out or ("fig2d_spectrum3d_linear" if args.linear
                        else "fig2d_spectrum3d")
    out = HERE / name
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=340)
    print("wrote %s.pdf / .png" % out.name)


if __name__ == "__main__":
    main()
