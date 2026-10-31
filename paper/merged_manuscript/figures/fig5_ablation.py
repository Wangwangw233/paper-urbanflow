r"""Figure 5 -- what the road arrangement buys, and what it does not.

TWO QUESTIONS, ONE VISUAL GRAMMAR.  (a) SPECIFICITY: is the road arrangement
better than an arbitrary subspace of the same rank?  (b) INCOMPLETENESS: at
matched medium degrees of freedom, does the road description explain as much of
the held-out routing as a medium inferred from mobility alone?  They are
different questions measured by different scores, and the earlier edition
apologised for that in the caption -- "the two panels use different scores and
should not be compared".  An apology is not an encoding.  The panels carry
DIFFERENT NAMES, DIFFERENT X-AXIS QUANTITIES and the ONE MARKER THAT MEANS THE
SAME THING IN BOTH:

    solid square, city colour    =  the real road medium (identical in both)
    grey violin                  =  a rank-matched surrogate road base   (a)
    dark filled circle           =  medium inferred from mobility alone   (b)
    pale open triangle           =  destroyed road arrangement            (b)

So the eye learns one glyph per role and never has to carry a caveat across.

PANEL (a) IS AN EFFECT DISTRIBUTION, NOT A RAW ERROR.  The quantity is the share
of the observed routing energy left OUTSIDE the representation,
`E(U) = 1 - sum_t ||U^T F_t||^2 / sum_t ||F_t||^2` -- a subspace-capture fraction
of the observed field, with no model, no fit and no prediction in it.  On the
real road basis it is 0.0997 in Porto and 0.0995 in San Francisco; the 999
rank-matched subspaces live in 0.330-0.403 and 0.382-0.427.  Drawn raw, one
honest linear axis is 58 % empty and the panel degrades to "a point and a distant
density".  The drawn quantity is therefore the deterministic transform
`dE = E_null - E_road`: the same 999 draws, translated so that the road basis
sits at ZERO.  Positive means the random subspace leaves more energy uncaptured
than the road basis does, and the whole distribution is far to the right of the
zero reference.  The only compression left is the axis break between the
reference and the distribution, and it is drawn rather than hidden.

PANEL (b) IS A DUMPBELL, ONE ROW PER CITY.  Three markers on each city's row --
destroyed arrangement, road medium, mobility-inferred -- so the eye reads the
ladder `arrangement matters / roads matter / roads are incomplete` off a single
axis whose zero is the constant medium.  An exact paired-date 95 % interval is
printed for the contrast the panel turns on, `mobility - road`.  Drawn AT THE
SELECTED RANK ONLY, because the text does not discuss the rank path and a
multi-curve DOF sweep invites a reading of the path that nothing in the paper
supports.

NOTHING IS FITTED.  (a) reads the frozen `e_null` arrays of
`final_ablation/results/partA_topology.json`; (b) reads
`benchmark/domain_rankmatched_results.csv`, whose anchors are asserted
in-process: at `R = 1` every medium collapses to the constant one and reproduces
`D0` exactly, and at `R = n_int` the three sources reproduce the frozen rows.  The
printed intervals come from `benchmark/domain_rankmatched_pairwise.csv`.
"""
from __future__ import annotations

import csv

import numpy as np
import matplotlib.pyplot as plt

import figstyle as FS

S = FS.S

TOPO = FS.ROOT / "final_ablation" / "results" / "partA_topology.json"
RM = FS.ROOT / "benchmark" / "domain_rankmatched_results.csv"
PW_CSV = FS.ROOT / "benchmark" / "domain_rankmatched_pairwise.csv"

# ---- layout, in inches ----------------------------------------------------
W, H = 7.16, 2.34
FOOT, TOP = 0.50, 0.24
XL, RIGHT, GAPX = 0.62, 0.22, 0.62
PW = (W - XL - RIGHT - GAPX) / 2.0
PH = H - FOOT - TOP
X = (XL, XL + PW + GAPX)
TITLE = {"a": "Specificity", "b": "Incompleteness"}     # what each panel asks

# The broken x axis of panel (a): a narrow zero reference, a gap for the
# twin-slash mark, and the wide piece that holds the effect distribution.
A_W1, A_GAP = 0.20, 0.085
A_W2 = PW - A_W1 - A_GAP
A_Z = (-0.014, 0.014)                   # the narrow piece, around dE = 0
AMP = 0.34                              # violin half-height, in row units
# Both panels use the same y range so the two city baselines line up across them.

SHUF_COLOR = "#9aa0a6"                  # the destroyed arrangement's pale glyph
DUMBBELL = ("shuf", "road", "mob")      # the ladder, in reading order


def nulls():
    summ = FS.jload(TOPO)["summaries"]
    out = {}
    for city in FS.CITIES:
        row = [s for s in summ if s["city"] == city and s["pool"] == "full"][0]
        road = float(row["road"])
        d = np.asarray(row["e_null"], float) - road       # the drawn quantity
        out[city] = dict(de=d, road=road, q01=float(np.percentile(d, 1)))
        print("%-6s E_road %.4f  dE %.4f..%.4f (median %.4f, q01 %.4f)  n %d"
              % (city, road, d.min(), d.max(), float(np.median(d)),
                 out[city]["q01"], d.size))
    FS.assert_close(out["Porto"]["road"], 0.099744, 1e-4, "Porto E_road")
    FS.assert_close(out["SF"]["road"], 0.099523, 1e-4, "SF E_road")
    for city in FS.CITIES:
        assert out[city]["de"].min() > 0.15, (city, "distribution reaches zero")
    return out


def sweep():
    """The matched-DOF medium sweep, reduced to the frozen selected rank.

    The path is read only to check the two anchors and to confirm the selected
    rank is the largest `R` in the file; it is never drawn.
    """
    rows = list(csv.DictReader(open(RM)))
    ci = list(csv.DictReader(open(PW_CSV)))
    out = {}
    for city in ("Porto", "San Francisco"):
        d = {}
        for r in rows:
            if r["city"] != city:
                continue
            d.setdefault(r["source"], {})[int(r["medium_dof"])] = float(r["R2_cond"])
        Rs = sorted(d["road"])
        base = d["road"][1]                       # R = 1 reproduces D0 exactly
        nint = max(Rs)
        assert base == d["mobility"][1], (city, "R=1 anchors disagree", base)
        FS.assert_close(base, 0.613027061 if city == "Porto" else 0.743364177,
                        1e-6, "%s D0 anchor" % city)
        pair = [r for r in ci if r["city"] == city
                and r["contrast"] == "mobility - road"
                and int(r["R"]) == nint]
        assert len(pair) == 1, (city, "no mobility-road interval at R = n_int")
        out[city] = dict(nint=float(nint), base=base,
                         road=d["road"][nint] - base,
                         mob=d["mobility"][nint] - base,
                         shuf=d["mobility_shuf"][nint] - base,
                         gap=float(pair[0]["delta_mean"]),
                         gap_ci=(float(pair[0]["ci95_lo"]),
                                 float(pair[0]["ci95_hi"])))
        print("%-14s selected rank R = %d  road %+.4f  mob %+.4f  shuf %+.4f  "
              "mob-road %+.4f [%+.4f,%+.4f]"
              % (city, nint, out[city]["road"], out[city]["mob"],
                 out[city]["shuf"], out[city]["gap"], out[city]["gap_ci"][0],
                 out[city]["gap_ci"][1]))
        FS.assert_close(out[city]["gap"], out[city]["mob"] - out[city]["road"],
                        1e-6, "%s mobility-road gap" % city)
    FS.assert_close(out["Porto"]["mob"], 0.0917, 1e-3, "Porto mob gain")
    FS.assert_close(out["San Francisco"]["mob"], 0.0851, 1e-3, "SF mob gain")
    FS.assert_close(out["Porto"]["shuf"], -0.0467, 1e-3, "Porto destroyed gain")
    FS.assert_close(out["San Francisco"]["shuf"], -0.0524, 1e-3,
                    "SF destroyed gain")
    return out


def density(draws, grid):
    """Gaussian KDE of the 999 surrogates, normalised to unit peak."""
    h = 1.06 * draws.std(ddof=1) * draws.size ** (-0.2)
    z = (grid[:, None] - draws[None, :]) / h
    return np.exp(-0.5 * z * z).sum(axis=1) / (draws.size * h * np.sqrt(2 * np.pi))


def row_index(city):
    return FS.CITIES.index("Porto" if city == "Porto" else "SF")


def panel_title(ax, key):
    ax.text(0.0, 1.035, TITLE[key], transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7.8, fontweight="bold", color=FS.INK)


def break_marks(ax, side):
    """The break mark: one diagonal stroke crossing the bottom spine.

    Each piece draws ONE stroke, at its inner corner, so the two together read
    as the standard `//` mark across the discontinuity.  Both lean the same way
    and are drawn at the same size, so the pair is symmetric about the gap; at
    the previous 0.042 in / 0.9 pt they were barely visible at print size.
    """
    pos = ax.get_position()
    dx = (0.055 / W) / pos.width
    dy = (0.055 / H) / pos.height
    x = 1.0 if side == "l" else 0.0
    ax.plot([x - dx, x + dx], [-dy, dy], transform=ax.transAxes, lw=1.3,
            color=FS.AXCOL, clip_on=False, zorder=6, solid_capstyle="butt")


def main():
    nl, sel = nulls(), sweep()
    # Every drawn value must be inside the axes; read off the data, not off the
    # layout arithmetic.
    for city in sel:
        for k in DUMBBELL:
            v = sel[city][k]
            assert -0.072 <= v <= 0.130, ("%s %s leaves the range" % (city, k), v)

    fig = plt.figure(figsize=(W, H))

    # ---- (a) specificity: the road medium as the ZERO of an effect ---------
    axa1 = fig.add_axes([X[0] / W, FOOT / H, A_W1 / W, PH / H])
    axa2 = fig.add_axes([(X[0] + A_W1 + A_GAP) / W, FOOT / H, A_W2 / W, PH / H])
    lo = min(float(nl[c]["de"].min()) for c in FS.CITIES) - 0.014
    hi = max(float(nl[c]["de"].max()) for c in FS.CITIES) + 0.014
    grid = np.linspace(lo, hi, 300)
    for i, city in enumerate(FS.CITIES):
        col = FS.CITY_COLOR[city]
        d = nl[city]["de"]
        dens = density(d, grid)
        dens = dens / dens.max()
        axa2.fill_between(grid, i - dens * AMP, i + dens * AMP, color=S.NULL,
                          alpha=0.72, lw=0.5, edgecolor="#6f7377", zorder=2)
        axa2.plot([d.min(), d.max()], [i, i], "-", color="#8a9096", lw=0.6,
                  zorder=3)
        # The closest surrogate is marked, not labelled: two text labels in this
        # panel crowded the violin and said only what the caption can say once,
        # for both cities.  The dashed rule stands where the 1st percentile is.
        axa2.plot([nl[city]["q01"]] * 2, [i - AMP, i + AMP], ls=(0, (2, 2)),
                  lw=0.9, color="#5c6166", zorder=4)
        # the road medium IS the zero of this axis; it is drawn as a marker on
        # its own rule so the reference is an object and not just a grid line.
        axa1.plot([0.0, 0.0], [i - AMP, i + AMP], "-", color=col, lw=1.7,
                  zorder=4)
        axa1.plot([0.0], [i], "s", color=col, ms=6.0, mec="w", mew=0.8,
                  zorder=5)
    for ax in (axa1, axa2):
        ax.set_ylim(-0.62, 1.95)
        FS.light_axes(ax, grid=False)
    axa1.set_xlim(*A_Z)
    axa1.set_xticks([0.0])
    axa1.set_xticklabels(["road\nmedium"])      # the zero of the axis is an object
    axa1.set_yticks([0, 1])
    axa1.set_yticklabels(list(FS.CITIES))
    axa1.tick_params(axis="x", labelsize=5.0)
    axa2.set_xlim(lo, hi)
    axa2.set_xticks([0.24, 0.28, 0.32])
    axa2.spines["left"].set_visible(False)
    axa2.tick_params(axis="y", which="both", left=False, labelleft=False)
    for ax in (axa1, axa2):
        ax.spines["top"].set_visible(False)
    axa2.spines["right"].set_visible(False)
    break_marks(axa1, "l")
    break_marks(axa2, "r")
    # the shared x label sits under the WHOLE broken axis, not under its right
    # piece, which is where matplotlib would put it.
    axa2.set_xlabel(r"excess uncaptured routing energy, "
                    r"$\Delta E = E_{\rm null} - E_{\rm road}$"
                    "\n(higher $=$ real road better)", fontsize=6.4, labelpad=1.6,
                    linespacing=1.5)
    axa2.xaxis.set_label_coords((PW / 2 - A_W1 - A_GAP) / A_W2, -0.165)
    panel_title(axa1, "a")
    S.panel_tag(axa1, "a")

    # ---- (b) incompleteness: one dumbbell row per city ---------------------
    axb = fig.add_axes([X[1] / W, FOOT / H, PW / W, PH / H])
    axb.axvline(0.0, color="#9aa0a6", lw=0.8, ls="--", zorder=1)
    for city in ("Porto", "San Francisco"):
        i = row_index(city)
        col = FS.CITY_COLOR["Porto" if city == "Porto" else "SF"]
        vals = [sel[city][k] for k in DUMBBELL]
        axb.plot([min(vals), max(vals)], [i, i], "-", color="#d3d8dd", lw=2.2,
                 solid_capstyle="round", zorder=2)
        for k, mk, c in (("shuf", "v", SHUF_COLOR), ("road", "s", col),
                         ("mob", "o", col)):
            axb.plot([sel[city][k]], [i], marker=mk, ls="none", ms=5.2,
                     color=c, mec="w", mew=0.7, zorder=5)
        axb.text(sel[city]["mob"], i + 0.26, "%.3f" % sel[city]["mob"],
                 fontsize=5.2, color=col, ha="center", va="bottom")
        axb.text(sel[city]["shuf"], i + 0.26, "%.3f" % sel[city]["shuf"],
                 fontsize=5.2, color="#6f7377", ha="center", va="bottom")
        axb.text(-0.072, i - 0.40,
                 "mobility $-$ road  $%+.3f\\;[%+.3f, %+.3f]$"
                 % (sel[city]["gap"], sel[city]["gap_ci"][0],
                    sel[city]["gap_ci"][1]),
                 fontsize=5.2, color="#6f7377", ha="left", va="top",
                 bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none",
                           alpha=0.92), zorder=6)
    axb.set_ylim(-0.62, 1.95)
    axb.set_yticks([0.0, 1.0])
    axb.set_yticklabels(list(FS.CITIES))
    axb.set_xlim(-0.072, 0.130)
    axb.set_xticks([-0.05, 0.0, 0.05, 0.10])
    axb.set_xticklabels(["$-0.05$", "0", "0.05", "0.10"])
    axb.set_xlabel(r"gain in conditional $R^2$ over a constant medium  "
                   r"(higher is better)", fontsize=6.6, labelpad=1.6)
    FS.light_axes(axb, grid=False)
    # The key sits in the empty band above the two dumbbell rows, at a size the
    # glyphs can actually be told apart at; at 5.1 pt the three markers were
    # near-identical smudges.
    key = [plt.Line2D([], [], ls="none", marker=mk, ms=5.4,
                      color=c, mec="w", mew=0.7, label=lab)
           for mk, c, lab in (("v", SHUF_COLOR, "destroyed arrangement"),
                              ("s", FS.FG, "road medium"),
                              ("o", FS.FG, "mobility-inferred"))]
    axb.legend(handles=key, frameon=False, fontsize=6.0, loc="upper left",
               borderpad=0.0, borderaxespad=0.10, handletextpad=0.50,
               labelspacing=0.30)
    panel_title(axb, "b")
    S.panel_tag(axb, "b")
    return fig


if __name__ == "__main__":
    for name in ("fig5_ablation", "fig5_ablation_v5"):
        for p in FS.save(main(), name):
            print("wrote", p)
