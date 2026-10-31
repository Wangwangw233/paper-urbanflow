"""The 3-D spectrum panel used by the public supplementary figure."""
from __future__ import annotations
import matplotlib as mpl
import figstyle as FS
import fig2d_spectrum3d as F2
from fig2d_spectrum3d import (BOX_ZOOM, CMAP, ELEV, H_MAX, SHORT, TICK_PAD_X,
                              TICK_PAD_Y, Z_FLOOR, bars, frame)
S = FS.S
EDGE3 = ("#1b3a57", "#42678c", 0.34, 0.30)
CITIES = ("Porto", "SF")
FULL = {"Porto": "Porto", "SF": "San Francisco"}

def spectra_panel(fig, rect, q, city, tag, city_label=True, zoom=BOX_ZOOM):
    """One source-excitation 3D spectrum, on Figure 2's own conventions.

    `city_label=False` suppresses the in-panel city name for callers that label
    a whole column once instead of repeating the city in every panel.

    `zoom` is passed straight to `frame`; it magnifies the 3D box inside its axes
    rectangle.  The default is the module's measured value.  A caller whose
    rectangles are laid out differently -- the 2 x 2 main figure, whose panels are
    wider and closer than the three-across composite -- may raise it, and the
    caller owns the check that the enlarged boxes do not reach into a neighbour's
    numerals.

    Every drawing decision inside is Figure 2's: the limits, the inverted k axis,
    the shared z floor, the every-third tick, the tick padding, the view.  What
    differs is only the array handed to `bars`.
    """
    nh, nk = q.shape
    ax = fig.add_axes(list(rect), projection="3d")

    ax.set_xlim(-0.7, H_MAX + 0.7)
    # k is drawn back-to-front, as in Figure 2: at this azimuth the tallest bar
    # would otherwise stand at the front corner and wall off the rows behind it.
    ax.set_ylim(nk + 0.5, 0.5)
    ax.set_zlim(Z_FLOOR, 0.0)
    ax.set_zticks([-3.5, -3, -2, -1, 0])
    ax.set_xticks(range(0, H_MAX + 1, 3))
    ax.set_xticklabels([str(h) for h in range(0, H_MAX + 1, 3)], fontsize=5.8)
    ax.set_yticks((1, 4, 7))
    ax.set_yticklabels(["1", "4", "7"], fontsize=5.4, color=S.AXCOL)
    ax.get_yticklabels()[0].set_color(S.FG)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.set_zticklabels([])
    ax.tick_params(axis="x", pad=TICK_PAD_X)
    ax.tick_params(axis="y", pad=TICK_PAD_Y)
    ax.view_init(elev=ELEV, azim=F2.AZIM)
    frame(ax, zoom)

    norm = mpl.colors.Normalize(vmin=Z_FLOOR, vmax=0.0)
    bars(ax, q, CMAP, norm, Z_FLOOR, edge=EDGE3)

    ax.text2D(0.020, 0.985, tag, transform=ax.transAxes, ha="left", va="top",
              fontsize=8.4, color=S.FG)
    if city_label:
        ax.text2D(0.980, 0.985, SHORT[FULL[city]], transform=ax.transAxes,
                  ha="right", va="top", fontsize=8.4, color=S.FG)
    return ax, norm
