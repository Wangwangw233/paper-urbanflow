from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "frozen" / "spectra" / "phase_metrics.json"
OUT = Path(os.environ.get("URBANFLOW_FIGURE_OUT",
                          ROOT / "figures" / "main"))
CITY = ("Porto", "SF", "Chengdu")
LABEL = {"Porto": "Porto", "SF": "San Francisco", "Chengdu": "Chengdu"}
COL = {"Porto": "#315f85", "SF": "#a76539", "Chengdu": "#49735b"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0,
    "axes.labelsize": 7.2, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.linewidth": 0.55,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))["cities"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.85), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.815,
                        wspace=0.24, hspace=0.40)
    for j, city in enumerate(CITY):
        folds = data[city]["folds"]
        r2 = np.asarray([f["bin_R2"] for f in folds])
        sg = np.asarray([f["bin_signed_pooled"] for f in folds])
        ab = np.asarray([f["bin_abs_mean"] for f in folds])
        m = np.arange(r2.shape[1])
        for row, arr, ylabel in ((0, r2, "Phase-sensitive $R_m^2$"),
                                 (1, sg, "Signed alignment")):
            ax = axes[row, j]
            mean = arr.mean(0)
            if len(folds) > 1:
                ax.fill_between(m, arr.min(0), arr.max(0), color=COL[city],
                                alpha=0.13, lw=0)
            ax.plot(m, mean, "o-", color=COL[city], lw=1.2, ms=3.0,
                    mfc="white", mew=0.8, label="phase-sensitive")
            if row == 0:
                ax.plot(m, ab.mean(0), ":", color="#6d757c", lw=1.0,
                        label="phase-insensitive $|\\gamma|$")
            ax.axhline(0, color="#9aa1a8", lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.set_xlim(-0.3, m[-1] + 0.3)
            ax.set_xticks([0, 3, 6, 9, 12] if city != "Chengdu" else [0, 3, 6, 9])
            ax.set_ylim(-0.45, 1.05) if row == 0 else (-0.25, 1.05)
            ax.grid(axis="y", color="#e4e7ea", lw=0.45, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(width=0.5, length=2.2, pad=1.5)
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=3.0)
            if row == 1:
                ax.set_xlabel("Temporal harmonic $m$" if city != "Chengdu"
                              else "18-point DFT bin $m$", labelpad=2.0)
            if row == 0:
                ax.set_title(f"{LABEL[city]}\n" +
                             ("5 folds; mean and range" if city == "SF"
                              else f"{len(folds[0]['test_dates'])} TEST dates"),
                             color=COL[city], pad=3.0)
            ax.text(0.03, 0.06, "abc"[j] if row == 0 else "def"[j],
                    transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                    va="bottom", ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig03_frequency_response.{ext}", dpi=600,
                    bbox_inches="tight", pad_inches=0.03)
    print("wrote", OUT / "fig03_frequency_response.pdf")


if __name__ == "__main__":
    main()
