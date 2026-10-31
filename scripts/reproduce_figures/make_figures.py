"""Build the seven supplement figures from archived products only.

No model is fitted here. Every input is hashed, every plotted quantity is
read from an archive or formed by the explicit transform recorded in the
manifest, and every figure is written below figures/supplementary/.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
ROOT = Path(os.environ.get("URBANFLOW_ROOT", REPO_ROOT))
FIGURES = Path(os.environ.get("URBANFLOW_FIGURE_OUT", REPO_ROOT / "figures" / "supplementary"))
RESULT_OUT = Path(os.environ.get("URBANFLOW_RESULT_OUT", REPO_ROOT / "results" / "derived"))
MANIFEST = RESULT_OUT / "threecity_supplement_manifest.json"
FIGURES.mkdir(parents=True, exist_ok=True)

CITIES = ("Porto", "SF", "Chengdu")
NAME = {"Porto": "Porto", "SF": "San Francisco", "Chengdu": "Chengdu"}
COL = {"Porto": "#315f85", "SF": "#a76539", "Chengdu": "#49735b"}
GREY = "#626a72"
LIGHT = "#d9dee3"
MIN_FONT = 6.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": MIN_FONT,
    "ytick.labelsize": MIN_FONT,
    "legend.fontsize": MIN_FONT,
    "axes.linewidth": 0.55,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.3,
    "ytick.major.size": 2.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

INPUT_HASHES: dict[str, str] = {}
CHECKS: list[str] = []


def track(path: Path) -> Path:
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if rel in INPUT_HASHES:
        assert INPUT_HASHES[rel] == digest
    INPUT_HASHES[rel] = digest
    return path


def jload(path: Path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def nload(path: Path):
    return np.load(track(path), allow_pickle=False)


def csvload(path: Path):
    with track(path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def check(condition, message: str):
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def close(a, b, tol, message: str):
    check(abs(float(a) - float(b)) <= tol,
          f"{message}: {float(a):.12g} vs {float(b):.12g} (tol {tol:g})")


def tidy(ax, grid=False):
    if grid:
        ax.grid(axis="y", color="#dfe3e7", lw=0.45, ls=(0, (2, 2)), zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=MIN_FONT, pad=1.5)


def tag(ax, text):
    ax.text(0.025, 0.965, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, fontweight="bold", color="#222222", zorder=20,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.78))


def metadata(title: str, subject: str):
    return {
        "Title": title,
        "Author": "UrbanFlow manuscript authors",
        "Creator": "UrbanFlow three-city supplement",
        "Subject": subject,
    }


def save(fig, stem: str, title: str, subject: str):
    path = FIGURES / f"{stem}.pdf"
    fig.savefig(path, format="pdf", dpi=600, facecolor="white",
                metadata=metadata(title, subject))
    plt.close(fig)
    check(path.is_file() and path.stat().st_size > 1000, f"{stem}: non-empty PDF written")
    return path


def epoch_day(value: int) -> np.datetime64:
    return np.datetime64("1970-01-01") + np.timedelta64(int(value), "D")


def date_string(value: int) -> str:
    return str(epoch_day(value))


# ---------------------------------------------------------------------------
# Shared archived objects: primary fold plus every archived alternative fold.
# ---------------------------------------------------------------------------
RES = ROOT / "results" / "frozen" / "constitutive"
OBJECTS: dict[str, dict[int, dict]] = {c: {} for c in CITIES}
for city in CITIES:
    suffixes = ("", "_f1", "_f2", "_f3", "_f4") if city in ("Porto", "SF") else ("",)
    for suffix in suffixes:
        d = jload(RES / f"obj_{city}{suffix}.json")
        fold = int(d["meta"]["fold"])
        check(d["meta"]["name"] == city, f"{city} fold {fold}: city metadata")
        check(fold not in OBJECTS[city], f"{city} fold {fold}: unique object")
        OBJECTS[city][fold] = d
        fz, fc = d["frames_frozen"], d["frames_complete_rows"]
        for frame in ("P", "D"):
            close(fz[f"R_{frame}"], fc[f"R_{frame}"], 2e-9,
                  f"{city} fold {fold}: frozen/complete-row {frame} consistency")
        dates = d["per_date"]["dates"]
        weights = np.asarray([r["share"] for r in dates], float)
        scores = np.asarray([r["R"] for r in dates], float)
        close(np.sum(weights * scores), fz["R_P"], 2e-9,
              f"{city} fold {fold}: energy-weighted daily P equals pooled R_P")
        check(d["meta"]["kmax"] == (4 if city == "Chengdu" else 12),
              f"{city} fold {fold}: reported harmonic band")

# Porto fold 1--4 files duplicate fold 0 and are not independent splits.
for fold in range(1, 5):
    for key in ("R_P", "R_D", "R_I", "R_CLIM"):
        close(OBJECTS["Porto"][fold]["frames_frozen"][key],
              OBJECTS["Porto"][0]["frames_frozen"][key], 0.0,
              f"Porto duplicate fold {fold}: {key} identical to fold 0")
check(sorted(OBJECTS["SF"]) == [0, 1, 2, 3, 4], "SF: five distinct archived folds")

fold_rows = csvload(RES / "folds.csv")
for row in fold_rows:
    city = row["object"]
    if city not in ("Porto", "SF"):
        continue
    fold = int(row["fold"])
    for frame in ("P", "D"):
        close(float(row[f"R_{frame}"]), OBJECTS[city][fold]["frames_frozen"][f"R_{frame}"],
              2e-9, f"folds.csv cross-check: {city} fold {fold} {frame}")


# ---------------------------------------------------------------------------
# S01: audited joint spectra as 2-D discrete heat maps.
# ---------------------------------------------------------------------------
def make_s01():
    base = ROOT / "results" / "frozen" / "spectra"
    z = nload(base / "verified_joint.npz")
    audit = jload(base / "joint_spectrum_audit.json")
    drawn = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.05), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.895, bottom=0.145, top=0.825,
                        wspace=0.15, hspace=0.30)
    mappable = None
    for row, field in enumerate(("source", "routing")):
        for col, city in enumerate(CITIES):
            q = np.asarray(z[f"{city}_{field}"], float)
            n = 18 if city == "Chengdu" else 24
            expected = (10, 8) if city == "Chengdu" else (13, 8)
            check(q.shape == expected, f"S01 {city} {field}: shape {expected}")
            check(np.isfinite(q).all() and (q > 0).all(), f"S01 {city} {field}: positive finite shares")
            close(q.sum(), audit[city][field]["kept"], 2e-12,
                  f"S01 {city} {field}: audited shown share")
            freq = np.arange(q.shape[0], dtype=float) / n
            if city == "Chengdu":
                check(np.allclose(freq, np.arange(10) / 18.0),
                      "S01 Chengdu: 18-point DFT gives exactly 10 one-sided bins")
            # Exact one-sided support: half-width DC/Nyquist bins, interior midpoints.
            edges = np.concatenate(([0.0], (freq[:-1] + freq[1:]) / 2.0, [0.5]))
            ax = axes[row, col]
            mappable = ax.pcolormesh(np.arange(0.5, 9.0, 1.0), edges, q,
                                     cmap="cividis", norm=LogNorm(vmin=1e-4, vmax=1.0),
                                     edgecolors="white", linewidth=0.18, shading="flat")
            ax.set_xlim(0.5, 8.5)
            ax.set_ylim(0.0, 0.5)
            ax.set_xticks([1, 4, 8])
            ax.set_yticks([0.0, 0.25, 0.5])
            ax.set_yticks(freq, minor=True)
            ax.tick_params(axis="y", which="minor", length=1.2, width=0.35)
            ax.set_yticklabels(["0", "0.25", "0.50"])
            tidy(ax)
            tag(ax, "abcdef"[row * 3 + col])
            if row == 0:
                ax.set_title(f"{NAME[city]}\n" +
                             ("18-point DFT; 10 bins; 06–23 h" if city == "Chengdu"
                              else "24-point DFT; 13 bins; 00–23 h"),
                             fontsize=8.0, pad=5.0, color=COL[city])
            else:
                ax.set_xlabel("Spatial SVD rank k", labelpad=2.0)
            if col == 0:
                ax.set_ylabel(("Source" if row == 0 else "Routing") + "\nfrequency", labelpad=3.0)
            ax.text(0.975, 0.035, f"shown {q.sum():.1%}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=MIN_FONT, color="#222222",
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.82))
            drawn[f"{city}_{field}"] = {
                "shape": list(q.shape), "shown_share": float(q.sum()),
                "frequency_cycles_per_observed_hour": freq.tolist(),
                "minimum_share": float(q.min()), "maximum_share": float(q.max()),
                "color_floor_clipped_cells": int((q < 1e-4).sum()),
            }
    cax = fig.add_axes([0.905, 0.23, 0.016, 0.47])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    cb.set_ticklabels(["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"])
    cb.set_label("Joint energy share (log scale)", fontsize=7.0, labelpad=4.0)
    cb.ax.tick_params(labelsize=MIN_FONT, length=2.0)
    cb.outline.set_linewidth(0.45)
    fig.text(0.49, 0.045, "Temporal frequency: cycles per observed hour; discrete bins, no interpolation",
             ha="center", fontsize=MIN_FONT, color=GREY)
    path = save(fig, "figS01_joint_spectrum_2d", "Three-city joint spectra as 2-D heat maps",
                "Audited multiplicity-weighted TRAIN joint energy shares; source and routing")
    return path, drawn


# ---------------------------------------------------------------------------
# S02: actual per-date P plus fold-level pooled P and D diagnostics.
# ---------------------------------------------------------------------------
def make_s02():
    fig = plt.figure(figsize=(7.0, 4.35))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.19, top=0.86,
                          hspace=0.72, wspace=0.18, height_ratios=[1.10, 1.0])
    drawn = {"daily_primary_fold": {}, "fold_level": {}}
    for j, city in enumerate(CITIES):
        d = OBJECTS[city][0]
        records = d["per_date"]["dates"]
        dates = np.asarray([epoch_day(r["date"]) for r in records])
        score = np.asarray([r["R"] for r in records], float)
        share = np.asarray([r["share"] for r in records], float)
        pooled = float(d["frames_frozen"]["R_P"])
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(dates, score, s=24, color=COL[city], edgecolor="white", linewidth=0.45,
                   alpha=0.92, zorder=3)
        ax.axhline(pooled, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.text(0.98, 0.06, f"pooled P = {pooled:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=MIN_FONT, color=GREY,
                bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.85))
        ax.set_ylim(0.55, 0.76)
        ax.set_yticks([0.55, 0.65, 0.75])
        ax.set_title(f"{NAME[city]}\n{len(records)} TEST dates · " +
                     ("unique split" if city == "Porto" else "fold 0" if city == "SF" else "single split"),
                     fontsize=8.0, color=COL[city], pad=4.0)
        if city == "Porto":
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=21))
        else:
            ax.set_xticks(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        year = str(dates[0])[:4]
        ax.set_xlabel(f"{year} TEST date (month-day)", labelpad=2.0)
        if j == 0:
            ax.set_ylabel("Daily held-out P score", labelpad=3.0)
        tidy(ax, grid=True)
        tag(ax, "abc"[j])
        drawn["daily_primary_fold"][city] = {
            "fold": 0,
            "definition": "per_date.dates[].R; raw P frame; points are not interpolated",
            "pooled_definition": "frames_frozen.R_P = sum_d per_date.share[d] * per_date.R[d]",
            "dates_epoch_day": [int(r["date"]) for r in records],
            "dates_iso": [date_string(r["date"]) for r in records],
            "daily_P": score.tolist(), "energy_share": share.tolist(),
            "pooled_P": pooled,
        }
    ax = fig.add_subplot(gs[1, :])
    ax.set_ylim(0.0, 0.75)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75])
    ax.set_xlim(-0.55, 2.55)
    jitter = {0: 0.0, 1: -0.055, 2: -0.0225, 3: 0.0225, 4: 0.055}
    for i, city in enumerate(CITIES):
        folds = [0] if city in ("Porto", "Chengdu") else [0, 1, 2, 3, 4]
        drawn["fold_level"][city] = []
        for frame, offset, marker in (("P", -0.12, "o"), ("D", 0.12, "D")):
            for fold in folds:
                value = float(OBJECTS[city][fold]["frames_frozen"][f"R_{frame}"])
                primary = fold == 0
                x = i + offset + (jitter[fold] if city == "SF" else 0.0)
                ax.plot([x], [value], marker=marker, ms=5.0, ls="none",
                        mfc=COL[city] if primary else "white", mec=COL[city],
                        mew=1.0 if primary else 0.85, zorder=4)
                drawn["fold_level"][city].append({
                    "fold": fold, "frame": frame, "pooled_score": value,
                    "primary_fold": primary,
                    "n_test_complete_dates": int(OBJECTS[city][fold]["data"]["n_test_complete"]),
                })
        if city == "Porto":
            note = "1 unique split"
        elif city == "SF":
            note = "5 folds"
        else:
            note = "1 split"
        ax.text(i, -0.115, f"{NAME[city]}\n{note}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=MIN_FONT, color=COL[city], linespacing=1.25)
    ax.set_xticks([])
    ax.set_ylabel("Fold-level pooled score", labelpad=3.0)
    tidy(ax, grid=True)
    tag(ax, "d")
    handles = [
        Line2D([], [], marker="o", ls="none", color=GREY, ms=5.0, label="P frame"),
        Line2D([], [], marker="D", ls="none", color=GREY, ms=4.7, label="D departure"),
        Line2D([], [], marker="o", ls="none", mfc=GREY, mec=GREY, ms=5.0, label="fold 0 / only split"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=5.0, label="other SF folds"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False,
              handletextpad=0.35, columnspacing=0.8, borderaxespad=0.2, labelspacing=0.25)
    fig.text(0.5, 0.026,
             "Daily D is not archived for any city; panel d uses real fold-level pooled P/D. "
             "Porto's duplicate folds are excluded.\nNo fold mean is presented as a pooled score.",
             ha="center", va="bottom", fontsize=MIN_FONT, color=GREY)
    path = save(fig, "figS02_heldout_daily_scores", "Held-out daily P scores and fold-level P/D diagnostics",
                "Actual per-date P scores; fold-level pooled P and D; no daily D fabrication")
    return path, drawn


# ---------------------------------------------------------------------------
# S03: transfer gaps, complementing rather than repeating main Figure 4.
# ---------------------------------------------------------------------------
def coherent_values(d, split: str):
    ks = np.asarray(sorted(int(k) for k in d["coherent"]), int)
    alignment, ratio = [], []
    for k in ks:
        rec = d["coherent"][str(k)][split]
        alignment.append(float(rec["truth_vs_op"]["cos"]))
        ratio.append(float(rec["gain_op"]) / float(rec["gain_emp"]))
    return ks, np.asarray(alignment), np.asarray(ratio)


def make_s03():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.86,
                        wspace=0.20, hspace=0.38)
    series, sources = {}, {}
    all_cos, all_ratio = [], []
    for city in CITIES:
        folds = [0] if city in ("Porto", "Chengdu") else [0, 1, 2, 3, 4]
        fold_inputs = []
        if city == "Chengdu":
            response_path = ROOT / "results" / "frozen" / "constitutive" / "chengdu_response18.npz"
            with nload(response_path) as z:
                ks = np.asarray(z["m"], int)
                tr_c = np.asarray(z["train_cos"], float)
                tr_r = np.asarray(z["train_rho"], float)
                te_c = np.asarray(z["cos"], float)
                te_r = np.asarray(z["rho"], float)
                freq = np.asarray(z["freq_cycles_per_observed_hour"], float)
                test_dates = np.asarray(z["test_dates"], int)
            check(np.array_equal(ks, np.arange(10)) and
                  all(a.shape == (10,) for a in (tr_c, tr_r, te_c, te_r, freq)),
                  "S03 Chengdu: TRAIN/TEST response contains all ten bins")
            check(np.allclose(freq, np.arange(10) / 18.0),
                  "S03 Chengdu: frequency is m/18 cycles per observed hour")
            check(np.isfinite(np.column_stack([tr_c, tr_r, te_c, te_r])).all(),
                  "S03 Chengdu: finite TRAIN/TEST response fields")
            check(np.array_equal(test_dates, np.asarray([r["date"] for r in OBJECTS[city][0]["per_date"]["dates"]], int)),
                  "S03 Chengdu: response TEST dates match spectral object")
            fold_inputs.append((0, ks, tr_c, tr_r, ks.copy(), te_c, te_r))
            sources[city] = {"path": response_path.relative_to(ROOT).as_posix(),
                             "bin_definition": "m=0..9 of the observed 18-hour DFT",
                             "frequency_cycles_per_observed_hour": freq.tolist(),
                             "fields": "train_cos/train_rho and TEST cos/rho"}
        else:
            for fold in folds:
                ks, tr_c, tr_r = coherent_values(OBJECTS[city][fold], "train")
                ks2, te_c, te_r = coherent_values(OBJECTS[city][fold], "test")
                fold_inputs.append((fold, ks, tr_c, tr_r, ks2, te_c, te_r))
            sources[city] = {"path": f"results/frozen/constitutive/obj_{city}*.json",
                             "fields": "coherent.train/test truth_vs_op.cos and gain_op/gain_emp"}
        city_rows = []
        for fold, ks, tr_c, tr_r, ks2, te_c, te_r in fold_inputs:
            check(np.array_equal(ks, ks2), f"S03 {city} fold {fold}: matching TRAIN/TEST bins")
            dc, dr = te_c - tr_c, te_r - tr_r
            city_rows.append({"fold": fold, "h": ks.tolist(), "train_alignment": tr_c.tolist(),
                              "test_alignment": te_c.tolist(), "train_norm_ratio": tr_r.tolist(),
                              "test_norm_ratio": te_r.tolist(), "delta_alignment": dc.tolist(),
                              "delta_norm_ratio": dr.tolist()})
            all_cos.extend(dc)
            all_ratio.extend(dr)
        series[city] = city_rows
    lim_c = max(0.2, np.ceil((max(abs(np.asarray(all_cos))) + 0.025) * 10) / 10)
    lim_r = max(0.2, np.ceil((max(abs(np.asarray(all_ratio))) + 0.025) * 10) / 10)
    for j, city in enumerate(CITIES):
        rows = series[city]
        for row_idx, (key, lim, ylabel) in enumerate((("delta_alignment", lim_c, "Δ alignment\nTEST − TRAIN"),
                                                       ("delta_norm_ratio", lim_r, "Δ norm ratio\nTEST − TRAIN"))):
            ax = axes[row_idx, j]
            arr = np.asarray([r[key] for r in rows], float)
            ks = np.asarray(rows[0]["h"], int)
            if len(rows) > 1:
                ax.fill_between(ks, arr.min(axis=0), arr.max(axis=0), color=COL[city], alpha=0.10, lw=0)
                for r in rows:
                    ax.plot(r["h"], r[key], color=COL[city], lw=0.50, alpha=0.30, zorder=2)
                ax.plot(ks, arr.mean(axis=0), "o-", color=COL[city], lw=1.1, ms=3.0,
                        mfc="white", mew=0.75, zorder=4)
            else:
                ax.plot(ks, arr[0], "o-", color=COL[city], lw=1.1, ms=3.0,
                        mfc="white", mew=0.75, zorder=4)
            ax.axhline(0.0, color="#888f96", lw=0.7, ls=(0, (3, 2)), zorder=1)
            if city == "Chengdu":
                ax.set_xlim(-0.3, 9.3)
                ax.set_xticks([0, 3, 6, 9])
            else:
                ax.set_xlim(-0.3, 12.3)
                ax.set_xticks([0, 4, 8, 12])
            ax.set_ylim(-lim, lim)
            ax.set_yticks([-lim, 0.0, lim])
            ax.set_yticklabels([f"{-lim:.1f}", "0", f"{lim:.1f}"])
            tidy(ax, grid=True)
            tag(ax, "abcdef"[row_idx * 3 + j])
            if row_idx == 0:
                sub = ("5 folds; bold = fold mean" if city == "SF" else
                       "18-point DFT; 10 bins" if city == "Chengdu" else "single split")
                ax.set_title(f"{NAME[city]}\n{sub}", fontsize=8.0, color=COL[city], pad=4.0)
            else:
                ax.set_xlabel("18-hour DFT bin m" if city == "Chengdu" else "Clock harmonic h",
                              labelpad=2.0)
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=3.0, linespacing=1.25)
    fig.text(0.5, 0.025,
             "Norm ratio: gain_op/gain_emp (Porto/SF) or archived rho (Chengdu); SF shading = fold range.\n"
             "Chengdu uses all 10 observed-18-hour DFT bins; these are coherent-mode diagnostics, not pooled scores.",
             ha="center", va="bottom", fontsize=MIN_FONT, color=GREY)
    path = save(fig, "figS03_coherent_response_train_test", "Coherent-response TRAIN-to-TEST transfer gaps",
                "TEST-minus-TRAIN alignment and predicted/observed norm-ratio changes; Chengdu uses all ten 18-hour DFT bins")
    return path, {"per_city_fold_series": series, "sources": sources,
                  "difference_from_main_figure_4": "main Figure 4 shows absolute TEST alignment and norm ratio; this supplement shows TEST-minus-TRAIN changes"}


# ---------------------------------------------------------------------------
# S04: three-city empirical CDFs with explicit atoms at zero and one.
# ---------------------------------------------------------------------------
def ecdf(ax, values, color, linestyle, linewidth):
    x = np.sort(np.asarray(values, float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    ax.step(np.concatenate(([0.0], x, [1.0])), np.concatenate(([0.0], y, [1.0])),
            where="post", color=color, ls=linestyle, lw=linewidth, solid_capstyle="butt", zorder=3)


def make_s04():
    z = nload(ROOT / "results" / "frozen" / "anisotropy" / "fo4_fig5_data.npz")
    cd = nload(ROOT / "results" / "frozen" / "anisotropy" / "chengdu_anisotropy.npz")
    cds = jload(ROOT / "results" / "frozen" / "anisotropy" / "chengdu_anisotropy_summary.json")
    values = {c: {key: np.asarray(z[f"{c}_{key}"], float) for key in ("A_C", "A_U")}
              for c in ("Porto", "SF")}
    values["Chengdu"] = {key: np.asarray(cd[f"Chengdu_{key}"], float) for key in ("A_C", "A_U")}
    frame = cds["frames"][cds["primary_frame"]]
    close(np.median(values["Chengdu"]["A_C"]), frame["median_A_C"], 1e-12,
          "S04 Chengdu: primary-frame median A_C")
    close(np.median(values["Chengdu"]["A_U"]), frame["median_A_U"], 1e-12,
          "S04 Chengdu: primary-frame median A_U")
    check(int((values["Chengdu"]["A_C"] == 0).sum()) == frame["zero_response_cells"],
          "S04 Chengdu: zero-response count")
    drawn = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.970, bottom=0.23, top=0.73, wspace=0.16)
    for j, city in enumerate(CITIES):
        ac_raw, au_raw = values[city]["A_C"], values[city]["A_U"]
        check(ac_raw.shape == au_raw.shape and np.isfinite(ac_raw).all() and np.isfinite(au_raw).all(),
              f"S04 {city}: finite paired tensors")
        check(np.all((ac_raw >= -5e-15) & (ac_raw <= 1 + 5e-15) &
                     (au_raw >= -5e-15) & (au_raw <= 1 + 5e-15)),
              f"S04 {city}: anisotropy within [0,1] up to machine roundoff")
        # Some archived one-dimensional tensors evaluate to 1 + 1e-16.
        # Clamp only that floating-point endpoint roundoff, not measured values.
        ac, au = np.clip(ac_raw, 0, 1), np.clip(au_raw, 0, 1)
        clipped = int(np.count_nonzero(ac != ac_raw) + np.count_nonzero(au != au_raw))
        check(np.array_equal(ac == 0, au == 0), f"S04 {city}: shared zero-response mask")
        ax = axes[j]
        ecdf(ax, ac, COL[city], "-", 1.25)
        ecdf(ax, au, COL[city], (0, (2.2, 1.8)), 1.05)
        ax.plot([np.median(ac)], [0.5], "o", ms=3.4, mfc=COL[city], mec="white", mew=0.5, zorder=5)
        ax.plot([np.median(au)], [0.5], "o", ms=3.4, mfc="white", mec=COL[city], mew=0.8, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f"{NAME[city]}\nn = {ac.size}; A = 0: {int((ac == 0).sum())}",
                     fontsize=8.0, color=COL[city], pad=4.0)
        tidy(ax, grid=True)
        tag(ax, "abc"[j])
        if j == 0:
            ax.set_ylabel("Empirical CDF", labelpad=3.0)
        ax.text(0.97, 0.055,
                f"median A_C = {np.median(ac):.3f}\nmedian A_U = {np.median(au):.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=MIN_FONT,
                color=GREY, linespacing=1.3,
                bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none", alpha=0.88))
        drawn[city] = {
            "n_cells": int(ac.size), "zero_cells": int((ac == 0).sum()),
            "one_cells_A_C": int((ac == 1).sum()), "one_cells_A_U": int((au == 1).sum()),
            "median_A_C": float(np.median(ac)), "median_A_U": float(np.median(au)),
            "mean_A_C": float(ac.mean()), "mean_A_U": float(au.mean()),
            "endpoint_roundoff_clipped_values": clipped,
            "endpoint_rule": "np.clip to [0,1] only after all values pass a 5e-15 endpoint tolerance",
        }
    handles = [Line2D([], [], color=GREY, lw=1.25, label="Response A_C"),
               Line2D([], [], color=GREY, lw=1.05, ls=(0, (2.2, 1.8)), label="Road support A_U")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2,
               frameon=False, handletextpad=0.45, columnspacing=1.6)
    fig.text(0.52, 0.045, "Local anisotropy A; all cells retained, including the atoms at A = 0 and A = 1",
             ha="center", fontsize=MIN_FONT, color=GREY)
    path = save(fig, "figS04_anisotropy_distributions", "Three-city anisotropy distributions",
                "Empirical CDFs of response and road-support anisotropy on identical cells")
    return path, drawn


# ---------------------------------------------------------------------------
# S05: matched-capacity media and the archived paired-date contrast.
# ---------------------------------------------------------------------------
def matched_capacity_data():
    rows = csvload(ROOT / "results" / "frozen" / "matched_capacity" / "domain_rankmatched_results.csv")
    pairs = csvload(ROOT / "results" / "frozen" / "matched_capacity" / "domain_rankmatched_pairwise.csv")
    cd = jload(ROOT / "results" / "frozen" / "anisotropy" / "partF_summary.json")
    out = {}
    for city, label in (("Porto", "Porto"), ("SF", "San Francisco")):
        d = {}
        for r in rows:
            if r["city"] == label:
                d.setdefault(r["source"], {})[int(r["medium_dof"])] = float(r["R2_cond"])
        nint = max(d["road"])
        base = d["road"][1]
        check(d["mobility"][1] == base and d["mobility_shuf"][1] == base,
              f"S05 {city}: constant-medium R=1 anchor")
        pair = [r for r in pairs if r["city"] == label and r["contrast"] == "mobility - road"
                and int(r["R"]) == nint]
        check(len(pair) == 1, f"S05 {city}: unique mobility-road interval")
        p = pair[0]
        rep = cd[city]
        check(rep["reproduction"]["max_abs_diff"] < 1e-9 and
              rep["reproduction"]["pairwise_max_abs_diff"] < 1e-9,
              f"S05 {city}: independent reproduction gate")
        close(rep["D0_R2"], base, 1e-12, f"S05 {city}: baseline across readers")
        for source in ("road", "mobility", "mobility_shuf"):
            close(rep["anchors_R_eq_n_int"][source], d[source][nint], 1e-12,
                  f"S05 {city}: selected {source} across readers")
        out[city] = {
            "n_int": nint, "constant_R2": base,
            "road": d["road"][nint] - base,
            "mobility": d["mobility"][nint] - base,
            "permuted": d["mobility_shuf"][nint] - base,
            "gap": float(p["delta_mean"]),
            "ci95": [float(p["ci95_lo"]), float(p["ci95_hi"])],
            "n_dates": int(p["n_dates"]), "n_boot": int(p["n_boot"]),
            "channel": rep["frozen_registry"]["channel"], "basis": rep["frozen_registry"]["basis"],
        }
    c = cd["Chengdu"]
    nint = int(c["n_int"])
    d = {r["source"]: float(r["R2_cond"]) for r in c["results"] if int(r["R"]) == nint}
    base = float(c["D0_R2"])
    for source, value in c["anchors_R_eq_n_int"].items():
        close(value, d[source], 1e-12, f"S05 Chengdu: selected {source} anchor")
    p = [r for r in c["pairwise"] if r["contrast"] == "mobility - road" and int(r["R"]) == nint]
    check(len(p) == 1, "S05 Chengdu: unique mobility-road interval")
    p = p[0]
    out["Chengdu"] = {
        "n_int": nint, "constant_R2": base,
        "road": d["road"] - base, "mobility": d["mobility"] - base,
        "permuted": d["mobility_shuf"] - base,
        "gap": float(p["delta_mean"]), "ci95": [float(p["ci95_lo"]), float(p["ci95_hi"])],
        "n_dates": int(p["n_dates"]), "n_boot": int(p["n_boot"]),
        "channel": c["frozen_registry"]["channel"], "basis": c["frozen_registry"]["basis"],
    }
    for city in CITIES:
        close(out[city]["gap"], out[city]["mobility"] - out[city]["road"], 2e-9,
              f"S05 {city}: mobility-road equals point difference")
    return out


def make_s05():
    data = matched_capacity_data()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 2.70), sharey=True,
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.245, top=0.72, wspace=0.12)
    for city in CITIES:
        i = 2 - CITIES.index(city)
        v = data[city]
        axl.plot([v["road"], v["mobility"]], [i, i], color="#e1e5e9", lw=1.8,
                 solid_capstyle="round", zorder=1)
        axl.plot([v["permuted"]], [i], "v", ms=5.0, color="#969da4", mec="white", mew=0.55, zorder=4)
        axl.plot([v["road"]], [i], "s", ms=4.8, color=COL[city], mec="white", mew=0.55, zorder=5)
        axl.plot([v["mobility"]], [i], "o", ms=5.0, color=COL[city], mec="white", mew=0.55, zorder=5)
        lo, hi = v["ci95"]
        axr.plot([lo, hi], [i, i], color=COL[city], lw=2.0, solid_capstyle="butt", zorder=3)
        axr.plot([lo, lo], [i - 0.09, i + 0.09], color=COL[city], lw=0.9, zorder=3)
        axr.plot([hi, hi], [i - 0.09, i + 0.09], color=COL[city], lw=0.9, zorder=3)
        axr.plot([v["gap"]], [i], "o", ms=3.8, mfc="white", mec=COL[city], mew=1.1, zorder=5)
    axl.axvline(0.0, color="#777f87", lw=0.75, ls=(0, (3, 2)), zorder=1)
    axl.set_xlim(-0.065, 0.105)
    axl.set_xticks([-0.05, 0.0, 0.05, 0.10])
    axl.set_xlabel("Gain in conditional R² over constant medium", labelpad=3.0)
    axl.set_ylim(-0.60, 2.60)
    axl.set_yticks([2, 1, 0])
    axl.set_yticklabels([NAME[c] for c in CITIES])
    axl.set_title("a  Three media at matched capacity", loc="left", fontsize=8.0,
                  fontweight="bold", pad=4.0)
    axr.set_xlim(0.0, 0.12)
    axr.set_xticks([0.0, 0.05, 0.10])
    axr.set_xlabel("Mobility − road\npaired-date 95% interval", labelpad=3.0, linespacing=1.3)
    axr.set_title("b  Archived contrast", loc="left", fontsize=8.0, fontweight="bold", pad=4.0)
    axr.tick_params(axis="y", which="both", left=False, labelleft=False)
    tidy(axl, grid=True)
    tidy(axr, grid=True)
    handles = [Line2D([], [], marker="v", ls="none", color="#969da4", ms=5.0,
                      label="permuted mobility"),
               Line2D([], [], marker="s", ls="none", color=GREY, ms=4.8, label="road medium"),
               Line2D([], [], marker="o", ls="none", color=GREY, ms=5.0, label="mobility-inferred")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.975), ncol=3,
               frameon=False, handletextpad=0.4, columnspacing=1.1)
    fig.text(0.53, 0.035,
             "R = n_int: 1984 / 5512 / 7080; intervals resample 51 / 15 / 4 held-out dates "
             "(Porto / SF / Chengdu). No single-media uncertainty is implied.",
             ha="center", fontsize=MIN_FONT, color=GREY)
    path = save(fig, "figS05_matched_capacity", "Three-city matched-capacity media",
                "Conditional R-squared gains and paired mobility-minus-road intervals at matched capacity")
    return path, data


# ---------------------------------------------------------------------------
# S06: observed and reconstructed physical fields in all three cities.
# ---------------------------------------------------------------------------
def snapshot_data():
    artifacts = ROOT / "results" / "frozen" / "crosscity" / "artifacts"
    meta = jload(artifacts / "routing_snapshot_meta.json")["cities"]
    fam = jload(ROOT / "results" / "frozen" / "crosscity" / "cc_families.json")["cities"]
    native = jload(ROOT / "results" / "frozen" / "crosscity" / "cc_native_paired.json")["cities"]
    cd_manifest = jload(ROOT / "results" / "frozen" / "constitutive" / "chengdu_frozen_response_manifest.json")
    out = {}
    compact = ROOT / "results" / "frozen" / "reconstruction" / "reconstruction.npz"
    for city in ("Porto", "SF"):
        if compact.is_file():
            with nload(compact) as z:
                obs = np.asarray(z[f"{city}__obs"], float)
                hat = np.asarray(z[f"{city}__hat"], float)
                i = int(z[f"{city}__index"])
                phase = int(z[f"{city}__phase"])
                picked_date = int(z[f"{city}__test_date"])
            check(picked_date == int(meta[city]["picked_date"]) and
                  phase == int(meta[city]["picked_phase"]),
                  f"S06 {city}: compact picked date and phase")
        else:
            with nload(artifacts / f"routing_snapshot_{city}.npz") as z:
                i, phase = int(z["picked_i"]), int(z["picked_phase"])
                obs = np.asarray(z["date24"][i, phase], float)
                hat = np.asarray(z["hat24"][i, phase], float)
                dates = np.asarray(z["test_dates"], int)
                check(np.array_equal(np.asarray(z["date24"][i, phase]), np.asarray(z["proj24"][i, phase])),
                      f"S06 {city}: observed array is the road-subspace projection")
                picked_date = int(z["picked_date"])
            check(picked_date == int(dates[i]) == int(meta[city]["picked_date"]) and
                  phase == int(meta[city]["picked_phase"]),
                  f"S06 {city}: archived picked date and phase")
        m = meta[city]
        check(m["abs_err"] < 1e-8, f"S06 {city}: snapshot reproduction gate")
        close(m["E_test_sel"], fam[city]["folds"][0]["arms"]["full"]["E_test_sel"], 1e-12,
              f"S06 {city}: full-arm frozen score")
        close(np.max(np.abs(np.asarray(m["E_per_date"], float) -
                            np.asarray(native[city][0]["families"]["in_graph_low"]["E_per_date_full"], float))),
              0.0, 1e-12, f"S06 {city}: per-date archived error vector")
        close(m["picked_date_E"], m["E_per_date"][i], 1e-12,
              f"S06 {city}: picked-date error matches picked_i")
        close(m["one_minus_E"], OBJECTS[city][0]["frames_frozen"]["R_P"], 2e-9,
              f"S06 {city}: P-frame score agrees with spectral object")
        out[city] = dict(obs=obs, hat=hat, index=i, phase=phase, date=picked_date,
                         frame="P", score=float(m["one_minus_E"]), source_meta=m)
    with nload(ROOT / "results" / "frozen" / "constitutive" / "chengdu_snapshot18.npz") as z:
        i, phase = int(z["picked_i"]), int(z["picked_phase"])
        obs = np.asarray(z["date18"][i, phase], float)
        hat = np.asarray(z["hat18"][i, phase], float)
        dates = np.asarray(z["test_dates"], int)
        r_daily = np.asarray(z["per_date_R_P"], float)
        check("per_date_R_P" in z.files and "per_date_R_D" not in z.files,
              "S06 Chengdu: latest archive stores P-frame daily scores only")
        check(np.array_equal(np.asarray(z["date18"][i, phase]), np.asarray(z["proj18"][i, phase])),
              "S06 Chengdu: observed array is the road-subspace projection")
        picked_date = int(z["picked_date"])
    expected_i = int(np.argmin(np.abs(r_daily - np.median(r_daily))))
    check(i == expected_i and phase == 6 and picked_date == int(dates[i]),
          "S06 Chengdu: median-P-error date and 12:00 phase 6")
    close(np.max(np.abs(r_daily - np.asarray([r["R"] for r in OBJECTS["Chengdu"][0]["per_date"]["dates"]], float))),
          0.0, 2e-9, "S06 Chengdu: daily P scores agree with spectral object")
    close(cd_manifest["R_P"], OBJECTS["Chengdu"][0]["frames_frozen"]["R_P"], 2e-9,
          "S06 Chengdu: P-frame score agrees with spectral object")
    close(cd_manifest["R_D"], OBJECTS["Chengdu"][0]["frames_frozen"]["R_D"], 2e-9,
          "S06 Chengdu: independent D-only cross-check")
    check(cd_manifest["snapshot_rule"].startswith("P-frame readout") and
          cd_manifest["snapshot_rule"].endswith("12:00 (phase 6 of the 06--23 h window)"),
          "S06 Chengdu: exporter confirms P-frame and 12:00 selection rule")
    out["Chengdu"] = dict(obs=obs, hat=hat, index=i, phase=phase, date=int(dates[i]),
                          frame="P", score=float(cd_manifest["R_P"]),
                          daily_R_P=r_daily.tolist(), source_meta=cd_manifest)
    return out


def make_s06():
    snap = snapshot_data()
    with nload(ROOT / "routing_urban_medium_constitutive" / "results" / "um_ctx.npz") as z:
        geometry = {}
        for city in CITIES:
            nx, ny = int(z[f"{city}__nx"]), int(z[f"{city}__ny"])
            p0 = np.asarray(z[f"{city}__face_pairs_p0"], int)
            p1 = np.asarray(z[f"{city}__face_pairs_p1"], int)
            check(len(p0) == len(p1) == snap[city]["obs"].size == snap[city]["hat"].size,
                  f"S06 {city}: face-pair geometry matches snapshot vectors")
            geometry[city] = (nx, ny, p0, p1)
    # All maps share one physical height; widths retain their true grid aspect ratios.
    mh = 1.42
    gap_pair, gap_city, margin = 0.020, 0.100, 0.080
    widths = {c: mh * (geometry[c][0] / geometry[c][1]) for c in CITIES}
    block_w = {c: 2 * widths[c] + gap_pair for c in CITIES}
    W = sum(block_w.values()) + 2 * gap_city + 2 * margin
    H = 2.55
    fig = plt.figure(figsize=(W, H))
    drawn = {}
    x = margin
    for ci, city in enumerate(CITIES):
        nx, ny, p0, p1 = geometry[city]
        dx = 0.25
        x0 = (p0 % nx + 0.5) * dx
        y0 = (p0 // nx + 0.5) * dx
        x1 = (p1 % nx + 0.5) * dx
        y1 = (p1 // nx + 0.5) * dx
        obs, hat = snap[city]["obs"], snap[city]["hat"]
        check(np.isfinite(obs).all() and np.isfinite(hat).all(), f"S06 {city}: finite drawn fields")
        vmax = float(np.quantile(np.abs(obs), 0.99))
        check(vmax > 0, f"S06 {city}: positive opacity scale")
        # Same observed-selected sites in the observed and reconstructed panels.
        stride = max(1, int(round(np.sqrt(len(obs) / 55.0))))
        owner_i, owner_j = np.divmod(p0, nx)
        keep = (owner_i % stride == 0) & (owner_j % stride == 0) & (np.abs(obs) > 0.08 * vmax)
        check(8 <= int(keep.sum()) <= 180, f"S06 {city}: legible shared glyph count")
        phase = snap[city]["phase"]
        clock = f"{phase + 6:02d}:00" if city == "Chengdu" else f"{phase:02d}:00"
        for pi, (label, values) in enumerate((("Obs.", obs), ("Recon.", hat))):
            ax = fig.add_axes([(x + pi * (widths[city] + gap_pair)) / W, 0.644 / H,
                               widths[city] / W, mh / H])
            norm = np.clip(np.abs(values) / vmax, 0, 1)
            segments = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1])], axis=1)
            rgba = np.tile(to_rgba(COL[city]), (len(values), 1))
            rgba[:, 3] = 0.82 * np.sqrt(norm)
            lc = LineCollection(segments, colors=rgba, linewidths=0.38 + 0.45 * norm,
                                capstyle="round", zorder=2)
            ax.add_collection(lc)
            signs = np.sign(values[keep])
            qx = 0.5 * (x0[keep] + x1[keep])
            qy = 0.5 * (y0[keep] + y1[keep])
            qu = 0.5 * (x1[keep] - x0[keep]) * signs
            qv = 0.5 * (y1[keep] - y0[keep]) * signs
            # Uniform on-paper arrow length within each city; opacity carries magnitude.
            scale = (0.25 * (widths[city] / (nx * dx))) / 0.070
            ax.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy", scale=scale,
                      color=COL[city], width=0.0045, headwidth=3.2, headlength=4.0,
                      headaxislength=3.5, pivot="mid", alpha=0.90, zorder=4)
            ax.set_xlim(0, nx * dx)
            ax.set_ylim(0, ny * dx)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#aab1b8")
                spine.set_linewidth(0.45)
            ax.text(0.0 if pi == 0 else 1.0, 1.025, label, transform=ax.transAxes,
                    ha="left" if pi == 0 else "right", va="bottom",
                    fontsize=MIN_FONT, color=GREY)
            tag(ax, "abcdef"[ci * 2 + pi])
            if pi == 0:
                ax.plot([0.35, 2.35], [0.35, 0.35], color="#30343a", lw=1.0, zorder=7)
                ax.text(1.35, 0.50, "2 km", ha="center", va="bottom", fontsize=MIN_FONT,
                        color="#30343a", zorder=7,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75))
        center = x + block_w[city] / 2
        fig.text(center / W, 0.990, NAME[city], ha="center", va="top", fontsize=8.0,
                 fontweight="bold", color=COL[city])
        fig.text(center / W, 0.940, f"{snap[city]['frame']} · {clock}", ha="center", va="top",
                 fontsize=MIN_FONT, color=GREY)
        fig.text(center / W, 0.895, date_string(snap[city]["date"]), ha="center", va="top",
                 fontsize=MIN_FONT, color=GREY)
        drawn[city] = {
            "date_epoch_day": snap[city]["date"], "date_iso": date_string(snap[city]["date"]),
            "picked_i": snap[city]["index"], "picked_phase": phase, "clock": clock,
            "frame": snap[city]["frame"], "score": snap[city]["score"],
            "score_label": "R_P",
            "grid_nx": nx, "grid_ny": ny, "n_faces": int(len(obs)),
            "picked_rule": ("median archived per-date P error" if city == "Chengdu"
                             else "archived picked_i/picked_phase preserved; not reselected"),
            "opacity_scale_p99_observed": vmax, "glyph_stride": stride,
            "shared_glyph_faces": int(keep.sum()), "observed_label": "date24/date18 road-subspace field",
            "reconstructed_label": "hat24/hat18; raw24/raw18 not drawn",
            "observed_energy": float(np.sum(obs ** 2)), "reconstructed_energy": float(np.sum(hat ** 2)),
        }
        if city == "Chengdu":
            drawn[city]["daily_R_P"] = snap[city]["daily_R_P"]
            drawn[city]["P_frame_used"] = True
            drawn[city]["D_role"] = "independent cross-check only; not used for selection or drawing"
        x += block_w[city] + gap_city
    fig.text(0.5, 0.125,
             "Obs. and Recon. share each city's observed-p99 opacity scale and the same observed-selected arrow sites.",
             ha="center", va="center", fontsize=MIN_FONT, color=GREY)
    fig.text(0.5, 0.060,
             "All cities: P-frame reconstructions from raw TEST inputs/targets; D is retained only as an independent check.",
             ha="center", va="center", fontsize=MIN_FONT, color=GREY)
    path = save(fig, "figS06_heldout_reconstruction", "Three-city held-out physical-field reconstruction",
                "Observed and reconstructed road-subspace fields; uniform P-frame raw TEST inputs and targets")
    return path, drawn


# ---------------------------------------------------------------------------
# S07: explicitly exploratory Chengdu full-FACE nonlinear diagnostics.
# ---------------------------------------------------------------------------
def make_s07():
    base = ROOT / "results" / "frozen" / "baselines" / "chengdu_nonlinear"
    result_path = base / "chengdu_nonlinear_exploratory.json"
    result = jload(result_path)
    validation = jload(base / "validation.json")
    config_path = track(ROOT / "configs" / "experiments" / "chengdu_nonlinear.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    check(result["status"] == "COMPLETE_VALIDATED_EXPLORATORY" and result["validation"]["all_pass"],
          "S07: experiment status is complete and validated")
    check(validation["all_pass"] and len(validation["checks"]) == 16 and
          all(v["pass"] for v in validation["checks"].values()),
          "S07: all 16 archived experiment validations pass")
    check(validation["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest(),
          "S07: validation file points to the exact result hash")
    check(validation["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest() ==
          result["config"]["sha256"], "S07: config hashes agree across result and validation")
    check(config["experiment"].startswith("Chengdu 18-phase full-FACE") and
          result["no_absence_of_nonlinearity_claim"],
          "S07: full-FACE exploratory scope and no-absence caveat")
    check(result["A"]["n_test_dates"] == result["D"]["n_test_dates"] == 3 and
          validation["statistical_limitation"]["n_test_dates"] == 3,
          "S07: exactly three TEST dates")
    check(result["D"]["decision"] == "UNDERPOWERED_EXPLORATORY",
          "S07: underpowered exploratory decision retained")

    arms = ["M_zero", "M_shared", "M_shared_bal", "M_freq", "M_pool"]
    full_a = result["A"]["full_face"]
    check(set(arms) == set(full_a["E"]) == set(full_a["per_date"]),
          "S07 A: all five full-FACE arms present")
    a_records = {}
    for arm in arms:
        pooled = float(full_a["E"][arm])
        daily = np.asarray(full_a["per_date"][arm], float)
        check(daily.shape == (3,) and np.isfinite(daily).all() and np.isfinite(pooled),
              f"S07 A {arm}: three finite daily errors and one pooled error")
        a_records[arm] = {"pooled_full_FACE_error": pooled,
                          "daily_full_FACE_errors": daily.tolist(),
                          "aggregation": "pooled archive value is not replaced by a daily mean"}

    full_d = result["D"]["full_face"]
    d_records = {}
    lin_daily = np.asarray(full_d["per_date"]["linF"], float)
    for arm in ("quadF", "krrF"):
        summary = full_d["summary"][arm]
        daily = np.asarray(full_d["per_date"][arm], float)
        dR2 = float(summary["dR2"])
        ci = np.asarray(summary["ci"], float)
        check(daily.shape == (3,) and ci.shape == (2,) and summary["n"] == 3,
              f"S07 D {arm}: three dates and one 95% interval")
        close(np.mean(lin_daily) - np.mean(daily), dR2, 2e-12,
              f"S07 D {arm}: dR2 recomputed from full-FACE daily errors")
        d_records[arm] = {"dR2": dR2, "ci95": ci.tolist(),
                          "daily_full_FACE_errors": daily.tolist(),
                          "linF_daily_full_FACE_errors": lin_daily.tolist(),
                          "frac_improved": float(summary["frac_improved"])}
    check(d_records["quadF"]["ci95"][1] < 0, "S07 D quadF: archived interval is negative")
    check(d_records["krrF"]["ci95"][0] < 0 < d_records["krrF"]["ci95"][1],
          "S07 D krrF: archived interval crosses zero")

    fig, (axa, axd) = plt.subplots(1, 2, figsize=(7.0, 2.90),
                                   gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.235, top=0.735, wspace=0.34)
    green = COL["Chengdu"]
    for j, arm in enumerate(arms):
        y = len(arms) - 1 - j
        daily = np.asarray(a_records[arm]["daily_full_FACE_errors"], float)
        pooled = a_records[arm]["pooled_full_FACE_error"]
        color = "#7c838a" if arm == "M_zero" else green
        axa.scatter(daily, np.full(3, y) + np.arange(-1, 2) * 0.09, s=15,
                    facecolor="white", edgecolor=color, linewidth=0.65, alpha=0.72, zorder=3)
        axa.scatter([pooled], [y], s=39, color=color, edgecolor="white", linewidth=0.55, zorder=5)
        axa.text(pooled * 1.045, y, f"{pooled:.4f}" if arm != "M_zero" else "1.000",
                 ha="left", va="center", fontsize=MIN_FONT, color=color)
    axa.set_xscale("log")
    axa.set_xlim(0.065, 1.26)
    axa.set_xticks([0.07, 0.10, 0.20, 0.50, 1.00])
    axa.set_xticklabels(["0.07", "0.10", "0.20", "0.50", "1.00"])
    axa.set_ylim(-0.65, 4.65)
    axa.set_yticks([4, 3, 2, 1, 0])
    axa.set_yticklabels(arms)
    axa.set_xlabel("full-FACE TEST error (log scale; lower is better)", labelpad=3.0)
    axa.set_title("a  A frequency arms: full-FACE errors", loc="left", fontsize=8.0,
                  fontweight="bold", pad=4.0)
    tidy(axa, grid=True)
    axa.grid(axis="x", color="#dfe3e7", lw=0.45, ls=(0, (2, 2)), zorder=0)
    axa.grid(axis="y", visible=False)

    for y, arm in ((1, "quadF"), (0, "krrF")):
        rec = d_records[arm]
        lo, hi = rec["ci95"]
        value = rec["dR2"]
        axd.plot([lo, hi], [y, y], color=green, lw=2.0, solid_capstyle="butt", zorder=3)
        for x in (lo, hi):
            axd.plot([x, x], [y - 0.09, y + 0.09], color=green, lw=0.9, zorder=3)
        axd.plot([value], [y], "o", ms=4.5, mfc="white", mec=green, mew=1.1, zorder=5)
        axd.text(-0.00645, y + 0.23,
                 f"{value:+.5f}  [{lo:+.5f}, {hi:+.5f}]",
                 ha="left", va="bottom", fontsize=MIN_FONT, color=GREY)
    axd.axvline(0.0, color="#777f87", lw=0.75, ls=(0, (3, 2)), zorder=1)
    axd.set_xlim(-0.0067, 0.00115)
    axd.set_xticks([-0.006, -0.004, -0.002, 0.0])
    axd.set_xticklabels(["-0.006", "-0.004", "-0.002", "0"])
    axd.set_ylim(-0.65, 1.65)
    axd.set_yticks([1, 0])
    axd.set_yticklabels(["quadF", "krrF"])
    axd.set_xlabel("full-FACE ΔR² vs linF\npositive favors nonlinear",
                   labelpad=3.0, linespacing=1.3)
    axd.set_title("b  D: full-FACE increments", loc="left", fontsize=8.0,
                  fontweight="bold", pad=4.0)
    tidy(axd, grid=True)

    handles = [Line2D([], [], marker="o", ls="none", color=green, ms=5.2,
                      label="pooled archived TEST error"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=green, ms=4.5,
                      label="three TEST dates")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.895), ncol=2,
               frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.text(0.5, 0.970, "Chengdu · exploratory; 3 TEST dates; underpowered",
             ha="center", va="top", fontsize=8.0, fontweight="bold", color="#8b542f")
    fig.text(0.5, 0.040,
             "full-FACE retains out-of-POD energy; these are not inside-POD scores.\n"
             "Descriptive only: no verdict of nonlinearity absence or frequency-flatness is supported.",
             ha="center", va="bottom", fontsize=MIN_FONT, color=GREY, linespacing=1.35)
    path = save(fig, "figS07_nonlinear_exploratory", "Chengdu exploratory nonlinear diagnostics",
                "Exploratory full-FACE frequency-arm errors and nonlinear increments; three TEST dates; underpowered")
    return path, {
        "scope": "Chengdu only; exploratory; 3 TEST dates; underpowered; no cross-city or no-nonlinearity inference",
        "full_FACE_definition": full_a["definition"],
        "POD_policy": "Only full-FACE values are drawn. Inside-TRAIN-POD scores are archived separately and are not mixed into either panel.",
        "A": a_records,
        "D": d_records,
        "D_definition": result["config_protocol"]["D"]["increment"],
        "archived_validation_checks": len(validation["checks"]),
        "statistical_limitation": validation["statistical_limitation"],
    }


def main():
    outputs = []
    figure_records = {}
    for maker in (make_s01, make_s02, make_s03, make_s04, make_s05, make_s06, make_s07):
        path, record = maker()
        outputs.append(path)
        figure_records[path.stem] = record
        print(f"wrote {path}", flush=True)
    out_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
    manifest = {
        "schema": "threecity-supplement-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": ".",
        "scope": "Only the figure output directory and result manifest are written; frozen inputs are read-only.",
        "cities": list(CITIES),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "style": {"pdf_fonttype": 42, "configured_minimum_font_pt": MIN_FONT,
                  "vector_pdf": True, "city_colors": COL},
        "inputs_sha256": dict(sorted(INPUT_HASHES.items())),
        "outputs_sha256": out_hashes,
        "figures": {
            "figS01_joint_spectrum_2d": {
                "data_definition": "verified_joint.npz multiplicity-weighted full-band TRAIN joint energy shares, plotted as discrete 2-D heat maps; x = frequency-specific SVD rank, y = one-sided DFT frequency per observed hour.",
                "chengdu_frequency": "18 observed hours (06--23), 18-point DFT, bins m=0..9 at m/18 cycles per observed hour; no nighttime zero filling.",
                "records": figure_records["figS01_joint_spectrum_2d"],
            },
            "figS02_heldout_daily_scores": {
                "data_definition": "Top: obj_*.json per_date.dates[].R (daily raw P) for fold 0/only split; dashed line is energy-weighted pooled R_P. Bottom: each fold's frames_frozen pooled R_P and R_D.",
                "daily_D_policy": "No city archives daily D. The real available fold-level pooled D diagnostic is used instead; no daily D values are invented.",
                "fold_policy": "SF uses folds 0--4. Porto folds 1--4 exactly duplicate fold 0 and are excluded as independent evidence. No fold mean is called pooled.",
                "records": figure_records["figS02_heldout_daily_scores"],
            },
            "figS03_coherent_response_train_test": {
                "data_definition": "For each bin and fold: delta_alignment = TEST cos - TRAIN cos; delta_norm_ratio = TEST norm ratio - TRAIN norm ratio. Porto/SF use coherent.{train,test} and gain_op/gain_emp; Chengdu uses chengdu_response18.npz train_cos/train_rho and TEST cos/rho for every m=0..9.",
                "chengdu_frequency": "All ten bins of the observed 18-hour DFT at m/18 cycles per observed hour; the old obj_Chengdu h<=4 display and not-reported region are not used.",
                "aggregation": "SF thin lines are five folds, bold is their arithmetic mean and shading their range; Porto/Chengdu are single-split curves. These are coherent-mode summaries, not pooled prediction scores.",
                "records": figure_records["figS03_coherent_response_train_test"],
            },
            "figS04_anisotropy_distributions": {
                "data_definition": "ECDF of A_C (response) and A_U (road support) on identical full cell sets; A=(lambda1-lambda2)/(lambda1+lambda2). Zero-response cells remain in the distributions.",
                "frame": "Chengdu primary frame D; frame P is not substituted.",
                "records": figure_records["figS04_anisotropy_distributions"],
            },
            "figS05_matched_capacity": {
                "data_definition": "At R=n_int, conditional R2(source)-conditional R2(constant R=1) for permuted mobility, road and mobility-inferred media; separate axis is archived paired-date mobility-road mean and 95% bootstrap interval.",
                "limitations": "Chengdu uses default channel, perfold basis and four held-out dates; SF uses passenger channel. Intervals exist only for the contrast, not individual media.",
                "records": figure_records["figS05_matched_capacity"],
            },
            "figS06_heldout_reconstruction": {
                "data_definition": "Observed date24/date18 and reconstructed hat24/hat18 road-subspace fields at archived picked_i/picked_phase; raw24/raw18 are not drawn. Opacity uses each city's own observed p99 and is not an absolute cross-city scale.",
                "frame_policy": "All three cities use P-frame reconstructions from raw TEST inputs/targets. Chengdu hat18 is the P-frame pred_p readout; D is used only for an independent cross-check and not for selection or drawing.",
                "selection": "Porto/SF preserve their archived picked_i/picked_phase without reselection. Chengdu uses the median per-date P-error date (picked_i=2, picked_date=16304); phase 6 in the 06--23 window is 12:00.",
                "records": figure_records["figS06_heldout_reconstruction"],
            },
            "figS07_nonlinear_exploratory": {
                "data_definition": "Chengdu-only exploratory diagnostics. A: full-FACE pooled TEST errors for M_zero/M_shared/M_shared_bal/M_freq/M_pool plus the three daily errors. D: full-FACE dR2 and 95% date-bootstrap intervals for quadF/krrF relative to linF.",
                "scope": "exploratory; 3 TEST dates; underpowered. Only full-FACE values are drawn; inside-POD scores are not substituted. No absence-of-nonlinearity or confirmatory frequency-flat verdict is supported.",
                "records": figure_records["figS07_nonlinear_exploratory"],
            },
        },
        "checks_passed": CHECKS,
        "check_count": len(CHECKS),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(CHECKS)} checks; {len(INPUT_HASHES)} hashed inputs)", flush=True)


if __name__ == "__main__":
    main()
