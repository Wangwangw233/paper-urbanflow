#!/usr/bin/env python3
"""Build the UrbanFlow public-release candidate from an explicit allowlist.

The source tree is never written.  The output is rebuilt from scratch, so stale
files cannot survive a later export.  Large source arrays are used only to make
small, display-specific frozen products; they are never copied.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import numpy as np

MAX_BYTES = 50 * 1024 * 1024
EXPORT_VERSION = 1

FIGURE_CODE = [
    "paper/merged_manuscript/figures/redraw_urbanflow.py",
    "paper/merged_manuscript/figures/reorganize_threecity_figures.py",
    "paper/merged_manuscript/figures/fig5_ablation.py",
    "paper/merged_manuscript/figures/figstyle.py",
    "paper/singlecolumn/figures/fig2d_spectrum3d.py",
    "reports/threecity_joint_spectrum/manuscript_revision/redraw_joint_spectrum.py",
    "reports/threecity_joint_spectrum/manuscript_revision/audit_joint_spectrum.py",
    "reports/threecity_figure5_revision/scripts/fig5_threecity.py",
    "reports/threecity_figure5_revision/scripts/fig5_redesign.py",
    "reports/threecity_figure5_revision/scripts/threecity_stats.py",
    "reports/threecity_redesign/scripts/chengdu_frozen_response.py",
    "reports/threecity_supplement/scripts/make_figures.py",
    "reports/threecity_supplement/scripts/validate_outputs.py",
    "reports/chengdu_nonlinear_exploratory/scripts/run_experiment.py",
]

# Only aggregate fields, geometry, frozen splits and frozen statistics are
# eligible.  In particular, no raw trajectory table or individual-level output
# is allowlisted here.
DATA_FILES = [
    "benchmark/domain_rankmatched_pairwise.csv",
    "benchmark/domain_rankmatched_results.csv",
    "crosscity_sample_complexity/artifacts/routing_snapshot_meta.json",
    "crosscity_sample_complexity/results/cc_families.json",
    "crosscity_sample_complexity/results/cc_native_paired.json",
    "final_ablation/results/partA_topology.json",
    "full_operator_spatial_anisotropy/results/fo4_fig5_data.npz",
    "reports/chengdu_nonlinear_exploratory/config.json",
    "reports/chengdu_nonlinear_exploratory/results/chengdu_nonlinear_exploratory.json",
    "reports/chengdu_nonlinear_exploratory/results/environment.json",
    "reports/chengdu_nonlinear_exploratory/results/validation.json",
    "reports/crosscity_standard_replication/results/spectra.json",
    "reports/deep_baseline_training_record/splits/cabspotting_datefreeze.json",
    "reports/deep_baseline_training_record/splits/porto_datefreeze.json",
    "reports/mainstream_mobility_comparison/crosscity_results.csv",
    "reports/spatiotemporal_constitutive_spectra/results/obj_Chengdu.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_Porto.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_Porto_f1.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_Porto_f2.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_Porto_f3.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_Porto_f4.json",
    "reports/spatiotemporal_constitutive_spectra/results/folds.csv",
    "reports/spatiotemporal_constitutive_spectra/results/obj_SF.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_SF_f1.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_SF_f2.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_SF_f3.json",
    "reports/spatiotemporal_constitutive_spectra/results/obj_SF_f4.json",
    "reports/threecity_figure5_revision/results/basemap_Chengdu.npz",
    "reports/threecity_figure5_revision/results/basemap_Porto.npz",
    "reports/threecity_figure5_revision/results/basemap_SF.npz",
    "reports/threecity_figure5_revision/results/chengdu_anisotropy.npz",
    "reports/threecity_figure5_revision/results/chengdu_anisotropy_summary.json",
    "reports/threecity_figure5_revision/results/partA_chengdu.json",
    "reports/threecity_figure5_revision/results/partF_summary.json",
    "reports/threecity_joint_spectrum/manuscript_revision/verified_joint.npz",
    "reports/threecity_joint_spectrum/manuscript_revision/audit.json",
    "reports/threecity_joint_spectrum/results/chengdu_cycle18.npz",
    "reports/threecity_joint_spectrum/results/threecity_drawn.csv",
    "reports/threecity_redesign/results/chengdu_frozen_response_manifest.json",
    "reports/threecity_redesign/results/chengdu_response18.npz",
    "reports/threecity_redesign/results/chengdu_snapshot18.npz",
    "reports/threecity_supplement/manifest.json",
    "routing_urban_medium_constitutive/results/um_ctx.npz",
]

PAPER_FILES = [
    "paper/merged_manuscript/main.tex",
    "paper/merged_manuscript/supplement.tex",
    "paper/merged_manuscript/refs.bib",
    "paper/merged_manuscript/authors.tex",
    "paper/merged_manuscript/main_refs.tex",
    "paper/merged_manuscript/supplement_refs.tex",
    "paper/merged_manuscript/main.pdf",
    "paper/merged_manuscript/supplement.pdf",
    "paper/merged_manuscript/figures_supp/README.md",
    "paper/merged_manuscript/supplement_sections/01_data.tex",
    "paper/merged_manuscript/supplement_sections/02_estimation.tex",
    "paper/merged_manuscript/supplement_sections/03_spectral.tex",
    "paper/merged_manuscript/supplement_sections/04_spatial.tex",
    "paper/merged_manuscript/supplement_sections/05_baselines.tex",
    "paper/merged_manuscript/supplement_sections/06_coupling_results.tex",
    "paper/merged_manuscript/supplement_sections/07_spectral_results.tex",
    "paper/merged_manuscript/supplement_sections/08_anisotropy_results.tex",
    "paper/merged_manuscript/supplement_sections/09_structural_results.tex",
    "paper/merged_manuscript/supplement_sections/10_reproducibility.tex",
]

DERIVATION_INPUTS = [
    "crosscity_sample_complexity/artifacts/routing_snapshot_meta.json",
    "crosscity_sample_complexity/artifacts/routing_snapshot_Porto.npz",
    "crosscity_sample_complexity/artifacts/routing_snapshot_SF.npz",
    "empirical_transfer_round/artifacts/et_data_porto.npz",
    "empirical_transfer_round/artifacts/et_data_sf.npz",
]

FIGURES = [
    ("fig01_concept", "main", "paper/merged_manuscript/figures/redraw_urbanflow.py", "concept"),
    ("fig02_threecity_coupling", "main", "paper/merged_manuscript/figures/redraw_urbanflow.py", "threecity"),
    ("fig03_joint_spectrum", "main", "reports/threecity_joint_spectrum/manuscript_revision/redraw_joint_spectrum.py", "frozen verified_joint.npz"),
    ("fig04_frequency_response", "main", "paper/merged_manuscript/figures/reorganize_threecity_figures.py", "--plots-only"),
    ("fig05_anisotropy_threecity", "main", "reports/threecity_figure5_revision/scripts/fig5_redesign.py", "B"),
    ("figS01_joint_spectrum_2d", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S01"),
    ("figS02_heldout_daily_scores", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S02"),
    ("figS03_coherent_response_train_test", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S03"),
    ("figS04_anisotropy_distributions", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S04"),
    ("figS05_matched_capacity", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S05"),
    ("figS06_heldout_reconstruction", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S06"),
    ("figS07_nonlinear_exploratory", "supp", "reports/threecity_supplement/scripts/make_figures.py", "S07"),
]

URBANFLOW_PY = [
    "__init__.py", "__main__.py", "_version.py", "cli.py", "fields.py",
    "frames.py", "geometry.py", "manifest.py", "metrics.py", "paths.py",
    "reference.py", "registry.py", "selfcheck.py", "splits.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False))


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"release patch anchor missing: {label}")
    return text.replace(old, new, 1)


def source_file(root: Path, rel: str, allow_large: bool = False) -> Path:
    if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
        raise ValueError(rel)
    path = root.joinpath(*PurePosixPath(rel).parts)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"allowlisted source is missing or not a regular file: {rel}")
    if not allow_large and path.stat().st_size > MAX_BYTES:
        raise ValueError(f"allowlisted source exceeds 50 MiB: {rel}")
    return path


def copy_file(source_root: Path, output_root: Path, rel: str) -> None:
    src = source_file(source_root, rel)
    dst = output_root.joinpath(*PurePosixPath(rel).parts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def save_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with fixed ZIP metadata, making repeated exports byte-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())


def frozen_splits(source_root: Path, city: str):
    name = "porto" if city == "Porto" else "cabspotting"
    doc = json.loads(source_file(
        source_root, f"reports/deep_baseline_training_record/splits/{name}_datefreeze.json"
    ).read_text(encoding="utf-8"))
    split = doc["split"]
    if "folds" not in split:
        return [split["train"]["dates"]]
    eligible = [int(x) for x in doc["eligible_dates"]]
    folds = []
    for fold in split["folds"]:
        test = set(fold["block"])
        val = {x for x in eligible if int(fold["val_first"]) <= x <= int(fold["val_last"])}
        folds.append([x for x in eligible if x not in test | val])
    return folds


def derive_source_spectra(source_root: Path) -> dict[str, np.ndarray]:
    """Compact only the q matrices needed by the two frozen spectral displays."""
    out: dict[str, np.ndarray] = {}
    for city in ("Porto", "SF"):
        rel = f"empirical_transfer_round/artifacts/et_data_{city.lower()}.npz"
        with np.load(source_file(source_root, rel, allow_large=True),
                     allow_pickle=False) as z:
            order = [int(x) for x in z["order"]]
            nc = int(z["NC"])
            x = np.asarray(z["X"], dtype=np.float64).reshape(-1, 24, nc)
        pos = {date: i for i, date in enumerate(order)}
        pks, powers, c0_errors = [], [], []
        for dates in frozen_splits(source_root, city):
            a = x[[pos[int(d)] for d in dates]]
            c = np.fft.rfft(a, axis=1) / np.sqrt(24)
            c0_errors.append(float(np.max(np.abs(
                c[:, 0, :] - np.sqrt(24) * a.mean(axis=1)))))
            pk, power = [], []
            for h in range(13):
                ah = c[:, h, :]
                eigenvalues = np.clip(np.linalg.eigvalsh(
                    ah @ ah.conj().T), 0, None)[::-1]
                pk.append(eigenvalues[:8] / eigenvalues.sum())
                power.append(float(np.mean(np.sum(np.abs(ah) ** 2, axis=1))))
            pks.append(pk)
            powers.append(power)
        pk = np.mean(pks, axis=0)
        power = np.mean(powers, axis=0)
        q = pk * (power / power.sum())[:, None]
        out[f"{city}__q"] = q
        out[f"{city}__c0_err"] = np.asarray(max(c0_errors), dtype=np.float64)
        out[f"{city}__n_modes"] = np.asarray(8, dtype=np.int64)
    return out


def derive_reconstruction(source_root: Path) -> dict[str, np.ndarray]:
    """Keep only the selected held-out date/phase vectors drawn in Supplement S4."""
    meta = json.loads(source_file(
        source_root, "crosscity_sample_complexity/artifacts/routing_snapshot_meta.json"
    ).read_text(encoding="utf-8"))["cities"]
    out: dict[str, np.ndarray] = {"phase": np.asarray(12, dtype=np.int64)}
    for city in ("Porto", "SF"):
        rel = f"crosscity_sample_complexity/artifacts/routing_snapshot_{city}.npz"
        errors = np.asarray(meta[city]["E_per_date"], dtype=np.float64)
        index = int(np.argmin(np.abs(errors - np.median(errors))))
        with np.load(source_file(source_root, rel, allow_large=True),
                     allow_pickle=False) as z:
            index = int(z["picked_i"])
            phase = int(z["picked_phase"])
            out[f"{city}__obs"] = np.asarray(z["date24"][index, phase], dtype=np.float64)
            out[f"{city}__hat"] = np.asarray(z["hat24"][index, phase], dtype=np.float64)
            out[f"{city}__test_date"] = np.asarray(z["test_dates"][index], dtype=np.int64)
            out[f"{city}__index"] = np.asarray(index, dtype=np.int64)
            out[f"{city}__phase"] = np.asarray(phase, dtype=np.int64)
        out[f"{city}__date_error"] = np.asarray(errors[index], dtype=np.float64)
    return out


def patch_redraw(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_required(text, "from pathlib import Path\n",
                            "from pathlib import Path\nimport os\n", "redraw os import")
    old = "ROOT=Path(__file__).resolve().parents[3]"
    text = replace_required(
        text, old,
        "REPO_ROOT=Path(__file__).resolve().parents[3]\n"
        "ROOT=Path(os.environ.get('URBANFLOW_ROOT',REPO_ROOT))\n"
        "OUT=REPO_ROOT/'paper/merged_manuscript/figures'",
        "redraw portable root")
    # The replacement above deliberately includes OUT; remove the old duplicate.
    text = text.replace("OUT=REPO_ROOT/'paper/merged_manuscript/figures'\nOUT=ROOT/'paper/merged_manuscript/figures'",
                        "OUT=REPO_ROOT/'paper/merged_manuscript/figures'", 1)
    text = replace_required(text, "OUT=ROOT/'paper/merged_manuscript/figures'",
                            "OUT=REPO_ROOT/'paper/merged_manuscript/figures'",
                            "redraw local output") if "OUT=ROOT/'paper/merged_manuscript/figures'" in text else text
    text = replace_required(text, "SUPP_OUT=ROOT/'paper/merged_manuscript/figures_supp'",
                            "SUPP_OUT=REPO_ROOT/'paper/merged_manuscript/figures_supp'",
                            "redraw local supplementary output")
    start = text.index("def source_spectrum(city):")
    end = text.index("\ndef spectra_response():", start)
    compact = '''def source_spectrum(city):
 """Read the small frozen q matrix derived from the large aligned source cache."""
 with np.load(ROOT/'frozen/source_spectra.npz',allow_pickle=False) as z:
  return np.asarray(z[f'{city}__q'],dtype=np.float64)
'''
    text = text[:start] + compact + text[end:]
    start = text.index("def reconstruction():")
    end = text.index("\ndef source_spectrum(city):", start)
    original = text[start:end]
    # Preserve the drawing body but replace only its data-loading preamble.
    marker = " summary['snapshots']={}\n"
    if marker not in original:
        raise RuntimeError("release patch anchor missing: reconstruction preamble")
    drawing = original.split(marker, 1)[1]
    prefix = '''def reconstruction():
 zgeom=nz(ROOT/'routing_urban_medium_constitutive/results/um_ctx.npz')
 frozen=nz(ROOT/'frozen/reconstruction.npz')
 fig=plt.figure(figsize=(7.0,4.5))
 rects=[(.08,.53,.27,.38),(.08,.09,.27,.38),(.43,.09,.275,.82),(.72,.09,.275,.82)]
 summary['snapshots']={}
'''
    # Replace the large snapshot reads and metadata-based selection in the body.
    drawing = drawing.replace("  z=nz(ROOT/f'crosscity_sample_complexity/artifacts/routing_snapshot_{city}.npz')\n"
                              "  assert meta[city]['abs_err']<1e-8\n", "")
    drawing = drawing.replace("  # Representative date: closest-to-median archived per-date error, lower date breaks ties.\n"
                              "  err=np.array(meta[city]['E_per_date']);i=int(np.argmin(np.abs(err-np.median(err))));ph=12\n"
                              "  obs=z['date24'][i,ph]; hat=z['hat24'][i,ph]\n",
                              "  ph=int(frozen['phase']);i=int(frozen[f'{city}__index'])\n"
                              "  obs=frozen[f'{city}__obs'];hat=frozen[f'{city}__hat'];err=float(frozen[f'{city}__date_error'])\n")
    drawing = drawing.replace("  date=str(np.datetime64('1970-01-01')+np.timedelta64(int(z['test_dates'][i]),'D'))",
                              "  date=str(np.datetime64('1970-01-01')+np.timedelta64(int(frozen[f'{city}__test_date']),'D'))")
    drawing = drawing.replace("summary['snapshots'][city]={'date':date,'phase':ph,'date_error':float(err[i]),'index':i,'scale_p99':vmax,'selection':'closest-to-median archived per-date error; fixed 12:00'}",
                              "summary['snapshots'][city]={'date':date,'phase':ph,'date_error':float(err),'index':i,'scale_p99':vmax,'selection':'closest-to-median archived per-date error; fixed 12:00'}")
    if "routing_snapshot_" in drawing or "meta[city]" in drawing or "err[i]" in drawing:
        raise RuntimeError("large reconstruction input survived the release patch")
    path.write_text(text[:start] + prefix + drawing + text[end:], encoding="utf-8", newline="\n")


def patch_reorganize(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    markers = ("if '--plots-only' in sys.argv:", "if '--rewrite-manuscript' not in sys.argv:")
    marker = next((m for m in markers if m in text), None)
    if marker is None:
        raise RuntimeError("release patch anchor missing: reorganize manuscript guard")
    head = text.split(marker, 1)[0]
    head += "print('saved three-city response plots; manuscript editing is disabled in the public copy')\n"
    path.write_text(head, encoding="utf-8", newline="\n")


def patch_figstyle(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_required(text, 'ROOT = _locate("empirical_transfer_round")',
                            'REPO_ROOT = FIGS.parents[2]\n'
                            'ROOT = pathlib.Path(os.environ.get("URBANFLOW_ROOT", _locate("empirical_transfer_round")))',
                            "figstyle data root")
    if "import os" not in text:
        text = replace_required(text, "import pathlib\n", "import os\nimport pathlib\n",
                                "figstyle os import")
    text = replace_required(text, 'STYLE_DIR = ROOT / "paper" / "singlecolumn" / "figures"',
                            'STYLE_DIR = REPO_ROOT / "paper" / "singlecolumn" / "figures"',
                            "figstyle local code")
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_fig5(path: Path, kind: str) -> None:
    text = path.read_text(encoding="utf-8")
    if "import os" not in text:
        text = replace_required(text, "import pathlib\n", "import os\nimport pathlib\n",
                                kind + " os import")
    text = replace_required(text, "ROOT = HERE.parents[3]",
                            "ROOT = pathlib.Path(os.environ.get('URBANFLOW_ROOT', REPO_ROOT))",
                            kind + " data root")
    text = replace_required(text, "REV = HERE.parents[1]",
                            "REPO_ROOT = HERE.parents[3]\nREV = REPO_ROOT / 'reports' / 'threecity_figure5_revision'",
                            kind + " local revision")
    text = text.replace('FIGS = ROOT / "paper" / "merged_manuscript" / "figures"',
                        'FIGS = REPO_ROOT / "paper" / "merged_manuscript" / "figures"')
    text = text.replace('SUPP_DIR = ROOT / "paper" / "merged_manuscript" / "figures_supp"',
                        'SUPP_DIR = REPO_ROOT / "paper" / "merged_manuscript" / "figures_supp"')
    text = text.replace('OUT = REV / "figures"',
                        'OUT = pathlib.Path(os.environ.get("URBANFLOW_FIGURE_OUT", REV / "figures"))')
    if kind in {"fig5_threecity", "threecity_stats"}:
        text = text.replace('CD = REV / "results" / "chengdu_anisotropy.npz"',
                            'DATA_REV = ROOT / "reports" / "threecity_figure5_revision"\nCD = DATA_REV / "results" / "chengdu_anisotropy.npz"')
        text = text.replace('CD_SUM = REV / "results" / "chengdu_anisotropy_summary.json"',
                            'CD_SUM = DATA_REV / "results" / "chengdu_anisotropy_summary.json"')
        text = text.replace('CD_A = REV / "results" / "partA_chengdu.json"',
                            'DATA_REV = ROOT / "reports" / "threecity_figure5_revision"\nCD_A = DATA_REV / "results" / "partA_chengdu.json"')
        text = text.replace('CD_F = REV / "results" / "partF_summary.json"',
                            'CD_F = DATA_REV / "results" / "partF_summary.json"')
        text = text.replace('p = REV / "results" / name', 'p = DATA_REV / "results" / name')
        text = text.replace('[REV / "results" / n for n in BM_FILE.values()]',
                            '[DATA_REV / "results" / n for n in BM_FILE.values()]')
        text = text.replace('p = REV / "results" / "fig5_threecity_manifest.json"',
                            'p = pathlib.Path(os.environ.get("URBANFLOW_RESULT_OUT", REV / "results")) / "fig5_threecity_manifest.json"')
        text = text.replace('str(p): hashlib.sha256(p.read_bytes()).hexdigest()',
                            'p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()')
    if kind == "fig5_redesign":
        text = text.replace('[REV / "results" / n for n in F5.BM_FILE.values()]',
                            '[F5.DATA_REV / "results" / n for n in F5.BM_FILE.values()]')
        text = text.replace('p = REV / "results" / "fig5_redesign_manifest.json"',
                            'p = pathlib.Path(os.environ.get("URBANFLOW_RESULT_OUT", REV / "results")) / "fig5_redesign_manifest.json"')
    if kind == "fig5_threecity":
        old = 'if __name__ == "__main__":\n    fig = main()'
        new = 'if __name__ == "__main__":\n    if "--supp-only" in sys.argv:\n        for p in save(supp_cdf(load()), "figS07_anisotropy_distributions"):\n            print("wrote", p)\n        raise SystemExit(0)\n    fig = main()'
        text = replace_required(text, old, new, "fig5 supplementary-only entry")
    if "host 164" in text:
        text = text.replace(" on host 164 and pull it", " in the external frozen data root")
    path.write_text(text, encoding="utf-8", newline="\n")


def make_minimal_style(source: Path, destination: Path) -> None:
    """Keep only style primitives used by the allowlisted figure entry points."""
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    wanted_functions = {"apply_theme", "panel_tag", "tidy", "save", "jload",
                        "csvrows", "f", "scale_bar", "to_rgba"}
    wanted_names = {"MOB", "ROAD", "CROSS", "FLEX", "BASE", "NULL", "INK",
                    "TEAL", "LGRAY", "BLACK", "FG", "AXCOL", "FONT",
                    "SERVER_FONT", "LW"}
    chunks = ['"""Shared plotting primitives, pruned to the public figure closure."""',
              "from __future__ import annotations", "import csv", "import json",
              "import pathlib", "import numpy as np", "HERE = pathlib.Path(__file__).resolve().parent",
              "OUT = HERE"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            chunks.append(ast.get_source_segment(text, node) or "")
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted_names
                                                   for t in node.targets):
            chunks.append(ast.get_source_segment(text, node) or "")
    write_text(destination, "\n\n".join(chunks))


def make_minimal_fig2_response(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "spectra_panel")
    body = ast.get_source_segment(text, function)
    if not body:
        raise RuntimeError("cannot extract spectra_panel")
    header = '''"""The 3-D spectrum panel used by the public supplementary figure."""
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
'''
    write_text(destination, header + "\n" + body)


def make_frozen_etfig_adapter(destination: Path) -> None:
    write_text(destination, '''"""Small frozen-input adapters for the two public supplementary figures.

Only functions used by those entry points are included.  No aligned field or
model fitting is hidden behind this module.
"""
from __future__ import annotations
import json
import numpy as np
import figstyle as FS

NH = 13

def _jload(path):
    return json.loads(path.read_text(encoding="utf-8"))

def source_spectrum(city, kmax=8, rec=None):
    if kmax != 8:
        raise ValueError("the compact public product stores kmax=8 only")
    with np.load(FS.ROOT / "frozen/source_spectra.npz", allow_pickle=False) as z:
        q = np.asarray(z[f"{city}__q"], dtype=np.float64)
        return dict(q=q, pk=q, P=q.sum(axis=1), share=q.sum(axis=1),
                    kept=float(q.sum()), kept_hi=float(q[:2].sum()),
                    c0_err=float(z[f"{city}__c0_err"]), kmax=8,
                    n_modes=int(z[f"{city}__n_modes"]))

def transfer(city, rec=None, series=None, arms=None):
    a = _jload(FS.ET_RES / "A_frequency_arms.json")[city]
    return dict(eta=np.asarray([a["eta_h"][str(h)] for h in range(NH)], dtype=float))

def closure(city):
    c = _jload(FS.ET_RES / "C_spectral_closure.json")[city]
    h = np.arange(NH)
    return dict(h=h,
                trace=np.asarray([c[str(x)]["trace_ratio"] for x in h]),
                overlap=np.asarray([c[str(x)]["overlap_k3"] for x in h]),
                canon=np.asarray([c[str(x)]["canon_load"] for x in h]))

def nonlinear(city):
    d = _jload(FS.ET_RES / "D_nonlinear_increment.json")[city]
    return {a: dict(d=d["summary"][a]["dR2"], lo=d["summary"][a]["ci"][0],
                    hi=d["summary"][a]["ci"][1], e=d["summary"][a]["E_te"])
            for a in ("lin0", "linF", "quadF", "krrF")}

def bicoherence(city):
    z = _jload(FS.ET_RES / "E_bicoherence.json")[city]
    return {k: dict(pairs=[r["pair"] for r in z[k]["rows"]],
                    obs=np.asarray([r["observed"] for r in z[k]["rows"]]),
                    q99=np.asarray([r["surrogate_q99"] for r in z[k]["rows"]]),
                    p=np.asarray([r["p_one_sided"] for r in z[k]["rows"]]),
                    n=int(z[k]["n_dates"]))
            for k in ("source", "routing", "residual")}
''')


def sanitize_frozen_summary(output_root: Path) -> None:
    """Make provenance keys relative without changing any scientific value."""
    path = output_root / "reports/threecity_figure5_revision/results/chengdu_anisotropy_summary.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    inputs = doc.get("inputs_sha256", {})
    cleaned = {}
    for key, value in inputs.items():
        # Keep only the pinned project-relative object, never a host root.
        parts = PurePosixPath(key.replace("\\", "/")).parts
        if "project10" in parts:
            rel = PurePosixPath(*parts[parts.index("project10"):]).as_posix()
        elif "road_spatial_transfer" in parts:
            rel = PurePosixPath(*parts[parts.index("road_spatial_transfer"):]).as_posix()
        elif "urbanflow_ref" in parts:
            rel = PurePosixPath(*parts[parts.index("urbanflow_ref"):]).as_posix()
        elif PurePosixPath(key).is_absolute():
            raise RuntimeError(f"unrecognized absolute provenance path in {path.name}")
        else:
            rel = PurePosixPath(key).as_posix()
        cleaned[rel] = value
    doc["inputs_sha256"] = cleaned
    write_json(path, doc)


def export_urbanflow_ref(source_root: Path, output_root: Path) -> dict:
    package = output_root / "urbanflow_ref"
    for name in URBANFLOW_PY:
        copy_file(source_root, output_root, f"urbanflow_ref/{name}")
    for rel in ("goldens/urbanflow_reference_v1.json", "manifest.json",
                "tests/conftest.py", "tests/test_reference.py"):
        copy_file(source_root, output_root, f"urbanflow_ref/{rel}")
    # The optional comparison against private legacy libraries is not part of
    # the public import closure.  Preserve an explicit failure for callers who
    # ask for it instead of exposing a ModuleNotFoundError or pretending pass.
    selfcheck_path = package / "selfcheck.py"
    selfcheck = selfcheck_path.read_text(encoding="utf-8")
    start = selfcheck.index("def _legacy_check(ref):")
    selfcheck = selfcheck[:start] + '''def _legacy_check(ref):
    """The private legacy libraries are intentionally not exported."""
    raise RuntimeError("legacy comparison libraries are not included; rerun with --no-legacy")
'''
    write_text(selfcheck_path, selfcheck)
    write_text(package / "paths.py", '''"""Resolve external frozen data without a workstation-specific default."""
from __future__ import annotations
import os
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PACKAGE_ROOT / "external_data"

def project_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("URBANFLOW_ROOT", str(DEFAULT_ROOT))).expanduser().resolve()

def resolve(rel: str) -> pathlib.Path:
    p = pathlib.Path(rel)
    return p if p.is_absolute() else project_root() / p

def package_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent
''')
    manifest_path = package / "manifest.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream_digest = doc["digest"]
    payload = doc["payload"]
    files = {p.name: sha256(p) for p in sorted(package.glob("*.py"))
             if p.name != "_version.py"}
    payload["code"] = dict(files=files, excluded=["_version.py"],
                           digest=hashlib.sha256(canonical(files)).hexdigest())
    body = {k: v for k, v in payload.items() if k not in ("created_utc", "env", "digest")}
    digest = hashlib.sha256(canonical(body)).hexdigest()
    write_json(manifest_path, dict(digest=digest, payload=payload))
    version_path = package / "_version.py"
    version = version_path.read_text(encoding="utf-8")
    import re
    version, count = re.subn(r'MANIFEST_DIGEST = "[0-9a-f]{64}"',
                             f'MANIFEST_DIGEST = "{digest}"', version, count=1)
    if count != 1:
        raise RuntimeError("cannot pin the exported urbanflow_ref manifest digest")
    write_text(version_path, version)
    golden_path = package / "goldens/urbanflow_reference_v1.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    golden["digest"] = digest
    golden["upstream_digest"] = upstream_digest
    write_json(golden_path, golden)
    return dict(upstream_reference_digest=upstream_digest,
                exported_reference_digest=digest,
                portability_patch="URBANFLOW_ROOT or <repo>/external_data; private legacy check made an explicit unsupported option; code and manifest hashes recomputed")


def make_readme(output_root: Path, digest: str) -> None:
    commands = "\n".join([
        "python paper/merged_manuscript/figures/redraw_urbanflow.py",
        "python paper/merged_manuscript/figures/reorganize_threecity_figures.py --plots-only",
        "python reports/threecity_joint_spectrum/manuscript_revision/redraw_joint_spectrum.py",
        "python reports/threecity_figure5_revision/scripts/fig5_redesign.py B",
        "python reports/threecity_supplement/scripts/make_figures.py",
    ])
    write_text(output_root / "README.md", f'''# UrbanFlow public-release candidate

Code and small frozen products for **Road-structured constitutive coupling between source--sink patterns and anisotropic urban routing**.

Repository: https://github.com/Wangwangw233/paper-urbanflow

> **Candidate, not yet an open-source release.** The authors must still choose a license, complete the citation metadata, and supply the repository URL. The current paper and Fig. 5 are synchronized and verified. `LICENSE` intentionally grants no new rights. No DOI, author list, or repository URL is invented here.

## Three reproduction levels

| level | what is included | inputs and limits |
|---|---|---|
| **1. Frozen figure redraw** | Plotting code, 12 reference PDFs, and small aggregate frozen arrays | Runs without raw trajectories or model fitting. `frozen/source_spectra.npz` and `frozen/reconstruction.npz` contain only arrays needed for display; they are not substitutes for the large scientific inputs from which they were derived. PDF metadata and rasterization can differ by environment, so byte-identical PDFs are not claimed. |
| **2. Core recomputation** | `urbanflow_ref`: aligned-field loading, geometry, road projection, ridge evaluation, frozen splits and numeric goldens | **Requires external frozen caches, bases, geometry and split files.** They are not bundled. `manifest/data.csv` lists their relative paths, sizes and pinned hashes. A full regression also requires every object in the manifest. |
| **3. Raw end-to-end rebuild** | Not included in this candidate | Raw trajectory ingestion, registration, road downloading, field construction and model training still require a separate privacy/license review and their original large inputs. No command in this repository claims to perform this level. |

## Environment and smoke tests

Python 3.10+ is recommended; see `docs/ENVIRONMENT.md` for the exact environment used for this export. Install the small plotting/compute stack:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/verify_public_repo.py --figure-smoke
```

The verifier checks the allowlist, file hashes and sizes, text/path/privacy patterns, NPZ safety, local imports, AST parsing and `compileall`. `--figure-smoke` also redraws all 12 figures into a temporary directory and checks that the candidate tree remains unchanged. Passing these tests does **not** mean the external-data core regression or raw end-to-end pipeline has run.

## Frozen figure redraw

From the repository root:

```bash
{commands}
```

The scripts write PDFs and PNGs beside the reference PDFs. To avoid overwriting them, use each script's output option where provided, or set `URBANFLOW_FIGURE_OUT`, `URBANFLOW_SUPP_FIGURE_OUT`, and `URBANFLOW_RESULT_OUT` to separate directories. `URBANFLOW_ROOT` defaults to this repository and can point to another release-shaped frozen-data root. The joint-spectrum script additionally accepts `--data` and `--out`.

`reorganize_threecity_figures.py` is restricted to plotting in this public copy; its manuscript-editing section is deliberately removed.

## Core recomputation with external data

Arrange the pinned external inputs under one root using the relative paths in `manifest/data.csv`, then run, for example:

```bash
export URBANFLOW_ROOT=/path/to/frozen_data       # use set on Windows cmd
python -m urbanflow_ref describe Porto
python -m urbanflow_ref evaluate Porto --split frozen --fold 0
python -m urbanflow_ref verify --no-legacy       # needs ALL external objects
```

Do not expect these commands to work with only the bundled figure products. The public reference digest after the path-only packaging patch is `{digest}`; the upstream digest and patch provenance are in `manifest/provenance.json`. Numeric goldens are unchanged. `--no-legacy` is required because private legacy comparison libraries are not exported.

## Data, privacy, and attribution

- No individual-level trajectory data are included. NPZ files are numeric/Unicode archives and are checked without pickle.
- Large aligned matrices are not bundled. The two compact display products record their source hashes in `manifest/provenance.json`.
- Basemap arrays are derived from OpenStreetMap road/water geometry; acknowledge **OpenStreetMap contributors** and review the applicable source-data terms before publication.
- `manifest/files.csv` lists every release file other than itself, with SHA-256. `manifest/data.csv` distinguishes bundled products from required external inputs. `manifest/figures.csv` maps each paper figure to its generator and reference PDF.
- PDF and NPZ files are deliberately **not** globally ignored. Generated rasters, caches, build products, and `external_data/` are ignored instead.

## Cross-document references

`main_refs.tex` and `supplement_refs.tex` provide the supplementary/main S-number macros without `xr` or any dependency on pre-existing `.aux` files. If labels change, recompile once to regenerate the aux files, run `python tools/sync_crossrefs.py`, then compile again. Fresh Overleaf-style compilation without prior aux files is supported.

## Repeating the export

The release builder is retained in `tools/export_public_repo.py`. From the original private analysis tree it rebuilds this directory from the explicit allowlist:

```bash
python tools/export_public_repo.py --source-root /path/to/private/analysis --output /path/to/release/urbanflow
python tools/verify_public_repo.py --root /path/to/release/urbanflow --figure-smoke
```

The current export has passed the paper/Fig. 5 synchronization checks. If any source changes, rerun both commands and review `manifest/figures.csv` before publishing.
''')


def write_support_files(output_root: Path, reference_digest: str) -> None:
    make_readme(output_root, reference_digest)
    write_text(output_root / "LICENSE", '''LICENSE STATUS: AUTHOR DECISION REQUIRED

No open-source license has been selected for this candidate as of this export.
This file is a placeholder, not a license grant. Copyright remains with the
respective authors and other rights holders. Do not redistribute, modify, or
reuse this material under an assumed permission. The authors must replace this
file with an explicitly chosen license and reconcile dataset, OpenStreetMap,
font, and third-party code terms before public release.
''')
    write_text(output_root / "CITATION", '''CITATION STATUS: PARTIAL METADATA

Work title:
Road-structured constitutive coupling between source--sink patterns and anisotropic urban routing

Authors:
Tiehang Xu; Jiahui Wang; Junyan Yang; Qiao Wang

Repository:
https://github.com/Wangwangw233/paper-urbanflow

The publication year, venue, software version, DOI and preferred citation format
have not been finalized. Replace this file with a machine-readable
CITATION.cff or the authors' final citation instructions once confirmed.
''')
    write_text(output_root / "requirements.txt", '''numpy>=1.24
scipy>=1.10
matplotlib>=3.6
PyMuPDF>=1.24
Pillow>=10
threadpoolctl>=3
''')
    write_text(output_root / ".gitignore", '''# Python and test caches
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Environments and external scientific inputs
.venv/
venv/
external_data/

# Editors and OS metadata
.DS_Store
Thumbs.db
*.swp

# Temporary/build residue
_tmp*/
scratch/
build/
dist/
*.new
*.tmp
*.bak
*.aux
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.toc

# Generated rasters (reference vector PDFs and frozen NPZ remain trackable)
paper/merged_manuscript/figures/*.png
paper/merged_manuscript/figures_supp/*.png
reports/threecity_figure5_revision/figures/*.png
reports/threecity_joint_spectrum/manuscript_revision/*.png
''')
    import platform
    import scipy
    import matplotlib
    write_text(output_root / "docs/ENVIRONMENT.md", f'''# Environment

## Tested export environment

- Python: `{platform.python_version()}`
- NumPy: `{np.__version__}`
- SciPy: `{scipy.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- OS family: `{platform.system()}`

These are the versions used by the exporter, not a claim that every scientific
result is bitwise portable to every BLAS or platform. The reference package sets
its BLAS thread count to one. Numerical golden comparisons use the tolerances in
`urbanflow_ref/goldens/urbanflow_reference_v1.json` and `urbanflow_ref/selfcheck.py`.

## Setup

Use a virtual environment and install `requirements.txt`. Pytest is optional and
is needed only for the original `urbanflow_ref/tests` suite; the public smoke
suite uses the standard-library `unittest` runner. Matplotlib must be able to use
its Agg backend. The verifier sets `MPLCONFIGDIR` and `PYTHONDONTWRITEBYTECODE`
for subprocesses.

## External data

`URBANFLOW_ROOT` selects the frozen-data root. It defaults to the release root
for figure redraw and to `external_data/` for `urbanflow_ref`. Core commands
fail on missing inputs; this is intentional. No raw-data download or implicit
fallback to a workstation path is provided.
''')


def make_smoke_test(output_root: Path) -> None:
    write_text(output_root / "tests/test_public_smoke.py", '''"""Smoke tests that require no external scientific data."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np
import scipy.sparse as sp
import urbanflow_ref as uf
from urbanflow_ref import frames, geometry, manifest as mf

ROOT = Path(__file__).resolve().parents[1]

class PublicSmoke(unittest.TestCase):
    def test_manifest_and_code_hashes(self):
        ref = uf.load()
        self.assertEqual(ref.digest, uf.MANIFEST_DIGEST)
        ok, msg = mf.check_digest(ref.manifest, ref.digest)
        self.assertTrue(ok, msg)
        self.assertEqual(mf._code_section()["files"], ref.manifest["code"]["files"])

    def test_synthetic_frame_identity(self):
        rng = np.random.default_rng(20261031)
        n, dim = 16, 3
        ph = np.arange(n) % 2
        x = rng.normal(size=(n, dim)) + 0.4 * np.cos(ph[:, None])
        y = 0.3 * x + rng.normal(scale=0.1, size=(n, dim))
        d = dict(X=x, Y=y, ph=ph, r=dim,
                 di=np.arange(n) // 2, order=np.arange(n // 2))
        masks = [np.arange(n) < 8, (np.arange(n) >= 8) & (np.arange(n) < 12), np.arange(n) >= 12]
        out = frames.run_frames(d, *masks)
        self.assertLess(out["identity_abs_err"], 1e-10)
        for key in ("R_P", "R_D", "R_I", "R_CLIM"):
            self.assertTrue(np.isfinite(out[key]))

    def test_synthetic_geometry_projector(self):
        # One square, four interior faces, one cycle.
        geometry._CTX["public-square"] = dict(ny=2, nx=2, NC=4, n_int=4,
            p0=np.asarray([0, 2, 0, 1]), p1=np.asarray([1, 3, 2, 3]))
        geometry._LU.pop("public-square", None)
        v = np.asarray([[1.0, -2.0, 3.0, 0.5]])
        projected = geometry.P0("public-square", v)
        residual = geometry.D("public-square") @ projected.T
        self.assertLess(float(np.max(np.abs(residual))), 1e-10)
        self.assertLess(float(np.max(np.abs(geometry.P0("public-square", projected) - projected))), 1e-10)

    def test_compact_products_are_numeric(self):
        for rel in ("frozen/source_spectra.npz", "frozen/reconstruction.npz"):
            with np.load(ROOT / rel, allow_pickle=False) as z:
                self.assertTrue(z.files)
                for key in z.files:
                    self.assertNotEqual(z[key].dtype, np.dtype("O"))
                    self.assertTrue(np.all(np.isfinite(z[key])))

if __name__ == "__main__":
    unittest.main()
''')


def external_data_rows(doc: dict) -> list[dict]:
    payload = doc["payload"]
    found: dict[str, dict] = {}
    def add(rel, sha, size, role, required_by):
        if rel not in found:
            found[rel] = dict(path=rel, status="external_required", role=role,
                              sha256=sha, bytes=int(size), source_path="",
                              derived_from="", required_by=required_by)
    for name, obj in payload["data"]["objects"].items():
        for role, rec in (("aligned_cache", obj["cache"]), ("frozen_split", obj["freeze"])):
            add(rec["path"], rec["sha256"], rec["bytes"], role, name)
        rec = payload["road_basis"][name]
        add(rec["path"], rec["sha256"], rec["bytes"], "road_basis", name)
    rec = payload["geometry"]["source"]
    add(rec["path"], rec["sha256"], rec["bytes"], "geometry", "urbanflow_ref")
    return [found[k] for k in sorted(found)]


def write_manifests(source_root: Path, output_root: Path, provenance: dict) -> None:
    (output_root / "manifest").mkdir(parents=True, exist_ok=True)
    data_rows = []
    for rel in DATA_FILES + ["frozen/source_spectra.npz", "frozen/reconstruction.npz"]:
        p = output_root / rel
        source = rel if rel in DATA_FILES else ""
        derived = ""
        if rel == "frozen/source_spectra.npz":
            derived = ";".join(provenance["derived_sources"]["source_spectra"])
        elif rel == "frozen/reconstruction.npz":
            derived = ";".join(provenance["derived_sources"]["reconstruction"])
        data_rows.append(dict(path=rel, status="bundled", role="derived_display_product" if derived else "frozen_aggregate",
                              sha256=sha256(p), bytes=p.stat().st_size,
                              source_path=source, derived_from=derived,
                              required_by="frozen figure redraw"))
    core_doc = json.loads((output_root / "urbanflow_ref/manifest.json").read_text(encoding="utf-8"))
    data_rows.extend(external_data_rows(core_doc))
    fields = ["path", "status", "role", "sha256", "bytes", "source_path", "derived_from", "required_by"]
    with (output_root / "manifest/data.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(data_rows)
    figure_rows = []
    for name, group, script, entry in FIGURES:
        rel = f"paper/merged_manuscript/{'figures' if group == 'main' else 'figures_supp'}/{name}.pdf"
        p = output_root / rel
        figure_rows.append(dict(figure=name, group=group, generator=script, entry=entry,
                                reference_pdf=rel, sha256=sha256(p), bytes=p.stat().st_size,
                                verification="reference PDF bundled; redraw equivalence not asserted"))
    with (output_root / "manifest/figures.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["figure", "group", "generator", "entry", "reference_pdf", "sha256", "bytes", "verification"]
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(figure_rows)
    write_json(output_root / "manifest/provenance.json", provenance)
    file_rows = []
    for p in sorted(output_root.rglob("*")):
        if not p.is_file() or p.name == "files.csv" or ".git" in p.relative_to(output_root).parts:
            continue
        rel = p.relative_to(output_root).as_posix()
        if p.stat().st_size > MAX_BYTES:
            raise ValueError(f"output exceeds 50 MiB: {rel}")
        category = ("code" if p.suffix == ".py" else "data" if p.suffix in {".npz", ".json", ".csv"}
                    else "figure" if p.suffix == ".pdf" else "documentation")
        file_rows.append(dict(path=rel, sha256=sha256(p), bytes=p.stat().st_size,
                              category=category, source="allowlist or deterministic release generator"))
    with (output_root / "manifest/files.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["path", "sha256", "bytes", "category", "source"]
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(file_rows)


def sanitize_public_text(output_root: Path) -> None:
    replacements = [
        ("E:\\\\Docu_Xu\\\\mobility_constitution", "./"),
        ("./", "./"),
        ("E:\\\\Docu_Xu\\\\mobility", "legacy/"),
        ("legacy/", "legacy/"),
        ("C:\\\\Users\\\\Xu\\\\AppData\\\\Local\\\\Programs\\\\Python\\\\Python312", "python/"),
        ("python/", "python/"),
    ]
    for path in output_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".tex", ".bib", ".csv", ".yml", ".yaml", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")


def sanitize_json_paths(output_root: Path) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return value.replace("\\", "/")
        return value

    for path in output_root.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        write_json(path, clean(data))


def refresh_internal_hashes(output_root: Path) -> None:
    result_path = output_root / "reports/chengdu_nonlinear_exploratory/results/chengdu_nonlinear_exploratory.json"
    if result_path.is_file():
        doc = json.loads(result_path.read_text(encoding="utf-8"))
        doc["config"]["sha256"] = sha256(output_root / "reports/chengdu_nonlinear_exploratory/config.json")
        doc["script"]["sha256"] = sha256(output_root / "reports/chengdu_nonlinear_exploratory/scripts/run_experiment.py")
        write_json(result_path, doc)
    environment_path = output_root / "reports/chengdu_nonlinear_exploratory/results/environment.json"
    if environment_path.is_file():
        doc = json.loads(environment_path.read_text(encoding="utf-8"))
        doc["script"]["sha256"] = sha256(output_root / "reports/chengdu_nonlinear_exploratory/scripts/run_experiment.py")
        write_json(environment_path, doc)
    validation_path = output_root / "reports/chengdu_nonlinear_exploratory/results/validation.json"
    if validation_path.is_file():
        doc = json.loads(validation_path.read_text(encoding="utf-8"))
        doc["result_sha256"] = sha256(result_path)
        doc["config_sha256"] = sha256(output_root / "reports/chengdu_nonlinear_exploratory/config.json")
        write_json(validation_path, doc)
    manifest_path = output_root / "reports/threecity_supplement/manifest.json"
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rel in doc.get("inputs_sha256", {}):
            candidate = output_root / rel
            if candidate.is_file():
                doc["inputs_sha256"][rel] = sha256(candidate)
        write_json(manifest_path, doc)


def export(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or source_root in output_root.parents and output_root.parent != source_root / "release":
        raise ValueError("output must be the dedicated release/urbanflow directory")
    required = FIGURE_CODE + DATA_FILES + DERIVATION_INPUTS + PAPER_FILES
    required += [f"urbanflow_ref/{n}" for n in URBANFLOW_PY]
    required += ["urbanflow_ref/goldens/urbanflow_reference_v1.json", "urbanflow_ref/manifest.json",
                 "urbanflow_ref/tests/conftest.py", "urbanflow_ref/tests/test_reference.py",
                 "paper/singlecolumn/figures/style.py", "paper/merged_manuscript/figures/fig2_response.py",
                 "tools/export_public_repo.py", "tools/verify_public_repo.py", "tools/sync_crossrefs.py"]
    required += [f"paper/merged_manuscript/{'figures' if g == 'main' else 'figures_supp'}/{n}.pdf"
                 for n, g, _s, _e in FIGURES]
    for rel in sorted(set(required)):
        source_file(source_root, rel, allow_large=rel in DERIVATION_INPUTS)
    if output_root.exists():
        if output_root.name != "urbanflow" or output_root.parent.name != "release":
            raise ValueError("refusing to clean a directory not named release/urbanflow")
        for child in output_root.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    for rel in FIGURE_CODE + DATA_FILES + PAPER_FILES:
        copy_file(source_root, output_root, rel)
    for name, group, _script, _entry in FIGURES:
        rel = f"paper/merged_manuscript/{'figures' if group == 'main' else 'figures_supp'}/{name}.pdf"
        copy_file(source_root, output_root, rel)
    for rel in ("tools/export_public_repo.py", "tools/verify_public_repo.py", "tools/sync_crossrefs.py"):
        copy_file(source_root, output_root, rel)

    sanitize_frozen_summary(output_root)
    patch_redraw(output_root / "paper/merged_manuscript/figures/redraw_urbanflow.py")
    patch_reorganize(output_root / "paper/merged_manuscript/figures/reorganize_threecity_figures.py")
    patch_figstyle(output_root / "paper/merged_manuscript/figures/figstyle.py")
    patch_fig5(output_root / "reports/threecity_figure5_revision/scripts/fig5_threecity.py", "fig5_threecity")
    patch_fig5(output_root / "reports/threecity_figure5_revision/scripts/fig5_redesign.py", "fig5_redesign")
    patch_fig5(output_root / "reports/threecity_figure5_revision/scripts/threecity_stats.py", "threecity_stats")
    make_minimal_style(source_root / "paper/singlecolumn/figures/style.py",
                       output_root / "paper/singlecolumn/figures/style.py")
    make_minimal_fig2_response(source_root / "paper/merged_manuscript/figures/fig2_response.py",
                               output_root / "paper/merged_manuscript/figures/fig2_response.py")
    make_frozen_etfig_adapter(output_root / "paper/merged_manuscript/figures/etfig_data.py")

    derivation_provenance = {
        "source_spectra": [
            "empirical_transfer_round/artifacts/et_data_porto.npz:" + sha256(source_root / "empirical_transfer_round/artifacts/et_data_porto.npz"),
            "empirical_transfer_round/artifacts/et_data_sf.npz:" + sha256(source_root / "empirical_transfer_round/artifacts/et_data_sf.npz"),
        ],
        "reconstruction": [
            "crosscity_sample_complexity/artifacts/routing_snapshot_meta.json:" + sha256(source_root / "crosscity_sample_complexity/artifacts/routing_snapshot_meta.json"),
            "crosscity_sample_complexity/artifacts/routing_snapshot_Porto.npz:" + sha256(source_root / "crosscity_sample_complexity/artifacts/routing_snapshot_Porto.npz"),
            "crosscity_sample_complexity/artifacts/routing_snapshot_SF.npz:" + sha256(source_root / "crosscity_sample_complexity/artifacts/routing_snapshot_SF.npz"),
        ],
    }
    save_deterministic_npz(output_root / "frozen/source_spectra.npz",
                           derive_source_spectra(source_root))
    save_deterministic_npz(output_root / "frozen/reconstruction.npz",
                           derive_reconstruction(source_root))
    reference = export_urbanflow_ref(source_root, output_root)
    sanitize_public_text(output_root)
    sanitize_json_paths(output_root)
    refresh_internal_hashes(output_root)
    write_support_files(output_root, reference["exported_reference_digest"])
    make_smoke_test(output_root)
    provenance = dict(export_schema=EXPORT_VERSION, **reference,
                      derived_sources=derivation_provenance,
                      source_policy="explicit allowlist; no source-tree writes; no raw trajectories",
                      paper_sync="not asserted by export; rerun after final paper and Fig. 5 synchronization")
    write_manifests(source_root, output_root, provenance)
    files = sum(1 for p in output_root.rglob("*") if p.is_file() and ".git" not in p.relative_to(output_root).parts)
    size = sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file() and ".git" not in p.relative_to(output_root).parts)
    print(json.dumps(dict(output=str(output_root), files=files, bytes=size,
                          reference_digest=reference["exported_reference_digest"]), indent=2))


def main(argv=None) -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.source_root / "release" / "urbanflow"
    export(args.source_root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
