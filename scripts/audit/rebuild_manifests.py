#!/usr/bin/env python3
"""Rebuild the machine-readable manifests after a deliberate file move.

This script only hashes and indexes files.  It never edits a frozen result and
never invents an external input.  External-input rows are curated in
``data_manifest/external_inputs.csv``; bundled rows are discovered from the
result tree.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "data_manifest"
EXTERNAL = MAN / "external_inputs.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and ".git" not in p.relative_to(ROOT).parts
                  and "__pycache__" not in p.relative_to(ROOT).parts)


def main() -> None:
    all_files = files()
    file_rows = []
    for path in all_files:
        if rel(path) == "data_manifest/files.csv":
            continue
        file_rows.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (MAN / "files.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(file_rows)

    bundled_roots = [
        ROOT / "results" / "frozen",
        ROOT / "configs" / "experiments",
    ]
    bundled = []
    for base in bundled_roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".npz", ".yaml", ".md"}:
                bundled.append({
                    "path": rel(path), "status": "bundled", "role": "frozen_aggregate",
                    "sha256": sha256(path), "bytes": path.stat().st_size,
                    "source_path": rel(path), "derived_from": "",
                    "required_by": "figure/table reproduction",
                })
    # The geometry context is a package input and is also bundled.
    geom = ROOT / "routing_urban_medium_constitutive/results/um_ctx.npz"
    if geom.is_file():
        bundled.append({
            "path": rel(geom), "status": "bundled", "role": "geometry",
            "sha256": sha256(geom), "bytes": geom.stat().st_size,
            "source_path": rel(geom), "derived_from": "",
            "required_by": "urbanflow geometry",
        })
    external = []
    if EXTERNAL.is_file():
        with EXTERNAL.open(encoding="utf-8", newline="") as fh:
            external = list(csv.DictReader(fh))
    data_rows = bundled + external
    # Keep one row per path; bundled wins over an external duplicate.
    seen, unique = set(), []
    for row in data_rows:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        unique.append(row)
    with (MAN / "data.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "path", "status", "role", "sha256", "bytes", "source_path",
            "derived_from", "required_by"])
        writer.writeheader()
        writer.writerows(unique)

    fig_map = [
        ("fig01_concept", "main", "scripts/reproduce_figures/redraw_urbanflow.py", "concept"),
        ("fig02_threecity_coupling", "main", "scripts/reproduce_figures/redraw_urbanflow.py", "threecity"),
        ("fig03_frequency_response", "main", "scripts/reproduce_figures/fig_phase_response.py", "phase-sensitive"),
        ("fig04_anisotropy_road_tests", "main", "scripts/reproduce_figures/fig5_redesign.py", "B"),
        ("figS01_joint_spectrum_2d", "supp", "scripts/reproduce_figures/make_figures.py", "S01"),
        ("figS02_heldout_daily_scores", "supp", "scripts/reproduce_figures/make_figures.py", "S02"),
        ("figS03_coherent_response_train_test", "supp", "scripts/reproduce_figures/make_figures.py", "S03"),
        ("figS04_anisotropy_distributions", "supp", "scripts/reproduce_figures/make_figures.py", "S04"),
        ("figS05_matched_capacity", "supp", "scripts/reproduce_figures/make_figures.py", "S05"),
        ("figS06_heldout_reconstruction", "supp", "scripts/reproduce_figures/make_figures.py", "S06"),
        ("figS07_nonlinear_exploratory", "supp", "scripts/reproduce_figures/make_figures.py", "S07"),
    ]
    fig_rows = []
    for name, group, generator, entry in fig_map:
        path = ROOT / "figures" / ("main" if group == "main" else "supplementary") / f"{name}.pdf"
        fig_rows.append({
            "figure": name, "group": group, "generator": generator, "entry": entry,
            "reference_pdf": rel(path), "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "verification": "reference PDF bundled; redraw equivalence not asserted byte-for-byte",
        })
    with (MAN / "figures.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fig_rows[0]))
        writer.writeheader()
        writer.writerows(fig_rows)

    frozen = sorted((ROOT / "results" / "frozen").rglob("*"))
    with (MAN / "checksums.sha256").open("w", encoding="utf-8") as fh:
        for path in frozen:
            if path.is_file():
                fh.write(f"{sha256(path)}  {rel(path)}\n")
        for path in sorted((ROOT / "figures").rglob("*.pdf")):
            fh.write(f"{sha256(path)}  {rel(path)}\n")

    provenance = {
        "export_schema": 2,
        "layout": "src/urbanflow + experiments + results/frozen + figures",
        "reference_digest": "b7b85d7864eaf93c7916b1f4dfd24e9d23a4c7c807457c5767558905a9860566",
        "upstream_reference_digest": "833a07ff7d5e9adbabef6818f6a36c10a8dfd30f521841eea7c8ffafe7676ceb",
        "path_policy": "URBANFLOW_ROOT or <repo>/external_data; no workstation path fallback",
        "source_policy": "aggregate frozen products only; no raw trajectories or model weights",
        "paper_sync": "paper sources are preserved; figure files are canonical under figures/",
        "manifest_files": sorted(p.name for p in MAN.iterdir() if p.is_file()),
    }
    (MAN / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    print(json.dumps({"files": len(file_rows), "data_rows": len(unique),
                      "figures": len(fig_rows), "checksums": len(frozen)}, indent=2))


if __name__ == "__main__":
    main()
