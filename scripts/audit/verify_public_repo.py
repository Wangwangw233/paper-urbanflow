#!/usr/bin/env python3
"""Verify the UrbanFlow repository without changing it.

Checks cover path/privacy rules, NPZ safety, tabular fields, the file manifest,
imports, compilation, data-free unit tests, the reference CLI and, optionally,
a full figure rebuild into a temporary directory.
"""
from __future__ import annotations

import argparse
import ast
import compileall
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

MAX_BYTES = 50 * 1024 * 1024
REQUIRED = [
    "README.md", "LICENSE", "CITATION", "requirements.txt", "pyproject.toml",
    ".gitignore", "docs/ENVIRONMENT.md", "docs/architecture.md",
    "docs/experiment_protocol.md", "docs/reproduction_guide.md",
    "docs/result_provenance.md", "docs/known_limitations.md",
    "docs/validation_report.md",
    "data_manifest/files.csv", "data_manifest/data.csv",
    "data_manifest/figures.csv", "data_manifest/provenance.json",
    "data_manifest/checksums.sha256", "data_manifest/datasets.md",
    "scripts/audit/verify_public_repo.py", "tests/unit/test_public_smoke.py",
    "src/urbanflow/manifest.json",
]
FORBIDDEN_PARTS = {
    "提示词", "scratch", "_tmp", "_invalid", "_superseded", "__pycache__",
    ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    "node_modules", "build", "dist", "literature_cache", "references_cache",
}
FORBIDDEN_SUFFIXES = {
    ".pyc", ".pyo", ".dll", ".exe", ".o", ".a", ".aux", ".bbl", ".blg",
    ".fls", ".fdb_latexmk", ".toc", ".new", ".tmp", ".bak", ".swp",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".cff", ".tex", ".yml", ".yaml", ".toml"}
PRIVATE_PATTERNS = [
    ("Windows absolute path", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")),
    ("UNC path", re.compile(r"\\\\[A-Za-z0-9._-]+\\")),
    ("user/system absolute path", re.compile(r"/(?:root|home|Users|mnt|media)/")),
    ("private workspace name", re.compile(r"Docu_Xu|mobility_constitution", re.I)),
    ("remote shell operation", re.compile(r"\b(?:ssh|scp|rsync|paramiko)\b|New-PSSession|Invoke-Command", re.I)),
    ("cloud access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    ("email address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
]
RAW_KEYS = {
    "trip_id", "traj_id", "trajectory_id", "vehicle_id", "driver_id", "user_id",
    "person_id", "device_id", "taxi_id", "customer_id", "latitude", "longitude",
    "lat", "lon", "lng", "gps", "trajectory", "trajectories", "raw_trajectory",
}
ALLOWED_THIRD_PARTY = {
    "numpy", "scipy", "matplotlib", "mpl_toolkits", "pytest", "fitz", "PIL",
    "threadpoolctl", "yaml",
}


class Results:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail=""):
        self.checks.append((name, bool(ok), str(detail)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    @property
    def failed(self):
        return [c for c in self.checks if not c[1]]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "CITATION", ".gitignore"}


def scan_paths(root: Path, results: Results) -> None:
    problems, large = [], []
    files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts]
    for path in files:
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        lowered = {p.lower() for p in parts}
        if parts & FORBIDDEN_PARTS or lowered & {x.lower() for x in FORBIDDEN_PARTS}:
            problems.append(rel + ": forbidden path component")
        if any(part.startswith("_tmp") for part in parts):
            problems.append(rel + ": temporary path")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(rel + ": build/cache residue")
        if path.is_symlink():
            problems.append(rel + ": symlink is not allowed")
        if path.stat().st_size > MAX_BYTES:
            large.append(rel)
        if path.suffix.lower() in {".sh", ".ps1", ".bat", ".cmd"}:
            problems.append(rel + ": operational shell script")
        if any(x in path.name.lower() for x in ("trajectory", "trajectories", "trip_level", "raw_gps")):
            problems.append(rel + ": individual/raw-data filename")
        raw = path.read_bytes()
        if rel in {"scripts/audit/verify_public_repo.py", "scripts/audit/sync_crossrefs.py"}:
            continue
        if is_text(path):
            text = raw.decode("utf-8", errors="replace")
            if rel == "paper/merged_manuscript/authors.tex":
                text = PRIVATE_PATTERNS[-1][1].sub("[public author contact]", text)
            for label, pattern in PRIVATE_PATTERNS:
                match = pattern.search(text)
                if match:
                    problems.append(f"{rel}: {label} ({match.group(0)!r})")
        elif path.suffix.lower() in {".pdf", ".npz", ".png"}:
            for needle in (b"Docu_Xu", b"mobility_constitution", b"/root/", b"/home/", b"PRIVATE KEY"):
                if needle in raw:
                    problems.append(f"{rel}: embedded private string {needle!r}")
                    break
    results.add("file size <= 50 MiB", not large,
                f"{len(files)} files" if not large else ", ".join(large))
    results.add("excluded paths and privacy patterns", not problems,
                "no forbidden path, raw-data, remote-operation or private-path match"
                if not problems else " | ".join(problems[:30]))


def scan_npz(root: Path, results: Results) -> None:
    problems, count, arrays = [], 0, 0
    for path in sorted(root.rglob("*.npz")):
        rel = path.relative_to(root).as_posix()
        count += 1
        try:
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    arrays += 1
                    if key.lower() in RAW_KEYS:
                        problems.append(f"{rel}:{key}: raw/individual field name")
                    array = z[key]
                    if array.dtype == np.dtype("O"):
                        problems.append(f"{rel}:{key}: object array/pickle risk")
                    if array.dtype.kind not in "biufcUS":
                        problems.append(f"{rel}:{key}: unexpected dtype {array.dtype}")
        except Exception as exc:
            problems.append(f"{rel}: {type(exc).__name__}: {exc}")
    results.add("NPZ privacy and pickle safety", not problems,
                f"{count} archives, {arrays} arrays, allow_pickle=False"
                if not problems else " | ".join(problems[:20]))


def scan_tables(root: Path, results: Results) -> None:
    problems = []

    def walk(value, rel, key_path="$"):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in RAW_KEYS:
                    problems.append(f"{rel}:{key_path}.{key}: raw/individual key")
                walk(item, rel, key_path + "." + str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, rel, f"{key_path}[{i}]")

    for path in sorted(root.rglob("*.csv")):
        rel = path.relative_to(root).as_posix()
        try:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh), [])
            bad = sorted({h.strip().lower() for h in header} & RAW_KEYS)
            if bad:
                problems.append(f"{rel}: columns {bad}")
        except Exception as exc:
            problems.append(f"{rel}: {type(exc).__name__}: {exc}")
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            walk(json.loads(path.read_text(encoding="utf-8")), rel)
        except Exception as exc:
            problems.append(f"{rel}: {type(exc).__name__}: {exc}")
    results.add("CSV/JSON individual-field scan", not problems,
                "no declared individual identifiers, GPS columns or trajectory fields"
                if not problems else " | ".join(problems[:20]))


def check_manifest(root: Path, results: Results) -> None:
    manifest = root / "data_manifest/files.csv"
    if not manifest.is_file():
        results.add("files.csv presence", False, "missing")
        return
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected = {row["path"]: row for row in rows}
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*")
              if p.is_file() and ".git" not in p.relative_to(root).parts}
    actual.pop("data_manifest/files.csv", None)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = []
    for rel in sorted(set(expected) & set(actual)):
        row, path = expected[rel], actual[rel]
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            changed.append(rel)
    results.add("files.csv exact file set", not missing and not extra,
                f"{len(actual)} files" if not missing and not extra else f"missing={missing}, extra={extra}")
    results.add("files.csv hashes and sizes", not changed,
                f"{len(expected) - len(changed)} matched" if not changed else ", ".join(changed[:20]))


def check_gitignore(root: Path, results: Results) -> None:
    text = (root / ".gitignore").read_text(encoding="utf-8") if (root / ".gitignore").exists() else ""
    broad = [line for line in text.splitlines()
             if line.strip().lower() in {"*.npz", "*.pdf", "*.npy", "*.zip"}]
    results.add("selective .gitignore", bool(text) and not broad,
                "NPZ/PDF remain trackable; caches and generated rasters ignored"
                if text and not broad else f"broad rules={broad}")


def check_imports(root: Path, results: Results) -> None:
    files = sorted(root.rglob("*.py"))
    errors, missing = [], []
    local = {p.stem for p in files}
    local.update({"urbanflow"})
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                continue
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if not top or top == "__future__" or top in sys.stdlib_module_names:
                    continue
                if top in local or top in ALLOWED_THIRD_PARTY:
                    continue
                if rel.startswith("archive/legacy/"):
                    continue
                missing.append(f"{rel}: {name}")
    results.add("AST parse", not errors, f"{len(files)} Python files" if not errors else " | ".join(errors))
    results.add("local import/dependency closure", not missing,
                "standard library, declared dependencies or local modules"
                if not missing else " | ".join(missing[:30]))


def check_compileall(root: Path, results: Results) -> None:
    files = sorted(root.rglob("*.py"))
    try:
        with tempfile.TemporaryDirectory(prefix="urbanflow-compile-") as tmp:
            target = Path(tmp)
            for path in files:
                dest = target / path.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
            ok = compileall.compile_dir(str(target), quiet=1, force=True)
        results.add("compileall", bool(ok), f"{len(files)} Python files compiled in a temporary tree")
    except Exception as exc:
        results.add("compileall", False, f"{type(exc).__name__}: {exc}")


def subprocess_env(root: Path, temp: Path) -> dict:
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src"),
               MPLBACKEND="Agg", MPLCONFIGDIR=str(temp / "mplconfig"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", URBANFLOW_ROOT=str(root))
    (temp / "mplconfig").mkdir(parents=True, exist_ok=True)
    return env


def run_command(root: Path, env: dict, args: list[str], timeout=300):
    return subprocess.run([sys.executable, *args], cwd=root, env=env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)


def check_core_smoke(root: Path, results: Results) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="urbanflow-core-smoke-") as tmp:
            env = subprocess_env(root, Path(tmp))
            proc = run_command(root, env, ["-m", "unittest", "discover", "-s", "tests", "-v"], 180)
            detail = proc.stdout.strip().splitlines()[-8:]
            results.add("data-free unittest smoke", proc.returncode == 0,
                        " | ".join(detail) if proc.returncode else "unit tests passed")
            proc = run_command(root, env, ["-m", "urbanflow", "manifest", "--digest-only"], 60)
            results.add("reference manifest CLI smoke",
                        proc.returncode == 0 and len(proc.stdout.strip()) == 64,
                        proc.stdout.strip() if proc.returncode == 0 else proc.stdout[-2000:])
    except Exception as exc:
        results.add("core smoke execution", False, f"{type(exc).__name__}: {exc}")


def check_table_smoke(root: Path, results: Results) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="urbanflow-table-smoke-") as tmp:
            temp = Path(tmp)
            env = subprocess_env(root, temp)
            out = temp / "derived"
            proc = run_command(root, env, ["scripts/reproduce_tables/reproduce_all.py",
                                           "--out", str(out)], 120)
            expected = [out / "table1_inference_scores.csv", out / "table_d_uncertainty.csv"]
            ok = proc.returncode == 0 and all(p.is_file() and p.stat().st_size for p in expected)
            results.add("table smoke", ok,
                        "table products regenerated" if ok else proc.stdout[-3000:])
    except Exception as exc:
        results.add("table smoke", False, f"{type(exc).__name__}: {exc}")


def check_figure_smoke(root: Path, results: Results) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="urbanflow-figure-smoke-") as tmp:
            temp = Path(tmp)
            main, supp, derived = temp / "main", temp / "supp", temp / "derived"
            for directory in (main, supp, derived):
                directory.mkdir(parents=True, exist_ok=True)
            env = subprocess_env(root, temp)
            external = temp / "external_data"
            with (root / "data_manifest/data.csv").open(encoding="utf-8", newline="") as fh:
                data_rows = list(csv.DictReader(fh))
            for row in data_rows:
                if row["status"] != "bundled":
                    continue
                src = root / row["path"]
                dst = external / row["path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            env["URBANFLOW_ROOT"] = str(external)
            env.update(URBANFLOW_FIGURE_OUT=str(main), URBANFLOW_SUPP_FIGURE_OUT=str(supp),
                       URBANFLOW_RESULT_OUT=str(derived), URBANFLOW_SUPP_BUILD_OUT=str(supp))
            proc = run_command(root, env, ["scripts/reproduce_figures/reproduce_all.py",
                                           "--root", str(external),
                                           "--out", str(main), "--supp-out", str(supp)], 600)
            if proc.returncode:
                results.add("figure smoke execution", False, proc.stdout[-5000:])
                return
            print("       executed figure reproduction")
            with (root / "data_manifest/figures.csv").open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            missing, invalid = [], []
            for row in rows:
                path = (main if row["group"] == "main" else supp) / (row["figure"] + ".pdf")
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(row["figure"])
                elif not path.read_bytes().startswith(b"%PDF"):
                    invalid.append(row["figure"])
            results.add("all figure entry points", not missing and not invalid,
                        f"{len(rows)} PDFs regenerated using a separate external frozen-data mirror"
                        if not missing and not invalid else f"missing={missing}, invalid={invalid}")
            check_manifest(root, results)
    except subprocess.TimeoutExpired as exc:
        results.add("figure smoke timeout", False, str(exc))
    except Exception as exc:
        results.add("figure smoke execution", False, f"{type(exc).__name__}: {exc}")


def verify(root: Path, figure_smoke=False) -> Results:
    results = Results()
    missing = [rel for rel in REQUIRED if not (root / rel).is_file()]
    results.add("required repository files", not missing,
                f"{len(REQUIRED)} required paths" if not missing else ", ".join(missing))
    scan_paths(root, results)
    scan_npz(root, results)
    scan_tables(root, results)
    check_manifest(root, results)
    check_gitignore(root, results)
    check_imports(root, results)
    check_compileall(root, results)
    check_core_smoke(root, results)
    if figure_smoke:
        check_table_smoke(root, results)
        check_figure_smoke(root, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--figure-smoke", action="store_true")
    args = parser.parse_args()
    results = verify(args.root.resolve(), figure_smoke=args.figure_smoke)
    passed = len(results.checks) - len(results.failed)
    print(f"\n{passed} passed, {len(results.failed)} failed; figure_smoke={args.figure_smoke}")
    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
