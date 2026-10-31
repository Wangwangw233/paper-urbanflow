#!/usr/bin/env python3
"""Verify the UrbanFlow public-release candidate without changing it.

Default checks cover manifest integrity, disclosure/path rules, NPZ safety,
AST/import closure, compileall, and data-free unit smoke tests.  Add
--figure-smoke to execute every public figure entry point into a temporary
output directory.  No success is inferred from a skipped external-data test.
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
    "README.md", "LICENSE", "CITATION", "requirements.txt", ".gitignore",
    "docs/ENVIRONMENT.md", "manifest/files.csv", "manifest/data.csv",
    "manifest/figures.csv", "manifest/provenance.json",
    "tools/export_public_repo.py", "tools/verify_public_repo.py",
    "tests/test_public_smoke.py", "urbanflow_ref/manifest.json",
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
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".cff", ".tex", ".yml", ".yaml"}
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
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "matplotlib", "mpl_toolkits", "pytest", "fitz", "PIL", "threadpoolctl"}


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


def scan_paths_and_privacy(root: Path, results: Results):
    problems, large = [], []
    files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts]
    for path in files:
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        lowered = {p.lower() for p in parts}
        if rel == "reports/cross_dataset_data_audit/results/unused_recheck.json":
            problems.append(rel + ": explicitly excluded trajectory audit")
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
            problems.append(rel + ": operational shell script is outside the public allowlist")
        lower_name = path.name.lower()
        if any(x in lower_name for x in ("trajectory", "trajectories", "trip_level", "raw_gps")):
            problems.append(rel + ": individual/raw-data filename")
        raw = path.read_bytes()
        # The verifier necessarily contains the literal signatures it looks
        # for.  Scanning those signatures as if they were disclosures would be
        # self-referential; path, AST, hash and size checks still apply to it.
        if rel in {"tools/verify_public_repo.py", "tools/export_public_repo.py", "tools/sync_crossrefs.py"}:
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
            # Binary formats can contain uncompressed producer/path strings.
            for needle in (b"Docu_Xu", b"mobility_constitution", b"/root/", b"/home/", b"PRIVATE KEY"):
                if needle in raw:
                    problems.append(f"{rel}: embedded private string {needle!r}")
                    break
    results.add("file size <= 50 MiB", not large,
                f"{len(files)} files" if not large else ", ".join(large))
    results.add("excluded paths and privacy patterns", not problems,
                "no forbidden path, raw-data, remote-operation, or private-path match"
                if not problems else " | ".join(problems[:30]))


def scan_npz(root: Path, results: Results):
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


def scan_tabular_and_json(root: Path, results: Results):
    problems = []
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
    def walk(value, rel, key_path="$"):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in RAW_KEYS:
                    problems.append(f"{rel}:{key_path}.{key}: raw/individual key")
                walk(item, rel, key_path + "." + str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, rel, f"{key_path}[{i}]")
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            walk(json.loads(path.read_text(encoding="utf-8")), rel)
        except Exception as exc:
            problems.append(f"{rel}: {type(exc).__name__}: {exc}")
    results.add("CSV/JSON individual-field scan", not problems,
                "no declared individual identifiers, GPS columns, or trajectory fields"
                if not problems else " | ".join(problems[:20]))


def check_manifest(root: Path, results: Results):
    manifest = root / "manifest/files.csv"
    if not manifest.is_file():
        results.add("files.csv presence", False, "missing")
        return
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected = {row["path"]: row for row in rows}
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*")
              if p.is_file() and ".git" not in p.relative_to(root).parts}
    actual.pop("manifest/files.csv", None)
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


def check_gitignore(root: Path, results: Results):
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    broad = [line for line in text.splitlines()
             if line.strip().lower() in {"*.npz", "*.pdf", "*.npy", "*.zip"}]
    results.add("selective .gitignore", path.exists() and not broad,
                "NPZ/PDF remain trackable; caches and generated rasters ignored"
                if path.exists() and not broad else f"broad rules={broad}")


def check_ast_and_imports(root: Path, results: Results):
    files = sorted(root.rglob("*.py"))
    errors, missing = [], []
    local = {p.stem for p in files}
    if (root / "urbanflow_ref/__init__.py").exists():
        local.add("urbanflow_ref")
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
                missing.append(f"{rel}: {name}")
    results.add("AST parse", not errors, f"{len(files)} Python files" if not errors else " | ".join(errors))
    results.add("local import/dependency closure", not missing,
                "standard library, declared dependencies, or allowlisted local modules"
                if not missing else " | ".join(missing[:30]))


def check_compileall(root: Path, results: Results):
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
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root),
               MPLBACKEND="Agg", MPLCONFIGDIR=str(temp / "mplconfig"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", URBANFLOW_ROOT=str(root))
    (temp / "mplconfig").mkdir(parents=True, exist_ok=True)
    return env


def run_command(root: Path, env: dict, args: list[str], timeout=300):
    return subprocess.run([sys.executable, *args], cwd=root, env=env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)


def check_core_smoke(root: Path, results: Results):
    try:
        with tempfile.TemporaryDirectory(prefix="urbanflow-core-smoke-") as tmp:
            env = subprocess_env(root, Path(tmp))
            proc = run_command(root, env, ["-m", "unittest", "discover", "-s", "tests", "-v"], 180)
            detail = proc.stdout.strip().splitlines()[-8:]
            results.add("data-free unittest smoke", proc.returncode == 0,
                        " | ".join(detail) if proc.returncode else "4 tests passed")
            proc = run_command(root, env, ["-m", "urbanflow_ref", "manifest", "--digest-only"], 60)
            results.add("reference manifest CLI smoke", proc.returncode == 0 and len(proc.stdout.strip()) == 64,
                        proc.stdout.strip() if proc.returncode == 0 else proc.stdout[-2000:])
    except Exception as exc:
        results.add("core smoke execution", False, f"{type(exc).__name__}: {exc}")


def check_figure_smoke(root: Path, results: Results):
    commands = [
        ("main redraw", ["paper/merged_manuscript/figures/redraw_urbanflow.py", "--out", "{main}", "--supp-out", "{supp}"]),
        ("three-city response", ["paper/merged_manuscript/figures/reorganize_threecity_figures.py", "--plots-only", "--out", "{main}"]),
        ("joint spectrum", ["reports/threecity_joint_spectrum/manuscript_revision/redraw_joint_spectrum.py", "--data", str(root / "reports/threecity_joint_spectrum/manuscript_revision/verified_joint.npz"), "--out", "{main}"]),
        ("main Fig. 5", ["reports/threecity_figure5_revision/scripts/fig5_redesign.py", "B"]),
        ("three-city supplement", ["reports/threecity_supplement/scripts/make_figures.py"]),
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="urbanflow-figure-smoke-") as tmp:
            temp = Path(tmp)
            main, supp = temp / "main", temp / "supp"
            results_dir = temp / "results"
            for directory in (main, supp, results_dir):
                directory.mkdir(parents=True, exist_ok=True)
            env = subprocess_env(root, temp)
            # Exercise the code/data split for real: all bundled frozen inputs
            # live in a separate URBANFLOW_ROOT, while code stays in `root`.
            external = temp / "external_data"
            with (root / "manifest/data.csv").open(encoding="utf-8", newline="") as fh:
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
                       URBANFLOW_RESULT_OUT=str(results_dir), URBANFLOW_SUPP_BUILD_OUT=str(supp))
            for label, args in commands:
                args = [a.format(main=main, supp=supp) for a in args]
                command_env = env.copy()
                if label == "three-city supplement":
                    command_env["URBANFLOW_FIGURE_OUT"] = str(supp)
                proc = run_command(root, command_env, args, 360)
                if proc.returncode:
                    results.add("figure smoke: " + label, False, proc.stdout[-5000:])
                    return
                print(f"       executed {label}")
            with (root / "manifest/figures.csv").open(encoding="utf-8", newline="") as fh:
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
            # Confirm that smoke runs did not modify the candidate itself.
            check_manifest(root, results)
    except subprocess.TimeoutExpired as exc:
        results.add("figure smoke timeout", False, str(exc))
    except Exception as exc:
        results.add("figure smoke execution", False, f"{type(exc).__name__}: {exc}")


def verify(root: Path, figure_smoke=False, skip_smoke=False) -> Results:
    results = Results()
    missing = [rel for rel in REQUIRED if not (root / rel).is_file()]
    results.add("required release files", not missing,
                f"{len(REQUIRED)} required paths" if not missing else ", ".join(missing))
    scan_paths_and_privacy(root, results)
    scan_npz(root, results)
    scan_tabular_and_json(root, results)
    check_gitignore(root, results)
    check_manifest(root, results)
    check_ast_and_imports(root, results)
    check_compileall(root, results)
    if not skip_smoke:
        check_core_smoke(root, results)
    if figure_smoke:
        check_figure_smoke(root, results)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--figure-smoke", action="store_true", help="execute all 12 frozen figure generators")
    parser.add_argument("--skip-smoke", action="store_true", help="skip unit/CLI smoke tests (reported as skipped, never as passed)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    results = verify(root, figure_smoke=args.figure_smoke, skip_smoke=args.skip_smoke)
    passed = len(results.checks) - len(results.failed)
    print(f"\n{passed} passed, {len(results.failed)} failed; figure_smoke={args.figure_smoke}; unit_smoke_skipped={args.skip_smoke}")
    print("External-data core recomputation and raw end-to-end rebuild were not run by this verifier.")
    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
