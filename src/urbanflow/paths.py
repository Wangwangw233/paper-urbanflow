"""Resolve external frozen data without a workstation-specific default."""
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
