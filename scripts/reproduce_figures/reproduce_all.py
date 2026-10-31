#!/usr/bin/env python3
"""Regenerate the four main and seven supplementary figures from frozen data.

This entry point does not fit or refit any model.  The figure generators read
their inputs from ``results/frozen`` (or from the external mirror selected by
``URBANFLOW_ROOT``) and write PDFs/PNGs to ``figures/main`` and
``figures/supplementary`` by default.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="frozen-data root (default: repository root)")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "main")
    parser.add_argument("--supp-out", type=Path, default=ROOT / "figures" / "supplementary")
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("URBANFLOW_ROOT", str(args.root or ROOT))
    env["URBANFLOW_FIGURE_OUT"] = str(args.out)
    env["URBANFLOW_SUPP_FIGURE_OUT"] = str(args.supp_out)
    env["URBANFLOW_SUPP_BUILD_OUT"] = str(args.supp_out)
    args.out.mkdir(parents=True, exist_ok=True)
    args.supp_out.mkdir(parents=True, exist_ok=True)
    commands = [
        ("main redraw", [sys.executable, "scripts/reproduce_figures/redraw_urbanflow.py",
                         "--out", str(args.out), "--supp-out", str(args.supp_out)]),
        ("phase-sensitive response", [sys.executable, "scripts/reproduce_figures/fig_phase_response.py"]),
        ("main Fig. 4", [sys.executable, "scripts/reproduce_figures/fig5_redesign.py", "B"]),
        ("three-city supplement", [sys.executable, "scripts/reproduce_figures/make_figures.py"]),
    ]
    supplement_env = env.copy()
    supplement_env["URBANFLOW_FIGURE_OUT"] = str(args.supp_out)
    for label, command in commands:
        print(f"[reproduce] {label}", flush=True)
        command_env = supplement_env if label == "three-city supplement" else env
        proc = subprocess.run(command, cwd=ROOT, env=command_env, text=True)
        if proc.returncode:
            print(f"[reproduce] failed: {label}", file=sys.stderr)
            return proc.returncode
    print("[reproduce] all figure generators completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
