#!/usr/bin/env python3
"""Build machine-readable table products from frozen scores.

The script never types a measured value.  It exports the inference block that
is present in ``results/frozen/crosscity/crosscity_results.csv`` and the
primary D intervals from ``results/frozen/constitutive/d_uncertainty.json``.
Graph WaveNet rows are not regenerated here because their per-date errors are
not bundled; the manuscript table records them as archived values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "frozen"
DEFAULT_OUT = ROOT / "results" / "derived"

CITIES = ("Porto", "SF", "Chengdu")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (FROZEN / "crosscity" / "crosscity_results.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    inference = []
    for city in CITIES:
        for model in ("OURS", "RIDGE", "POD"):
            matches = [r for r in rows if r["object"] == city and r["frame"] == "P"
                       and r["support"] == "FACE" and r["model"] == model]
            if not matches:
                continue
            r = matches[0]
            inference.append({
                "city": city,
                "model": "UrbanFlow" if model == "OURS" else model.lower(),
                "frame": "P",
                "support": "FACE",
                "pooled_R": r["pooled_R"],
                "n_folds": r["n_folds"],
            })
    with (out / "table1_inference_scores.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(inference[0]))
        writer.writeheader()
        writer.writerows(inference)

    uncertainty = json.loads((FROZEN / "constitutive" / "d_uncertainty.json").read_text(
        encoding="utf-8"))["cities"]
    with (out / "table_d_uncertainty.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["city", "D_ROAD", "ci_low", "ci_high", "n_test_dates", "pairing_p"])
        for city in CITIES:
            rec = uncertainty[city]
            lo, hi = rec["bootstrap_ci95"]
            writer.writerow([city, rec["real_D"], lo, hi,
                             rec["total_test_dates"], rec.get("pairing_p")])
    print(f"wrote {out / 'table1_inference_scores.csv'}")
    print(f"wrote {out / 'table_d_uncertainty.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().out)
