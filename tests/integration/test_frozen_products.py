"""Integration checks over bundled frozen products."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class FrozenProducts(unittest.TestCase):
    def test_figure_manifest_matches_files(self):
        with (ROOT / "data_manifest/figures.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12)
        for row in rows:
            path = ROOT / row["reference_pdf"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, int(row["bytes"]))
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_frozen_checksums(self):
        listing = ROOT / "data_manifest/checksums.sha256"
        for line in listing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split("  ", 1)
            path = ROOT / rel
            self.assertTrue(path.is_file(), rel)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, rel)

    def test_result_families_exist(self):
        for name in ("crosscity", "constitutive", "spectra", "anisotropy",
                     "baselines", "matched_capacity", "reconstruction"):
            self.assertTrue((ROOT / "results" / "frozen" / name).is_dir(), name)


if __name__ == "__main__":
    unittest.main()
