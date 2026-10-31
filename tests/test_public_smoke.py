"""Smoke tests that require no external scientific data."""
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
