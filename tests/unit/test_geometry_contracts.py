"""Algebraic contracts that do not need external trajectories."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urbanflow import geometry  # noqa: E402


class GeometryContracts(unittest.TestCase):
    KEY = "unit-square"

    def setUp(self):
        ny, nx = 2, 3
        p0, p1 = [], []
        for y in range(ny):
            for x in range(nx - 1):
                a = y * nx + x
                p0.append(a); p1.append(a + 1)
        for y in range(ny - 1):
            for x in range(nx):
                a = y * nx + x
                p0.append(a); p1.append(a + nx)
        geometry._CTX[self.KEY] = dict(
            ny=ny, nx=nx, NC=ny * nx, n_int=len(p0),
            p0=np.asarray(p0, dtype=np.int64), p1=np.asarray(p1, dtype=np.int64))
        geometry._LU.pop(self.KEY, None)

    def test_divergence_matches_incidence(self):
        v = np.zeros((7, 1)); v[0, 0] = 1.0
        self.assertEqual(geometry.D(self.KEY).shape, (6, 7))
        expected = np.zeros((6, 1))
        expected[0, 0], expected[1, 0] = 1.0, -1.0
        np.testing.assert_allclose(geometry.D(self.KEY) @ v, expected, atol=1e-12)

    def test_hodge_reconstruction(self):
        rng = np.random.default_rng(20261031)
        v = rng.normal(size=(2, 7))
        div = np.asarray(geometry.D(self.KEY) @ v.T).T
        circulation = geometry.P0(self.KEY, v)
        gradient = geometry.ddag(self.KEY, div)
        np.testing.assert_allclose(circulation + gradient, v, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(geometry.D(self.KEY) @ circulation.T), 0.0, atol=1e-12)

    def test_road_basis_is_divergence_free_and_orthonormal(self):
        rng = np.random.default_rng(7)
        raw = geometry.P0(self.KEY, rng.normal(size=(3, 7)))
        q, _ = np.linalg.qr(raw.T)
        u = q[:, :2]
        np.testing.assert_allclose(
            np.asarray(geometry.D(self.KEY) @ u), 0.0, atol=1e-12)
        np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
