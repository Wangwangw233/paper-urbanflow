"""Grid geometry, the discrete divergence `D`, and the circulation projector.

This is the audited round-3 implementation (`audit_lib.D / ddag / P0 /
load_flux`), vendored so the reference package is importable from anywhere and
can never pick up a different copy by accident.  `tests` compares it against
`audit_lib` element by element, so "vendored" is a claim with a check, not a
fork.

The conserved field is a face flux: one scalar per interior edge, `NC` cells,
`n_int` faces.  `D` is the `NC x n_int` incidence matrix (`+1` on the head
cell, `-1` on the tail).  `P0 = I - D^+ D` projects onto `ker D`, the
divergence-free (circulation) subspace.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from . import paths

UM_CTX = "routing_urban_medium_constitutive/results/um_ctx.npz"

_CTX: dict = {}
_LU: dict = {}


# --------------------------------------------------------------------------
# grid and face ordering
# --------------------------------------------------------------------------
def ctx(key):
    """`dict(ny, nx, NC, n_int, p0, p1)` for a geometry key in `um_ctx.npz`."""
    if key not in _CTX:
        z = np.load(paths.resolve(UM_CTX), allow_pickle=True)
        _CTX[key] = dict(
            ny=int(z[key + "__ny"]), nx=int(z[key + "__nx"]),
            NC=int(z[key + "__NC"]), n_int=int(z[key + "__n_int"]),
            p0=np.asarray(z[key + "__face_pairs_p0"], np.int64),
            p1=np.asarray(z[key + "__face_pairs_p1"], np.int64))
    return _CTX[key]


def D(key):
    """Face-to-cell incidence matrix, `(NC, n_int)` CSR."""
    k = ("D", key)
    if k not in _CTX:
        c = ctx(key)
        f = np.arange(c["n_int"], dtype=np.int64)
        _CTX[k] = sp.csr_matrix(
            (np.concatenate([np.ones(f.size), -np.ones(f.size)]),
             (np.concatenate([c["p0"], c["p1"]]),
              np.concatenate([f, f]))),
            shape=(c["NC"], c["n_int"]))
    return _CTX[k]


def _lap_lu(key):
    if key not in _LU:
        Dm = D(key)
        L = (Dm @ Dm.T).tocsc()
        keep = np.arange(1, L.shape[0])          # cell 0 is the gauge
        _LU[key] = (keep, splu(L[1:, 1:].tocsc()))
    return _LU[key]


def pinv_apply(key, rhs):
    keep, lu = _lap_lu(key)
    R = np.atleast_2d(np.asarray(rhs, np.float64))
    out = np.zeros_like(R)
    out[:, keep] = lu.solve(np.ascontiguousarray(R[:, keep].T)).T
    return out


def ddag(key, x):
    """`D^+ x` (minimum-norm least squares), gauge cell 0 fixed at zero."""
    X = np.atleast_2d(np.asarray(x, np.float64))
    assert X.shape[1] == ctx(key)["NC"], (X.shape, ctx(key)["NC"])
    return (D(key).T @ pinv_apply(key, X).T).T


def P0(key, v):
    """Project a cell field onto `ker D`:  `v - D^+ (D v)`."""
    V = np.atleast_2d(np.asarray(v, np.float64))
    return V - ddag(key, (D(key) @ V.T).T)


# --------------------------------------------------------------------------
# the aligned cache
# --------------------------------------------------------------------------
def load_flux(relpath, key):
    """Observed face flux `(K, n_int)` plus its window metadata.

    The face ordering is rebuilt from the grid mask and **asserted** equal to
    `um_ctx`'s stored `p0/p1`; a cache with a different ordering cannot be
    silently evaluated.
    """
    z = np.load(paths.resolve(relpath), allow_pickle=True)
    FE = np.asarray(z["F_E"], np.float64)
    FN = np.asarray(z["F_N"], np.float64)
    meta = dict(wdate=np.asarray(z["wdate"], np.int64),
                wslot=np.asarray(z["wslot"], np.int64),
                wvalid=np.asarray(z["wvalid"], bool),
                times=np.asarray(z["times"], np.int64),
                scale_m=float(z["scale_m"]), l0_m=float(z["l0_m"]),
                dt_min=int(z["dt_min"]))
    if "udates" in z.files:
        meta["udates"] = np.asarray(z["udates"], np.int64)
    z.close()
    K = FE.shape[0]
    both = np.concatenate([FE.reshape(K, -1), FN.reshape(K, -1)], axis=1)
    c = ctx(key)
    assert (FE.shape[1], FE.shape[2]) == (c["ny"], c["nx"]), \
        f"cache grid {(FE.shape[1], FE.shape[2])} != ctx {(c['ny'], c['nx'])}"
    col = np.arange(c["NC"])
    m = np.concatenate([(col % c["nx"]) < (c["nx"] - 1),
                        (col // c["nx"]) < (c["ny"] - 1)])
    assert int(m.sum()) == c["n_int"]
    idx = np.flatnonzero(m)
    e = idx[idx < c["NC"]]
    n = idx[idx >= c["NC"]] - c["NC"]
    assert np.array_equal(np.concatenate([e, n]), c["p0"])
    assert np.array_equal(np.concatenate([e + 1, n + c["nx"]]), c["p1"])
    return both[:, m], meta


def load_basis(spec):
    """`(U, r)`: the road-response subspace, `n_int x r`, orthonormal columns."""
    b = spec["basis"]
    with np.load(paths.resolve(b["path"]), allow_pickle=True) as z:
        U = np.ascontiguousarray(z[b["u"]], np.float64)
        r = int(np.asarray(z[b["rank"]]))
    if U.shape[1] != r:
        raise SystemExit(f"basis {spec['name']}: {U.shape} vs rank {r}")
    if U.shape[0] != ctx(spec["geometry"])["n_int"]:
        raise SystemExit(f"basis {spec['name']}: rows {U.shape[0]} vs n_int")
    return U, r


def clear_cache():
    _CTX.clear()
    _LU.clear()
