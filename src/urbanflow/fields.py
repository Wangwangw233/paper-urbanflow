"""From an aligned cache to the design matrix of one split.

Rows are `(date, phase)` pairs, date-major, phase ascending.  A missing pair
contributes **no row** -- it is never a zero.  That is the difference between
"nothing moved" and "nothing was measured", and it is the reason Chengdu's
18-hour window is representable at all.

The phase is the scale's own clock phase: 24 per day at 60 min, 96 at 15 min.
Forcing a fine scale back onto 24 phases would handicap it by construction,
so `nslot` comes from the cache's `dt_min`, not from a constant.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G

_FLUX: dict = {}
#: a 15-min LaDe field is ~0.9 GB; keeping every object memoised would exhaust
#: memory long before the last golden is computed.  Two is enough for the
#: access pattern (folds of one object, then the next object).
_FLUX_MAX = 2


def flux(spec):
    """Memoised `(F, meta)` for an object, least-recently-used evicted."""
    name = spec["name"]
    if name not in _FLUX:
        while len(_FLUX) >= _FLUX_MAX:
            _FLUX.pop(next(iter(_FLUX)))
        F, meta = G.load_flux(spec["cache"], spec["geometry"])
        _FLUX[name] = (np.ascontiguousarray(F, np.float64), meta)
    return _FLUX[name]


def clear_cache():
    _FLUX.clear()


def stack_rows(F, meta, dates, hours=None):
    """Rows of `F` for the present `(date, phase)` pairs.

    Returns `(Frows, ph, di)` -- phase within the day (0 .. nslot-1) and date
    index into `dates`.  `hours` is expressed in clock hours and expands to
    the `nslot/24` phases that fall inside each hour.
    """
    wd = np.asarray(meta["wdate"], np.int64)
    ws = np.asarray(meta["wslot"], np.int64)
    nslot = 24 * 60 // int(meta["dt_min"])
    pos = {int(d): i for i, d in enumerate(dates)}
    sel = np.isin(wd, np.array(sorted(pos), np.int64))
    if "wvalid" in meta:
        sel &= np.asarray(meta["wvalid"], bool)
    sel &= (ws >= 0) & (ws < nslot)
    if hours is not None:
        per = nslot // 24
        keep_h = np.zeros(nslot, bool)
        for h in hours:
            keep_h[int(h) * per:(int(h) + 1) * per] = True
        sel &= keep_h[np.clip(ws, 0, nslot - 1)]
    idx = np.flatnonzero(sel)
    key = np.array([pos[int(wd[i])] * nslot + int(ws[i]) for i in idx],
                   np.int64)
    o = np.argsort(key, kind="stable")
    idx, key = idx[o], key[o]
    keep = np.ones(key.size, bool)
    keep[1:] = key[1:] != key[:-1]
    idx, key = idx[keep], key[keep]
    return F[idx], (key % nslot).astype(np.int64), (key // nslot).astype(np.int64)


def design(spec, dates, hours=None, keep_field=True):
    """`(X, Y, ph, di, ...)` -- the source design and the routing target.

        x_t = D F_t                       (NC faces -> cell divergence)
        F°_t = P0 F_t                     (divergence-free part)
        y_t = U^T F°_t                    (r road-response coordinates)

    `keep_field=False` drops `F` and `F°` after the target is built.  Scoring
    needs only `X` and `Y`, and at dt=15 the retained field alone is ~1.8 GB.
    """
    F, meta = flux(spec)
    Fr, ph, di = stack_rows(F, meta, dates, hours)
    key = spec["geometry"]
    X = np.ascontiguousarray((G.D(key) @ Fr.T).T, np.float64)
    Fc = G.P0(key, Fr)
    U, r = G.load_basis(spec)
    Y = np.ascontiguousarray(Fc @ U, np.float64)
    c = G.ctx(key)
    if not keep_field:
        Fr = Fc = None
    return dict(X=X, Y=Y, ph=ph, di=di, F=Fr, Fc=Fc, meta=meta, U=U, r=r,
                NC=c["NC"], n_int=c["n_int"], order=[int(d) for d in dates])
