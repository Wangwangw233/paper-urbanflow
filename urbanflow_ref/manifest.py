"""The manifest: what `urbanflow-reference-v1` is, as bytes.

The manifest records **identifiers only** -- file digests, ranks, counts,
date-protocol hashes, code hashes, operator text.  It deliberately contains no
measured quantity: a float that came out of BLAS would make the digest depend
on the machine, and a digest that changes when nothing changed is not a
version.

    digest = sha256(canonical(payload - {created_utc, env}))

so rebuilding the same code over the same data yields the same digest, and
changing one byte of one cache changes it.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys

import numpy as np
import scipy

from . import paths, registry, splits as SP
from ._version import (MANIFEST_SCHEMA, REFERENCE_ID, REFERENCE_VERSION)

#: keys that must not enter the digest
VOLATILE = ("created_utc", "env", "digest")

#: `_version.py` carries the digest, so it cannot be hashed into it
CODE_EXCLUDE = {"_version.py"}


# --------------------------------------------------------------------------
# canonical form
# --------------------------------------------------------------------------
def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest_of(payload) -> str:
    body = {k: v for k, v in payload.items() if k not in VOLATILE}
    return hashlib.sha256(canonical(body)).hexdigest()


def check_digest(payload, digest):
    got = digest_of(payload)
    return got == digest, f"manifest={digest} recomputed={got}"


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _h_dates(dates) -> str:
    return hashlib.sha256(canonical(sorted(int(d) for d in dates))).hexdigest()


def _file(path) -> dict:
    p = paths.resolve(path)
    return dict(path=str(path), bytes=int(p.stat().st_size),
                sha256=sha256_file(p))


# --------------------------------------------------------------------------
# the five pinned sections
# --------------------------------------------------------------------------
def _cache_info(rel) -> dict:
    p = paths.resolve(rel)
    with np.load(p, allow_pickle=True) as z:
        K, ny, nx = z["F_E"].shape
        wd = np.asarray(z["wdate"], np.int64)
        ws = np.asarray(z["wslot"], np.int64)
        wv = z["wvalid"] if "wvalid" in z.files else np.ones(K, bool)
        dt = int(z["dt_min"])
        scale = float(z["scale_m"])
        l0 = float(z["l0_m"])
        has_udates = "udates" in z.files
    dates = np.unique(wd[wv & (ws >= 0)])
    return dict(path=str(rel), bytes=int(p.stat().st_size),
                sha256=sha256_file(p),
                n_windows=int(K), grid=[int(ny), int(nx)],
                dt_min=dt, phases_per_day=int(24 * 60 // dt),
                cell_m=scale, gauge_m=l0,
                n_dates=int(dates.size),
                date_first=int(dates.min()) if dates.size else None,
                date_last=int(dates.max()) if dates.size else None,
                udates_index=bool(has_udates))


def _basis_info(b) -> dict:
    p = paths.resolve(b["path"])
    with np.load(p, allow_pickle=True) as z:
        U = np.ascontiguousarray(z[b["u"]], np.float64)
        r = int(np.asarray(z[b["rank"]]))
    n_int, k = U.shape
    if k != r:
        raise SystemExit(f"basis {b['path']}: {U.shape} vs rank {r}")
    return dict(path=str(b["path"]), bytes=int(p.stat().st_size),
                sha256=sha256_file(p), n_int=int(n_int), r=r,
                columns="orthonormal")


def _freeze_info(rel) -> dict:
    d = _file(rel)
    doc = json.loads(paths.resolve(rel).read_text(encoding="utf-8"))
    sp = doc.get("split", {})
    d["has_folds"] = "folds" in sp
    d["n_folds"] = len(sp.get("folds", [])) or 1
    d["n_eligible"] = len(doc.get("eligible_dates", []))
    return d


def _split_section(spec):
    out = {}
    for which, fold, sp in SP.all_splits(spec):
        rec = dict(fold=int(fold),
                   n_train=len(sp["train"]), n_val=len(sp["val"]),
                   n_test=len(sp["test"]),
                   n_eligible=len(sp.get("eligible", [])),
                   train_sha256=_h_dates(sp["train"]),
                   val=_hash_or_list(sp["val"]),
                   test=_hash_or_list(sp["test"]))
        if "hours" in sp:
            rec["hours"] = list(sp["hours"])
        out.setdefault(which, {})[str(fold)] = rec
    return out


def _hash_or_list(dates):
    """Small blocks are listed verbatim; long TRAIN lists only as a hash."""
    d = [int(x) for x in dates]
    if len(d) <= 64:
        return dict(values=sorted(d), sha256=_h_dates(d))
    return dict(n=len(d), sha256=_h_dates(d))


def _code_section() -> dict:
    d = paths.package_dir()
    files = {}
    for p in sorted(d.glob("*.py")):
        if p.name in CODE_EXCLUDE:
            continue
        files[p.name] = sha256_file(p)
    return dict(files=files, excluded=sorted(CODE_EXCLUDE),
                digest=hashlib.sha256(canonical(files)).hexdigest())


def _env_section() -> dict:
    return dict(python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, platform=platform.platform(),
                blas_threads=1)


OPERATORS = {
    "field": "F_t: observed face flux, one scalar per interior edge; "
             "missing (date, phase) pairs contribute no row, never a zero",
    "source": "x_t = D F_t, D the NC x n_int face-to-cell incidence matrix "
              "(+1 head cell, -1 tail cell), row-major east/north face order",
    "projector": "P_0 = I - D^+ D, gauge cell 0 pinned; D^+ from a sparse LU "
                 "of D D^T with row/column 0 removed",
    "target": "y_t = U^T (P_0 F_t), U the r-column orthonormal road-response "
              "basis of the object",
    "estimator": "ridge in dual form: a = V ((V^T T) / (s + lam)), "
                 "pred = (X_pred X_fit^T) a, s eig(X_fit X_fit^T)",
    "regularisation": "relative: lam = kappa * s_max(X_fit X_fit^T); kappa "
                      "chosen on VAL only, grid fixed in metrics.KAPPA",
    "complexity": "one eigendecomposition per split; no kernel, no "
                  "nonlinearity, no per-object tuning beyond kappa",
}

METRICS = {
    "E": "E(v, v_hat) = sum ||v - v_hat||^2 / sum ||v||^2 over rows of the "
         "evaluation block",
    "score": "1 - E, reported as R",
    "frames": {
        "P": "centre FIT rows only; predict from raw TEST x; score raw TEST y; "
             "denominator ||y||^2  (the manuscript's 1 - E)",
        "D": "centre every row with TRAIN phase means; score the anomaly z; "
             "denominator ||z||^2",
        "I": "D's prediction plus TRAIN phase mean; denominator ||y||^2",
        "CLIM": "TRAIN phase mean alone; denominator ||y||^2",
    },
    "identity": "R_I - R_CLIM = (1 - R_CLIM) R_D holds by construction for "
                "D/I/CLIM; checked every run against 1e-12",
    "bootstrap": "unit = independent TEST date; percentile 2.5/97.5; "
                 "n=10000, seed=20261028",
}


def build(objects=registry.ALL_OBJECTS, with_splits=True) -> dict:
    data = {}
    bases = {}
    sp = {}
    for name in objects:
        s = registry.spec(name)
        data[name] = dict(family=s["family"], geometry=s["geometry"],
                          phases=s["phases"], hours=s["hours"],
                          splits=list(s["splits"]), folds=s["folds"],
                          cache=_cache_info(s["cache"]),
                          freeze=_freeze_info(s["freeze"]))
        bases[name] = _basis_info(s["basis"])
        if with_splits:
            sp[name] = _split_section(s)

    geom = {}
    for name in objects:
        s = registry.spec(name)
        if s["geometry"] in geom:
            continue
        from . import geometry as G
        c = G.ctx(s["geometry"])
        geom[s["geometry"]] = dict(ny=c["ny"], nx=c["nx"], NC=c["NC"],
                                   n_int=c["n_int"])

    payload = {
        "schema": MANIFEST_SCHEMA,
        "reference_id": REFERENCE_ID,
        "version": REFERENCE_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "field": dict(
            definition=OPERATORS["field"],
            row="(date, clock phase), date-major, phase ascending",
            phase="the scale's own clock phase: 24 per day at dt=60 min, "
                  "96 at dt=15 min",
            missing="absent pair -> no row; N == 0 is not imputed"),
        "geometry": dict(
            source=_file("routing_urban_medium_constitutive/results/um_ctx.npz"),
            face_order="row-major, east faces first, then north; rebuilt from "
                       "the grid mask and asserted equal to um_ctx p0/p1 on "
                       "every load",
            grids=geom),
        "data": dict(objects=data),
        "road_basis": bases,
        "operators": OPERATORS,
        "metrics": METRICS,
        "splits": sp,
        "code": _code_section(),
        "env": _env_section(),
    }
    return payload


def save(payload, path):
    doc = dict(digest=digest_of(payload), payload=payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return doc["digest"]
