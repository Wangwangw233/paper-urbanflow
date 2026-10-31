"""The single entry point: `urbanflow_ref.load()` -> `Reference`.

A `Reference` is a manifest plus the code that produced it.  It can answer,
for any object it knows:

    ref.split("Porto")             which dates are TRAIN / VAL / TEST
    ref.describe("Porto")          field, data, basis, operator, metric, split
    ref.evaluate("Porto")          P / D / I / CLIM + identity + per-date

Every evaluation carries the reference id and digest in its output, so a
number quoted in a paper or a report can be traced back to exactly one
computational object.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from . import fields, frames, manifest as MF, metrics as M, paths, registry, splits


class Reference:
    """A loaded, digest-checked UrbanFlow reference."""

    def __init__(self, payload, digest):
        self.manifest = payload
        self.digest = digest
        self.id = payload["reference_id"]
        self.version = payload["version"]
        self._designs: dict = {}

    # -- construction ---------------------------------------------------
    @classmethod
    def load(cls, path=None, verify=True, allow_unpinned=False):
        from ._version import MANIFEST_DIGEST

        p = pathlib.Path(path) if path else paths.package_dir() / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run `python -m urbanflow_ref build-manifest`")
        doc = json.loads(p.read_text(encoding="utf-8"))
        payload = doc["payload"] if "payload" in doc else doc
        digest = doc["digest"] if "digest" in doc else MF.digest_of(payload)
        if verify:
            ok, msg = MF.check_digest(payload, digest)
            if not ok:
                raise RuntimeError(f"reference manifest failed verification: {msg}")
            pinned = MANIFEST_DIGEST not in ("", "UNBUILT")
            if pinned and not allow_unpinned and digest != MANIFEST_DIGEST:
                raise RuntimeError(
                    f"manifest digest {digest[:16]}... does not match the "
                    f"pinned {MANIFEST_DIGEST[:16]}...: this is not the "
                    f"reference it claims to be")
        return cls(payload, digest)

    # -- introspection --------------------------------------------------
    def object(self, name):
        registry.spec(name)                       # validates the name
        return self.manifest["data"]["objects"][name]

    def split(self, name, which="frozen", fold=0):
        spec = registry.spec(name)
        return splits.resolve(spec, which, fold)

    def describe(self, name):
        spec = registry.spec(name)
        data = self.object(name)
        basis = self.manifest["road_basis"][name]
        sp = splits.resolve(spec, spec["splits"][0], 0)
        return dict(
            reference=dict(id=self.id, version=self.version, digest=self.digest),
            object=dict(name=name, family=spec["family"],
                        geometry=spec["geometry"], phases=spec["phases"],
                        hours=spec["hours"]),
            field=dict(cache=data["cache"], dt_min=data["cache"]["dt_min"],
                       n_windows=data["cache"]["n_windows"]),
            data=dict(freeze=data["freeze"]),
            road_basis=basis,
            operator=self.manifest["operators"],
            metric=self.manifest["metrics"],
            splits={w: {str(f): s for f, s in _split_map(spec, w)}
                    for w in spec["splits"]},
            first_split=sp,
        )

    # -- the computation ------------------------------------------------
    def design(self, name, which="frozen", fold=0, keep_field=False):
        key = (name, which, int(fold))
        if key not in self._designs:
            while len(self._designs) >= 2:      # bounded: one design can be GBs
                self._designs.pop(next(iter(self._designs)))
            spec = registry.spec(name)
            sp = splits.resolve(spec, which, fold)
            order = list(dict.fromkeys(sp["train"] + sp["val"] + sp["test"]))
            D_ = fields.design(spec, order, hours=sp.get("hours"),
                               keep_field=keep_field)
            self._designs[key] = (D_, sp, order)
        return self._designs[key]

    def evaluate(self, name, which="frozen", fold=0, kappa_grid=M.KAPPA):
        spec = registry.spec(name)
        D_, sp, order = self.design(name, which, fold)
        m_tr = _mask(D_, order, sp["train"])
        m_va = _mask(D_, order, sp["val"])
        m_te = _mask(D_, order, sp["test"])
        out = frames.run_frames(D_, m_tr, m_va, m_te, kappa_grid)
        data = self.object(name)
        return dict(
            reference=dict(id=self.id, version=self.version,
                           digest=self.digest),
            object=dict(name=name, family=spec["family"],
                        geometry=spec["geometry"], phases=spec["phases"],
                        cache=data["cache"],
                        basis=self.manifest["road_basis"][name],
                        r=int(D_["r"]), n_int=int(D_["n_int"])),
            split=dict(which=which, fold=int(fold),
                       train=list(sp["train"]), val=list(sp["val"]),
                       test=list(sp["test"])),
            frames=dict(P=out["R_P"], D=out["R_D"], I=out["R_I"],
                        CLIM=out["R_CLIM"], dR_I=out["dR_I"]),
            estimates=dict(kappa=out["kappa"], kappa_P=out["kappa_P"],
                           df_eff=out["df_eff"], smax=out["smax"]),
            identity=out["identity"],
            counts=dict(n_train=out["n_train"], n_val=out["n_val"],
                        n_test=out["n_test"], n_dates=len(sp["test"]),
                        dim_out=out["dim_out"]),
            per_date=out["per_date"],
            bootstrap=dict(dR_I=out["boot_dR_I"], R_D=out["boot_R_D"],
                           R_I=out["boot_R_I"]),
            detail=out,
        )


# --------------------------------------------------------------------------
def _mask(D_, order, dates):
    pos = {int(d): i for i, d in enumerate(order)}
    idx = [pos[int(d)] for d in dates if int(d) in pos]
    if len(idx) != len(dates):
        missing = [int(d) for d in dates if int(d) not in pos]
        raise KeyError(f"{len(missing)} dates absent from the row pool: "
                       f"{missing[:5]}")
    return np.isin(D_["di"], idx)


def _split_map(spec, which):
    if which == "frozen" and spec["folds"] > 1:
        return [(f, splits.resolve(spec, which, f)) for f in range(spec["folds"])]
    return [(0, splits.resolve(spec, which, 0))]


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
