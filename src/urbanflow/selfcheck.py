"""Regression checks for UrbanFlow Reference v1.

`verify` answers one question -- "is this still the reference?" -- and it
answers it five ways, because each can fail alone:

    manifest   the file's own digest, and `_version.MANIFEST_DIGEST`
    code       every module hashed into the manifest is byte-identical
    splits     the split the code produces now has the hash the manifest
               recorded, blocks are disjoint, every date exists in the cache
    identity   R_I - R_CLIM = (1 - R_CLIM) R_D on every evaluated split
    goldens    the frozen numbers of the paper chain, to 1e-9
    legacy     (optional) the vendored operators agree with `audit_lib`,
               `r4lib`, `m7lib` element for element

The legacy check is what makes "vendored, not forked" a claim with evidence.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from . import manifest as MF
from . import paths, registry
from ._version import MANIFEST_DIGEST

#: what v1 promises to reproduce.  Porto and SF are the manuscript's own
#: objects; Rome is the external check; Chengdu is the 18-phase window; the
#: LaDe fleet is the second measurement family, both date protocols.
GOLDEN_SPECS = (
    [("Porto", "frozen", 0)]
    + [("SF", "frozen", f) for f in range(5)]
    + [("Rome", "frozen", 0), ("Chengdu", "window", 0)]
    + [(c, w, 0) for c in registry.LADE_CITIES
       for w in ("frozen", "refrozen")]
    + [(f"{c}@{dt}", w, 0) for c in registry.LADE_CITIES for dt in (15, 30)
       for w in ("frozen", "refrozen")]
)

GOLDEN_FIELDS = ("R_P", "R_D", "R_I", "R_CLIM", "dR_I", "kappa", "kappa_P",
                 "n_test", "identity_abs_err")


# --------------------------------------------------------------------------
def make_goldens(ref, tol=1e-9, specs=GOLDEN_SPECS, progress=False):
    out = dict(reference_id=ref.id, version=ref.version, digest=ref.digest,
               tolerance=tol, fields=list(GOLDEN_FIELDS), objects={})
    for i, (name, which, fold) in enumerate(specs, 1):
        if progress:
            print(f"  [{i}/{len(specs)}] {name}/{which}/{fold}", flush=True)
        out["objects"][f"{name}/{which}/{fold}"] = _rec(
            ref.evaluate(name, which, fold))
    return out


# --------------------------------------------------------------------------
# one evaluated split -> the golden record
# --------------------------------------------------------------------------
def _rec(r):
    return dict(R_P=r["frames"]["P"], R_D=r["frames"]["D"],
                R_I=r["frames"]["I"], R_CLIM=r["frames"]["CLIM"],
                dR_I=r["frames"]["dR_I"], kappa=r["estimates"]["kappa"],
                kappa_P=r["estimates"]["kappa_P"],
                n_test=r["counts"]["n_test"],
                identity_abs_err=r["identity"]["abs_err"])


def _diff(got, want, tol, prefix=""):
    out = []
    for k, v in want.items():
        g = got.get(k)
        if isinstance(v, dict):
            out += _diff(g or {}, v, tol, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if g is None or abs(float(g) - float(v)) > tol:
                out.append(f"{prefix}{k}: {g} != {v}")
        elif g != v:
            out.append(f"{prefix}{k}: {g} != {v}")
    return out


# --------------------------------------------------------------------------
def _cache_dates(rel):
    with np.load(paths.resolve(rel), allow_pickle=True) as z:
        wd = np.asarray(z["wdate"], np.int64)
        ws = np.asarray(z["wslot"], np.int64)
        wv = (np.asarray(z["wvalid"], bool) if "wvalid" in z.files
              else np.ones(wd.shape, bool))
    return set(int(x) for x in np.unique(wd[wv & (ws >= 0)]))


def run(ref, golden_path=None, only=None, tol=None, legacy=True):
    """Return `dict(ok, passed, failed, checks)`."""
    checks = []

    def add(name, ok, detail=""):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    # --- 1. the manifest's own digest ---------------------------------
    ok, msg = MF.check_digest(ref.manifest, ref.digest)
    add("manifest.digest", ok, msg)
    pinned = MANIFEST_DIGEST not in ("", "UNBUILT")
    add("manifest.pinned",
        (not pinned) or ref.digest == MANIFEST_DIGEST,
        f"file={ref.digest} _version={MANIFEST_DIGEST}")

    # --- 2. code ------------------------------------------------------
    cur = MF._code_section()
    want = ref.manifest["code"]
    drift = [f for f in want["files"]
             if cur["files"].get(f) != want["files"][f]]
    extra = [f for f in cur["files"] if f not in want["files"]]
    add("code.files", not drift and not extra,
        f"changed={drift} unlisted={extra}")

    # --- 3. splits ----------------------------------------------------
    split_ok, split_detail = True, []
    for name in registry.ALL_OBJECTS:
        try:
            sec = ref.manifest["splits"].get(name, {})
            if not sec:
                continue
            spec = registry.spec(name)
            seen = {}
            for which, fold, sp in _iter_splits(spec, sec):
                rec = sec[which][str(fold)]
                tr, va, te = (set(sp["train"]), set(sp["val"]),
                              set(sp["test"]))
                if _h(sp["train"]) != rec["train_sha256"]:
                    split_ok = False
                    split_detail.append(f"{name}/{which}/{fold} train hash")
                for key in ("val", "test"):
                    if _h(sp[key]) != rec[key]["sha256"]:
                        split_ok = False
                        split_detail.append(f"{name}/{which}/{fold} {key}")
                if tr & va or tr & te or va & te:
                    split_ok = False
                    split_detail.append(f"{name}/{which}/{fold} overlap")
                if not (tr and va and te):
                    split_ok = False
                    split_detail.append(f"{name}/{which}/{fold} empty block")
                seen[(which, fold)] = (tr, va, te)
        except Exception as e:                       # noqa: BLE001
            split_ok = False
            split_detail.append(f"{name}: {type(e).__name__}: {e}")
    add("splits.hash", split_ok, split_detail[:8])

    # every date of every split must exist in the cache
    miss = []
    for name in registry.ALL_OBJECTS:
        spec = registry.spec(name)
        try:
            have = _cache_dates(spec["cache"])
        except FileNotFoundError as e:
            miss.append(f"{name}: {e}")
            continue
        for which, fold, sp in _iter_splits(spec,
                                            ref.manifest["splits"].get(name, {})):
            gone = [d for d in sp["train"] + sp["val"] + sp["test"]
                    if int(d) not in have]
            if gone:
                miss.append(f"{name}/{which}/{fold}: {gone[:4]}")
    add("splits.present", not miss, miss[:8])

    # --- 4. identity + 5. goldens ------------------------------------
    gold = None
    if golden_path:
        gold = json.loads(pathlib.Path(golden_path).read_text(encoding="utf-8"))
        if tol is None:
            tol = gold.get("tolerance", 1e-9)
    tol = 1e-9 if tol is None else tol

    if gold:
        keys = list(gold["objects"])
    else:                               # no goldens: evaluate the spec list
        keys = [f"{n}/{w}/{f}" for n, w, f in GOLDEN_SPECS]
    if only:
        keys = [k for k in keys if only in k]

    ident_bad, gold_bad, n = [], [], 0
    for key in keys:
        name, which, fold = _parse(key)
        r = ref.evaluate(name, which, int(fold))
        n += 1
        if r["identity"]["abs_err"] > 1e-12:
            ident_bad.append(f"{key}: {r['identity']['abs_err']:.2e}")
        if gold:
            gold_bad += [f"{key} {d}" for d in
                         _diff(_rec(r), gold["objects"][key], tol)]
    add("identity", not ident_bad,
        ident_bad or f"{n} splits, max < 1e-12")
    add("goldens", not gold_bad,
        gold_bad[:8] if gold_bad else f"{n} splits within {tol:g}")

    # --- 6. legacy fidelity ------------------------------------------
    if legacy:
        try:
            add(*_legacy_check(ref))
        except Exception as e:                       # noqa: BLE001
            add("legacy", False, f"{type(e).__name__}: {e}")

    failed = [c for c in checks if not c["ok"]]
    return dict(ok=not failed, passed=len(checks) - len(failed),
                failed=len(failed), checks=checks)


# --------------------------------------------------------------------------
def _iter_splits(spec, sec):
    from . import splits as SP
    for which in sec:
        n = len(sec[which])
        for fold in range(n):
            yield which, fold, SP.resolve(spec, which, fold)


def _h(dates):
    import hashlib
    return hashlib.sha256(MF.canonical(sorted(int(d) for d in dates))
                          ).hexdigest()


def _parse(key):
    parts = key.split("/")
    return parts[0], parts[1], int(parts[2])


def _legacy_check(ref):
    """The private legacy libraries are intentionally not exported."""
    raise RuntimeError("legacy comparison libraries are not included; rerun with --no-legacy")
