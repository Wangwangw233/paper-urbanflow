"""Regression tests.  `pytest urbanflow_ref/tests` on the data host.

Fast tests (manifest, code, splits, identity on three objects) run in about a
minute.  `slow` is everything the goldens need -- the full paper chain over
the nine objects.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

import urbanflow_ref as uf
from urbanflow_ref import manifest as MF
from urbanflow_ref import paths, registry, selfcheck

ROOT = paths.project_root()
needs_data = pytest.mark.skipif(
    not (ROOT / "project10" / "outputs" / "p0_l0").exists(),
    reason=f"data root {ROOT} not present")


@pytest.fixture(scope="session")
def ref():
    return uf.load()


# --------------------------------------------------------------------------
# the freeze itself
# --------------------------------------------------------------------------
@needs_data
def test_manifest_digest_is_pinned(ref):
    assert uf.MANIFEST_DIGEST not in ("", "UNBUILT")
    assert ref.digest == uf.MANIFEST_DIGEST
    ok, msg = MF.check_digest(ref.manifest, ref.digest)
    assert ok, msg


@needs_data
def test_code_is_the_code_that_was_hashed(ref):
    cur = MF._code_section()
    assert cur["files"] == ref.manifest["code"]["files"]


@needs_data
def test_manifest_pins_the_five_things(ref):
    for name in registry.ALL_OBJECTS:
        sec = ref.describe(name)
        c = sec["field"]["cache"]
        assert c["sha256"] and c["bytes"] > 0 and c["dt_min"] in (15, 30, 60)
        assert sec["road_basis"]["sha256"] and sec["road_basis"]["r"] > 0
        assert sec["data"]["freeze"]["sha256"]
        assert sec["splits"], f"{name} has no split"
        assert sec["operator"]["projector"].startswith("P_0")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
@needs_data
@pytest.mark.parametrize("name", registry.ALL_OBJECTS)
def test_split_blocks_are_disjoint_and_present(ref, name):
    spec = registry.spec(name)
    have = selfcheck._cache_dates(spec["cache"])
    for which in spec["splits"]:
        for fold in range(spec["folds"]):
            sp = ref.split(name, which, fold)
            tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
            assert tr and va and te
            assert not (tr & va) and not (tr & te) and not (va & te)
            assert (tr | va | te) <= have
            sec = ref.manifest["splits"][name][which][str(fold)]
            assert selfcheck._h(sp["train"]) == sec["train_sha256"]


@needs_data
def test_sf_has_five_folds(ref):
    assert len(ref.manifest["splits"]["SF"]["frozen"]) == 5
    for f in range(5):
        sp = ref.split("SF", "frozen", f)
        assert len(sp["test"]) == 3


# --------------------------------------------------------------------------
# the algebra
# --------------------------------------------------------------------------
@needs_data
@pytest.mark.parametrize("name,which,fold", [
    ("Porto", "frozen", 0),
    ("Chengdu", "window", 0),
    ("Shanghai", "refrozen", 0),
])
def test_identity(ref, name, which, fold):
    r = ref.evaluate(name, which, fold)
    assert r["identity"]["abs_err"] < 1e-12


@needs_data
def test_p_and_d_are_different_frames(ref):
    """The collapse this package exists to prevent: P centres only the fit
    rows, D centres everything.  On Porto they differ by ~0.21."""
    r = ref.evaluate("Porto", "frozen", 0)
    assert abs(r["frames"]["P"] - r["frames"]["D"]) > 0.2


# --------------------------------------------------------------------------
# the published numbers
# --------------------------------------------------------------------------
@needs_data
@pytest.mark.slow
def test_goldens(ref):
    g = paths.package_dir() / "goldens" / "urbanflow_reference_v1.json"
    assert g.exists()
    rep = selfcheck.run(ref, golden_path=g, legacy=False)
    assert rep["ok"], [c for c in rep["checks"] if not c["ok"]]


@needs_data
@pytest.mark.slow
def test_sf_five_fold_mean(ref):
    scores = [ref.evaluate("SF", "frozen", f)["frames"]["P"] for f in range(5)]
    assert abs(float(np.mean(scores)) - 0.602166) < 1e-5


@needs_data
@pytest.mark.slow
def test_legacy_libraries_agree(ref):
    name, ok, detail = selfcheck._legacy_check(ref)
    assert ok, detail
