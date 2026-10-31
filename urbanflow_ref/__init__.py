"""UrbanFlow Reference v1 -- the single computation behind every published
number of the paper chain.

    import urbanflow_ref as uf
    ref = uf.load()                      # verifies the manifest digest
    ref.describe("Porto")                # field, data, basis, operator, metric, split
    r = ref.evaluate("Porto")            # P / D / I / CLIM + identity + per date

A cited number carries `ref.id` (`urbanflow-reference-v1`) and `ref.digest`
(sha256 of the manifest).  The manifest pins the field cache, the date
protocol, the road basis, the grid, the operators, the metric definitions and
the κ grid by content hash, so the citation resolves to exactly one
computation.

BLAS is pinned to one thread before numpy is imported: a multi-threaded
reduction changes the last bits of a score, and a reference that depends on
the thread count is not frozen.  If numpy is already imported this cannot be
honoured, and `manifest["env"]` records the thread setting that was used.
"""
from __future__ import annotations

import os
import sys

if "numpy" not in sys.modules:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

from ._version import (REFERENCE_ID, REFERENCE_VERSION,  # noqa: E402
                       MANIFEST_DIGEST)
from .reference import Reference                       # noqa: E402
from . import (fields, frames, geometry, manifest,     # noqa: E402
               metrics, registry, splits)


def load(path=None, verify=True):
    """Load and digest-check the reference."""
    return Reference.load(path, verify)


def evaluate(name, **kw):
    """`uf.evaluate("Porto")` -- convenience over `load().evaluate`."""
    return load().evaluate(name, **kw)


def describe(name, **kw):
    return load().describe(name)


__all__ = ["REFERENCE_ID", "REFERENCE_VERSION", "MANIFEST_DIGEST", "Reference",
           "load", "evaluate", "describe", "fields", "frames", "geometry",
           "manifest", "metrics", "registry", "splits"]
