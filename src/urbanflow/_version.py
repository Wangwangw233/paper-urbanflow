"""Immutable identifiers for the UrbanFlow reference implementation.

`REFERENCE_ID` is what an experiment cites.  `MANIFEST_DIGEST` is the
sha256 of the canonical manifest payload shipped next to this file
(`manifest.json`); `urbanflow_ref.load()` refuses to run if the payload does
not hash to it.  Two numbers together identify the *whole* computational
object: field, data, road basis, operators, metrics and splits.

`_version.py` is deliberately excluded from the code hashes inside the
manifest -- it carries the digest, so hashing it would be circular.  Every
other module is hashed.
"""
from __future__ import annotations

REFERENCE_ID = "urbanflow-reference-v1"
REFERENCE_VERSION = "1.0.0"
MANIFEST_SCHEMA = 1

# Filled by `python -m urbanflow_ref build-manifest`.  Until then the package
# still imports (so `describe`/`--help` work) but `load()` refuses.
MANIFEST_DIGEST = "b7b85d7864eaf93c7916b1f4dfd24e9d23a4c7c807457c5767558905a9860566"
