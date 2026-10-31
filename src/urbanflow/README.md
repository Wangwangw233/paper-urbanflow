# urbanflow package

The package is flat because every module shares one content-addressed
manifest and one set of numerical conventions. The domain mapping is:

| Domain | Module |
|---|---|
| aligned data and design rows | `fields.py` |
| incidence, divergence and Hodge projector | `geometry.py` |
| road basis and geometry context | `geometry.py` |
| constitutive scoring frames | `frames.py`, `reference.py` |
| metrics, ridge and bootstrap | `metrics.py` |
| frozen split resolution | `splits.py`, `registry.py` |
| manifest and path resolution | `manifest.py`, `paths.py` |
| evaluation CLI | `cli.py`, `__main__.py` |
| numerical self-checks | `selfcheck.py` |

The import name is `urbanflow`; historical strings inside vendored module
docstrings are retained because the reference manifest hashes the module
bytes. `manifest.json` and `goldens/` are package data.
