# Experiment configurations

The YAML files document the final analysis parameters and paths. They mirror
the values recorded in the archived experiment outputs.

`chengdu_nonlinear.json` is retained in its original JSON form because the
archived validation record pins its SHA-256. Its internal reference strings
name the historical external analysis tree (`urbanflow_ref/manifest.json`); the
current package equivalent is `src/urbanflow/manifest.json`. Editing this file
would break the archived config hash, so the historical strings are preserved
and the current mapping is documented in `docs/result_provenance.md`.
