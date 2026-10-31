# Environment

## Tested export environment

- Python: `3.12.10`
- NumPy: `2.4.4`
- SciPy: `1.17.1`
- Matplotlib: `3.10.9`
- OS family: `Windows`

These are the versions used for the release check, not a claim that every
scientific result is bitwise portable to every BLAS or platform. The reference
package pins its BLAS thread count to one. Numerical golden comparisons use the
tolerances in `src/urbanflow/goldens/urbanflow_reference_v1.json` and
`src/urbanflow/selfcheck.py`.

## Setup

Use a virtual environment and install `requirements.txt`; install the package
in editable mode for `python -m urbanflow`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Pytest is optional and is needed only for `tests/regression`. The unit and
public smoke suites use the standard-library `unittest` runner. Matplotlib must
be able to use its Agg backend. The audit sets `MPLCONFIGDIR` and
`PYTHONDONTWRITEBYTECODE` for subprocesses.

## External data

`URBANFLOW_ROOT` selects the frozen-data root. If it is unset, the reference
package defaults to `external_data/`; figure scripts default to the repository
root because their bundled frozen products are under `results/frozen/`. Core
commands fail on missing inputs; this is intentional. No raw-data download or
implicit fallback to a workstation path is provided.
