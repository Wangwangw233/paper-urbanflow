# Environment

## Tested export environment

- Python: `3.12.10`
- NumPy: `2.4.4`
- SciPy: `1.17.1`
- Matplotlib: `3.10.9`
- OS family: `Windows`

These are the versions used by the exporter, not a claim that every scientific
result is bitwise portable to every BLAS or platform. The reference package sets
its BLAS thread count to one. Numerical golden comparisons use the tolerances in
`urbanflow_ref/goldens/urbanflow_reference_v1.json` and `urbanflow_ref/selfcheck.py`.

## Setup

Use a virtual environment and install `requirements.txt`. Pytest is optional and
is needed only for the original `urbanflow_ref/tests` suite; the public smoke
suite uses the standard-library `unittest` runner. Matplotlib must be able to use
its Agg backend. The verifier sets `MPLCONFIGDIR` and `PYTHONDONTWRITEBYTECODE`
for subprocesses.

## External data

`URBANFLOW_ROOT` selects the frozen-data root. It defaults to the release root
for figure redraw and to `external_data/` for `urbanflow_ref`. Core commands
fail on missing inputs; this is intentional. No raw-data download or implicit
fallback to a workstation path is provided.
