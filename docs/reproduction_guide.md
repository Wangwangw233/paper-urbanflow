# Reproduction guide

## Levels

| Level | What runs | Inputs |
|---|---|---|
| A. Frozen figures and tables | figure and table scripts | bundled `results/frozen/` and `figures/` |
| B. Core metric recomputation | `experiments/` diagnostics | external aligned fields, bases, freeze files and geometry |
| C. Raw end-to-end rebuild | not included | raw trajectories, registration, road download and training |

Level A is available from a clean checkout. Level B fails loudly when an
external input is missing; there is no workstation path fallback. Level C is
not claimed.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Level A: figures and tables

```bash
python scripts/reproduce_figures/reproduce_all.py
python scripts/reproduce_tables/reproduce_all.py
```

Outputs go to `figures/main/`, `figures/supplementary/` and
`results/derived/`. The scripts read frozen products; they do not fit a model.

## Level B: core recomputation

Set the external data root and run an experiment entry point:

```bash
export URBANFLOW_ROOT=/path/to/frozen_data
python -m urbanflow evaluate Porto --split frozen --fold 0
python experiments/constitutive_coupling/d_uncertainty.py
python experiments/frequency_response/phase_metrics.py
python experiments/anisotropy/anisotropy_null.py
```

The external root must contain the relative paths recorded in
`data_manifest/data.csv`; the core package refuses missing or mismatched
manifests.

## Audit

```bash
python scripts/audit/verify_public_repo.py --figure-smoke
```

The audit checks path/privacy rules, NPZ safety, CSV/JSON identifiers, the
file manifest, imports, compilation, unit tests and figure rebuilds.
