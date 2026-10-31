# UrbanFlow

Reference code and frozen products for a study of **road-supported effective
constitutive coupling between source-sink patterns and anisotropic urban
routing**. The repository separates the numerical reference implementation,
experiment entry points, frozen results and figure/table reproduction.

Repository: https://github.com/Wangwangw233/paper-urbanflow

## What the code computes

For a city lattice, the aligned cache stores signed face fluxes `F_t`. The
incidence matrix `D` maps faces to cells, so `x_t = D F_t` is the source-sink
balance and `P0 = I - D^+ D` projects onto the divergence-free subspace. The
road-supported response is estimated in road coordinates
`y_t = U_R^T P0 F_t`, with a ridge operator selected on VAL and evaluated on
held-out dates. The four scoring frames are P (raw), D (departure), I
(identity-recombined) and CLIM (source-free phase mean). The code also
implements phase-sensitive frequency metrics, local anisotropy tensors and
matched-capacity diagnostics.

The estimated relation is partial and city-dependent; the repository does not
encode a claim that road geometry alone causes the observed coupling. See
`docs/known_limitations.md`.

## Cities

| City | Source | Grid | Time window |
|---|---|---|---|
| Porto | ECML/PKDD taxi trajectories | 32 x 32, 250 m | 24 h |
| San Francisco | Cabspotting | 32 x 88, 250 m | 24 h |
| Chengdu | DiDi ride-hailing sample | 60 x 60, 250 m | 06-23 h observed block |

Exact release identifiers were not archived and are marked `UNKNOWN` in
`data_manifest/datasets.md`. No individual-level trajectories are included.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Python 3.10+ is recommended. See `docs/ENVIRONMENT.md` for the tested export
environment.

## Repository layout

```text
configs/                 city, split, experiment and baseline definitions
data_manifest/           dataset notes, dictionary, checksums and manifests
src/urbanflow/           numerical reference package
experiments/             domain experiment entry points
scripts/reproduce_*      figure and table reproduction
scripts/audit/           release audit and cross-reference tools
results/frozen/          final JSON/CSV/NPZ products, grouped by result family
figures/                 final main and supplementary vector PDFs
tests/                   unit, integration and regression tests
docs/                    architecture, protocol, provenance and limitations
paper/                   manuscript sources and bibliography
archive/legacy/          historical scripts retained for provenance
```

`src/urbanflow/` is intentionally a single flat package: its modules share one
content-addressed manifest and one set of numerical conventions. Domain
ownership is documented in `docs/architecture.md`; experiment directories name
the scientific questions without duplicating numerical code.

## Reproduction levels

| Level | Available | Command / requirement |
|---|---|---|
| A. Figures and tables from frozen products | yes | `python scripts/reproduce_figures/reproduce_all.py` and `python scripts/reproduce_tables/reproduce_all.py` |
| B. Core metrics from aligned fields | yes, with external inputs | `URBANFLOW_ROOT=/path/to/frozen_data` and an entry point under `experiments/` |
| C. Raw-trajectory end-to-end rebuild | no | raw data, registration and model training are not bundled |

Level A is data-free beyond the bundled frozen products. Level B fails if an
external cache, road basis, freeze file or geometry file is missing; there is
no workstation path fallback. Level C is not claimed.

### Figures

```bash
python scripts/reproduce_figures/reproduce_all.py
```

Outputs: `figures/main/` and `figures/supplementary/`. The four generators are
`redraw_urbanflow.py`, `fig_phase_response.py`, `fig5_redesign.py` and
`make_figures.py`.

### Tables

```bash
python scripts/reproduce_tables/reproduce_all.py
```

Outputs: `results/derived/table1_inference_scores.csv` and
`results/derived/table_d_uncertainty.csv`. Graph WaveNet cells are not
regenerated because their per-date errors are not bundled; the manuscript
records them as archived values.

### Core recomputation

```bash
export URBANFLOW_ROOT=/path/to/frozen_data
python -m urbanflow evaluate Porto --split frozen --fold 0
python experiments/constitutive_coupling/d_uncertainty.py
python experiments/frequency_response/phase_metrics.py
python experiments/anisotropy/anisotropy_null.py
```

The package verifies the code and data manifest before evaluating. The
reference digest is
`b7b85d7864eaf93c7916b1f4dfd24e9d23a4c7c807457c5767558905a9860566`.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/audit/verify_public_repo.py --figure-smoke
```

The unit tests cover `D F = x`, `D U_R = 0`, `U_R^T U_R = I`, Hodge
reconstruction and the synthetic frame identity. The audit checks privacy,
NPZ safety, file hashes, imports, compilation and all figure entry points.

## Data, privacy and license

- No raw GPS coordinates, trip identifiers or vehicle identifiers are present.
- NPZ files are numeric/Unicode arrays and are checked without pickle.
- Road and basemap geometry derive from OpenStreetMap; retain the
  OpenStreetMap contributors attribution and review the source terms.
- `LICENSE` is a placeholder. The authors must choose a license and finalize
  citation metadata before redistribution.
- `data_manifest/checksums.sha256` records the frozen products; the
  machine-readable file and data manifests are under `data_manifest/`.
