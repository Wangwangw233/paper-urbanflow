# Architecture

## Scientific object

For each date and clock phase, the aligned cache stores a signed face flux
`F_t`. The discrete incidence matrix `D` maps faces to cells, so
`x_t = D F_t` is the source-sink balance. `P0 = I - D^+ D` projects onto the
divergence-free subspace. The road-supported response is fitted in road
coordinates `y_t = U_R^T P0 F_t`.

## Package layout

`src/urbanflow/` is a flat numerical package because the modules share one
reference manifest and one set of conventions:

| Module | Responsibility |
|---|---|
| `fields.py` | cache loading and `(date, phase)` design rows |
| `geometry.py` | grid context, `D`, `D^+`, `P0`, road-basis loading |
| `frames.py` | P/D/I/CLIM scoring frames and ridge paths |
| `metrics.py` | error, ridge, bootstrap and identity definitions |
| `registry.py`, `splits.py` | object registry and frozen split resolution |
| `manifest.py`, `paths.py`, `_version.py` | content-addressed reference manifest |
| `reference.py`, `cli.py` | public evaluation entry point and command line |
| `selfcheck.py` | golden and legacy checks |

Keeping these modules flat avoids duplicating definitions across domain
folders. The experiment directories name the scientific domains; the package
remains the single numerical implementation.

## Data and result flow

1. External aligned caches and road bases are pinned by
   `src/urbanflow/manifest.json`.
2. `experiments/` scripts recompute diagnostics or uncertainty from those
   inputs and write JSON to `results/frozen/`.
3. `scripts/reproduce_figures/` reads frozen products and writes
   `figures/main/` and `figures/supplementary/`.
4. `scripts/reproduce_tables/` reads frozen cross-city scores and writes a
   machine-readable table product.
5. `scripts/audit/` validates paths, manifests, hashes, imports and figure
   rebuilds.

The paper sources remain under `paper/`; they are not imported by the
numerical package.
