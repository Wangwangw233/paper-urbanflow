# UrbanFlow public-release candidate

Code and small frozen products for **Road-supported effective constitutive coupling between source--sink patterns and anisotropic urban routing**.

Repository: https://github.com/Wangwangw233/paper-urbanflow

> **Candidate, not yet an open-source release.** The authors must still choose a license, complete the citation metadata, and supply the repository URL. The current paper and figures are synchronized and verified. `LICENSE` intentionally grants no new rights. No DOI, author list, or repository URL is invented here.

## Three reproduction levels

| level | what is included | inputs and limits |
|---|---|---|
| **1. Frozen figure redraw** | Plotting code, 11 reference PDFs, and small aggregate frozen arrays | Runs without raw trajectories or model fitting. `frozen/source_spectra.npz` and `frozen/reconstruction.npz` contain only arrays needed for display; they are not substitutes for the large scientific inputs from which they were derived. PDF metadata and rasterization can differ by environment, so byte-identical PDFs are not claimed. |
| **2. Core recomputation** | `urbanflow_ref`: aligned-field loading, geometry, road projection, ridge evaluation, frozen splits and numeric goldens | **Requires external frozen caches, bases, geometry and split files.** They are not bundled. `manifest/data.csv` lists their relative paths, sizes and pinned hashes. A full regression also requires every object in the manifest. |
| **3. Raw end-to-end rebuild** | Not included in this candidate | Raw trajectory ingestion, registration, road downloading, field construction and model training still require a separate privacy/license review and their original large inputs. No command in this repository claims to perform this level. |

## Environment and smoke tests

Python 3.10+ is recommended; see `docs/ENVIRONMENT.md` for the exact environment used for this export. Install the small plotting/compute stack:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/verify_public_repo.py --figure-smoke
```

The verifier checks the allowlist, file hashes and sizes, text/path/privacy patterns, NPZ safety, local imports, AST parsing and `compileall`. `--figure-smoke` also redraws all 11 figures into a temporary directory and checks that the candidate tree remains unchanged. Passing these tests does **not** mean the external-data core regression or raw end-to-end pipeline has run.

## Frozen figure redraw

From the repository root:

```bash
python paper/merged_manuscript/figures/redraw_urbanflow.py
python reports/review_response/scripts/fig_phase_response.py
python reports/threecity_figure5_revision/scripts/fig5_redesign.py B
python reports/threecity_supplement/scripts/make_figures.py
```

The scripts write PDFs and PNGs beside the reference PDFs. To avoid overwriting them, use each script's output option where provided, or set `URBANFLOW_FIGURE_OUT`, `URBANFLOW_SUPP_FIGURE_OUT`, and `URBANFLOW_RESULT_OUT` to separate directories. `URBANFLOW_ROOT` defaults to this repository and can point to another release-shaped frozen-data root. The joint-spectrum script additionally accepts `--data` and `--out`.

`reorganize_threecity_figures.py` is restricted to plotting in this public copy; its manuscript-editing section is deliberately removed.

The matched-capacity construction modules `benchmark/scripts/domain_lib.py` and `domain_rankmatched.py` are included as archival code. They import the private `bench_lib`/`de_lib` chain and therefore are not standalone runnable without the external analysis tree; the frozen sweep outputs and their validation are included separately.

## Core recomputation with external data

Arrange the pinned external inputs under one root using the relative paths in `manifest/data.csv`, then run, for example:

```bash
export URBANFLOW_ROOT=/path/to/frozen_data       # use set on Windows cmd
python -m urbanflow_ref describe Porto
python -m urbanflow_ref evaluate Porto --split frozen --fold 0
python -m urbanflow_ref verify --no-legacy       # needs ALL external objects
```

Do not expect these commands to work with only the bundled figure products. The public reference digest after the path-only packaging patch is `b7b85d7864eaf93c7916b1f4dfd24e9d23a4c7c807457c5767558905a9860566`; the upstream digest and patch provenance are in `manifest/provenance.json`. Numeric goldens are unchanged. `--no-legacy` is required because private legacy comparison libraries are not exported.

## Data, privacy, and attribution

- No individual-level trajectory data are included. NPZ files are numeric/Unicode archives and are checked without pickle.
- Large aligned matrices are not bundled. The two compact display products record their source hashes in `manifest/provenance.json`.
- Basemap arrays are derived from OpenStreetMap road/water geometry; acknowledge **OpenStreetMap contributors** and review the applicable source-data terms before publication.
- `manifest/files.csv` lists every release file other than itself, with SHA-256. `manifest/data.csv` distinguishes bundled products from required external inputs. `manifest/figures.csv` maps each paper figure to its generator and reference PDF.
- PDF and NPZ files are deliberately **not** globally ignored. Generated rasters, caches, build products, and `external_data/` are ignored instead.

## Cross-document references

`main_refs.tex` and `supplement_refs.tex` provide the supplementary/main S-number macros without `xr` or any dependency on pre-existing `.aux` files. If labels change, recompile once to regenerate the aux files, run `python tools/sync_crossrefs.py`, then compile again. Fresh Overleaf-style compilation without prior aux files is supported.

## Repeating the export

The release builder is retained in `tools/export_public_repo.py`. From the original private analysis tree it rebuilds this directory from the explicit allowlist:

```bash
python tools/export_public_repo.py --source-root /path/to/private/analysis --output /path/to/release/urbanflow
python tools/verify_public_repo.py --root /path/to/release/urbanflow --figure-smoke
```

The current export has passed the paper and figure synchronization checks. If any source changes, rerun both commands and review `manifest/figures.csv` before publishing.
