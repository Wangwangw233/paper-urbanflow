# Result provenance

| Claim / product | Code | Frozen result | Figure |
|---|---|---|---|
| P/D/CLIM scores | `src/urbanflow/frames.py`, `reference.py` | `results/frozen/crosscity/crosscity_results.csv`; `results/frozen/constitutive/obj_*.json` | `figures/main/fig02_threecity_coupling.pdf` |
| Date-level D uncertainty and pairing null | `experiments/constitutive_coupling/d_uncertainty.py` | `results/frozen/constitutive/d_uncertainty.json` | Fig. 2 error bars |
| Phase-sensitive frequency response | `experiments/frequency_response/phase_metrics.py` | `results/frozen/spectra/phase_metrics.json` | `figures/main/fig03_frequency_response.pdf` |
| Joint three-city spectrum | `scripts/reproduce_figures/make_figures.py` | `results/frozen/spectra/verified_joint.npz`; `joint_spectrum_audit.json` | `figures/supplementary/figS01_joint_spectrum_2d.pdf` |
| Anisotropy tensors and orientation null | `experiments/anisotropy/anisotropy_null.py`; `gtheta_provenance.py` | `results/frozen/anisotropy/fo4_fig5_data.npz`; `chengdu_anisotropy.npz`; `anisotropy_null.json`; `gtheta_provenance.json` | `figures/main/fig04_anisotropy_road_tests.pdf` |
| Rank-matched road specificity | archived matched-capacity code | `results/frozen/anisotropy/partA_topology.json`; `partA_chengdu.json`; `results/frozen/matched_capacity/` | Fig. 4d |
| Matched-capacity sufficiency | archived matched-capacity code | `results/frozen/anisotropy/partF_summary.json` | Fig. 4e |
| Model comparison | archived baseline outputs | `results/frozen/crosscity/crosscity_results.csv`; `results/frozen/baselines/splits/` | Table 1 |
| Chengdu nonlinear exploration | `experiments/baseline_comparison/chengdu_nonlinear_exploratory.py` | `results/frozen/baselines/chengdu_nonlinear/` | `figS07_nonlinear_exploratory.pdf` |

The reference digest is `b7b85d7864eaf93c7916b1f4dfd24e9d23a4c7c807457c5767558905a9860566`.
It covers the package code and the pinned identifiers of the external data;
it does not encode measured floats. The upstream and packaging provenance is
recorded in `data_manifest/provenance.json`.
