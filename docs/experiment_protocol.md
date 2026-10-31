# Experiment protocol

## Splits and leakage control

The authoritative split dates are stored in `src/urbanflow/manifest.json`.
Porto uses one chronological split, San Francisco five blocked folds, and
Chengdu one 14-date observed-block split. The Chengdu matched-capacity sweep
uses a separate 10/4/4 chronological split.

Kappa is selected on VAL only. TEST is evaluated after selection. The
date-level bootstrap resamples held-out dates within each fold; faces and hours
are never treated as independent units.

## Scoring frames

- **P**: centre TRAIN rows only, predict raw TEST source, score raw TEST target.
- **D**: centre every row with TRAIN phase means, score the departure.
- **I**: D prediction plus the TRAIN phase mean, score the raw target.
- **CLIM**: TRAIN phase mean alone, score the raw target.

The identity `R_I - R_CLIM = (1 - R_CLIM) R_D` is checked at runtime; a
failure is an implementation error, not a result.

## Uncertainty

The primary D scores use 10,000 date-level bootstrap resamples within folds
and a 199-draw source-routing pairing null that permutes complete date blocks
within each split and refits the operator. Chengdu has three held-out dates;
its interval and null are descriptive, not significance tests.

The anisotropy analysis uses 999 spectrum-matched rotations of `K_R` in the
road output space. The archived anisotropy tensors are the single primary
source for `G_theta`; the control refit is used only to generate the rotation
null. See `results/frozen/anisotropy/gtheta_provenance.json`.

## Baselines

Graph WaveNet cells use archived pooled per-date error numerators and
denominators. Sealed-seed results are robustness checks, not a second pooled
ranking. Loading and OD-estimation rows diagnose the implemented pipelines;
they are not source-only inference results.
