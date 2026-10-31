# Field construction

`chengdu_frozen_response.py` reconstructs the Chengdu P-frame snapshot and the
18-point response coefficients from the pinned aligned field and road basis.
It requires `URBANFLOW_ROOT` to point at the external frozen-data tree and
fails if the archived scores do not reproduce.

Outputs are written to `results/frozen/constitutive/`.
