# Constitutive coupling

`d_uncertainty.py` computes primary D-score uncertainty: 10,000 date-level
bootstrap resamples within folds and a 199-draw source-routing pairing null
that refits the ridge operator in each draw. Chengdu's three held-out dates
make its interval and null descriptive.

Requires `URBANFLOW_ROOT` with the external aligned fields, bases and freeze
files. Output: `results/frozen/constitutive/d_uncertainty.json`.
