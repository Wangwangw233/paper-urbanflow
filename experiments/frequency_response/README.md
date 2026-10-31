# Frequency response

`phase_metrics.py` computes per-date complex coefficients and the three
frequency metrics: phase-sensitive `R_m^2`, date-pooled signed alignment and
the phase-insensitive magnitude from TEST-date mean coefficients.

Requires `URBANFLOW_ROOT` with external aligned fields, bases and freeze
files. Output: `results/frozen/spectra/phase_metrics.json`.
