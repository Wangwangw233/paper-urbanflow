# Anisotropy

`anisotropy_null.py` reconstructs the road-coordinate operator, evaluates
local response and road-support tensors, and generates the 999-draw
spectrum-matched rotation null. `gtheta_provenance.py` audits the archived
`G_theta` estimates against that null and records the single primary source.

Both scripts require `URBANFLOW_ROOT` with external aligned fields, bases and
freeze files. Outputs: `results/frozen/anisotropy/`.
