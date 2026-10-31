# Preprocessing

The released analysis starts from aligned face-flux caches. The original raw
trajectory registration is not part of this repository.

1. Project each city onto a regular 250 m lattice.
2. Accumulate signed crossings on interior lattice faces for each
   `(date, clock phase)` window.
3. Keep the two directional face families `F_E` and `F_N`. A missing window is
   omitted, never filled with zero.
4. Store the window date, phase, validity flag and resolution metadata beside
   the flux arrays.
5. Use the frozen TRAIN/VAL/TEST date lists in `src/urbanflow/manifest.json`.
   No TEST date participates in kappa selection or model selection.

Chengdu uses the observed 06:00-23:59 block (18 phases) without night-time
imputation. Porto and San Francisco use complete 24-hour days. The Chengdu
matched-capacity sweep uses a separate chronological 10/4/4 split, as recorded
in `configs/data/cities.yaml`.

The low-level face-ordering context is
`routing_urban_medium_constitutive/results/um_ctx.npz`. Its face pairs are
asserted against the cache mask when a field is loaded.
