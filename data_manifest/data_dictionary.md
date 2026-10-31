# Data dictionary

## Aligned flux cache (external, not bundled)

| Key | Shape / type | Meaning |
|---|---|---|
| `F_E` | `(K, ny, nx)` float | signed east-face crossings per window |
| `F_N` | `(K, ny, nx)` float | signed north-face crossings per window |
| `wdate` | `(K,)` int | integer date index |
| `wslot` | `(K,)` int | clock phase within the day |
| `wvalid` | `(K,)` bool | observed-window flag |
| `times` | `(K,)` int | original time index |
| `scale_m` | scalar | lattice spacing, metres |
| `l0_m` | scalar | gauge length, metres |
| `dt_min` | scalar | window length, minutes |
| `udates` | optional `(D,)` int | date lookup used by some caches |

## Geometry context (`um_ctx.npz`)

Keys follow `<city>__<field>`, for example `Porto__nx`, `Porto__ny`,
`Porto__NC`, `Porto__n_int`, `Porto__face_pairs_p0` and
`Porto__face_pairs_p1`. Cell 0 is the gauge cell.

## Reference package

`src/urbanflow/manifest.json` is content-addressed. Its `code` section hashes
the top-level package modules, `data` pins caches and freeze files, and
`road_basis`, `operators`, `metrics` and `splits` identify the numerical
objects. `src/urbanflow/_version.py` carries the expected digest and is
excluded from the code hash.

## Frozen result families

| Directory | Contents |
|---|---|
| `results/frozen/crosscity/` | pooled cross-city scores and snapshot metadata |
| `results/frozen/constitutive/` | per-fold constitutive objects, Chengdu response, D uncertainty |
| `results/frozen/spectra/` | spectra, joint-spectrum audit, phase-sensitive metrics |
| `results/frozen/anisotropy/` | anisotropy tensors, matched-capacity summary, null and provenance |
| `results/frozen/baselines/` | baseline date-freeze records and exploratory nonlinear results |
| `results/frozen/matched_capacity/` | rank-matched aggregate tables |
| `results/frozen/reconstruction/` | compact display products |
| `results/frozen/manifests/` | archived figure-build manifests |
