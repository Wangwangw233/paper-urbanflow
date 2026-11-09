# Validation report

## Repository state

- Branch: `release/urban-flow` (renamed from `release/nature-cities`)
- Refactor source commit before the public push: `d743958e2ec52bf211ee6b507ca4128e4dfdffad`
- Public push commit: `0dc73b2742850723d9d364cd2948f69108e70319`
- Files: 216 total; `data_manifest/files.csv` lists 215 (the manifest excludes itself).
- Python files: 53.
- NPZ archives: 12, containing 295 numeric/Unicode arrays; all loaded with `allow_pickle=False`.
- Frozen result files: 59 checksum entries including the 11 reference PDFs.

## Checks executed

```bash
python -m unittest discover -s tests -v
python scripts/audit/verify_public_repo.py --figure-smoke
```

Results:

- Unit tests: 13 passed. They cover `D F = x`, `D U_R = 0`,
  `U_R^T U_R = I`, Hodge reconstruction, the synthetic frame identity,
  manifest/code hashes, configuration loading and frozen-product checks.
- Repository audit: 17 passed, 0 failed.
- Figure smoke: all 11 figure PDFs were regenerated into a temporary
  external-data mirror.
- Table smoke: both table products were regenerated from frozen scores.
- Privacy scan: no raw trajectory fields, individual identifiers, private
  workstation paths, remote-shell operations or embedded private strings.
- Reference CLI: digest
  `b7b85d7864eaf93c7916b1f4dfd24e9d23a4c7c807457c5767558905a9860566`.

## Not executed

- External-data core regression over every reference object.
- Raw-trajectory end-to-end rebuilding, registration and model training.
- Rotated or rewired road-network intervention.

These are stated as limitations rather than replaced by inferred results.
