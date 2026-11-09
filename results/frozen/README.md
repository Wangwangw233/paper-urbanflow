# Frozen results

Final JSON, CSV and NPZ products are grouped by result family. File hashes are
recorded in `data_manifest/files.csv` and the frozen-result checksums are in
`data_manifest/checksums.sha256`. The archived tensors used for the primary
anisotropy estimate are
`anisotropy/fo4_fig5_data.npz` and `anisotropy/chengdu_anisotropy.npz`.

Additional result families:

- `road_medium/`: E/F/G road-modulated operator, matched-media and prior
  aggregates;
- `modern_benchmark/`: Task H D/FACE modern-baseline aggregates;
- `identification/`: Task I Porto identification and attribution records.
