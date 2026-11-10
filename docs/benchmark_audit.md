# Benchmark audit

## Scope and protocol

This audit covers the D/FACE comparison in main Table 1 and the frozen
Round-H/Benchmark outputs. Every arm uses:

- input: source departure \(\delta x_t\);
- target: full FACE routing departure \(\delta F_t^\circ\);
- TRAIN phase means removed; exact \(P_0\) projection on the output;
- frozen splits (Porto 151/50/51 dates; SF 9/3/3 per fold; Chengdu 8/3/3,
  06–23 h);
- pooled TEST \(R^2=1-\sum\|\delta F-\delta\hat F\|^2/\sum\|\delta F\|^2\);
- VAL-only selection, five final seeds for neural arms, one frozen fit for
  linear arms.

The zero-departure reference therefore has \(R^2=0\), not a positive score.
A negative score means the prediction is worse than predicting the TRAIN phase
mean.

## Audited Table 1 values

| Arm | Porto | San Francisco | Chengdu | Source |
|---|---:|---:|---:|---|
| Linear ridge | 0.5044 | 0.2582 | 0.1258 | frozen cross-city score |
| POD regression | 0.5044 | 0.2670 | 0.1262 | frozen cross-city score |
| UrbanFlow (road-supported) | 0.4110 | 0.2091 | 0.0851 | frozen `cc_families` D/FACE OURS |
| MLP, source | 0.4213 (0.0095) | 0.0701 (0.0038) | −0.0697 (0.0046) | H aggregate, five seeds |
| MLP, source + road | 0.4473 (0.0187) | 0.0975 (0.0058) | −0.0194 (0.0126) | H aggregate, five seeds |
| U-Net, source | 0.1827 (0.0148) | 0.1985 (0.0235) | 0.0120 (0.0041) | H aggregate, five seeds |
| U-Net, source + road | 0.1616 (0.0050) | 0.2319 (0.0122) | 0.0125 (0.0030) | H aggregate, five seeds |
| GraphGPS, source | 0.1575 (0.0147) | — | — | Porto-only |
| GraphGPS, source + road | 0.0193 (0.0240) | — | — | Porto-only |
| Fourier operator, source | 0.0288 (0.0100) | 0.1419 (0.0064) | 0.0003 (0.0002) | H aggregate, five seeds |
| Fourier operator, source + road | 0.1097 (0.0112) | 0.2073 (0.0061) | 0.0069 (0.0022) | H aggregate, five seeds |
| Ridge + E2 road block | 0.5405 | — | — | Porto-only linear contrast |
| Ridge + E2 and two matched blocks | 0.5343 | — | — | Porto-only linear contrast |

The main table does not mix P/FACE, ROAD-support, or network-loading scores.
The archived P/FACE, Graph WaveNet, Gravity and loading table is retained as
Supplementary Table S2.

## Cross-checks

- The MLP/U-Net/FNO values reproduce the five-seed means and standard
  deviations in `reports/roundH_modern_baselines/outputs/agg/*_summary.json`.
- The UrbanFlow D/FACE row is the frozen `cc_families` D/FACE `OURS` score,
  not a refit; the ROAD-support D scores in Fig. 2 are a different scoring
  space and are not substituted for it.
- GraphGPS has no San Francisco or Chengdu result in the frozen outputs; the
  table uses dashes rather than an inferred value.
- The `Ridge + E2` row is a linear contrast within the same family, not a new
  model class; it is not used to claim a universal road advantage.
- `Ridge + E2 and two matched blocks` uses the same frozen E2 response plus two
  same-size null response blocks; it is a sensitivity arm, not a new main
  model.

## Findings

No missing value was filled, no P/FACE number was copied into a D/FACE cell,
and no GraphGPS replication was invented. The remaining anomaly is the
Chengdu MLP/U-Net/FNO behavior, audited separately in
`chengdu_diagnostic.md`.
