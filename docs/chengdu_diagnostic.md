# Chengdu benchmark diagnostic

## Frozen protocol

- Chengdu window split: 8 TRAIN / 3 VAL / 3 TEST dates.
- Observed clock window: 06:00–23:59, 18 phases per date.
- Design rows: 8×18 = 144 TRAIN, 3×18 = 54 TEST; 3×18 = 54 VAL.
- Grid: 60×60 = 3,600 cells; 7,080 interior faces.
- Source: \(\delta x_t\); target: full FACE departure \(\delta F_t^\circ\).
- Output is projected by the exact \(P_0\). All arms use the same frozen
  splits, target, standardization and pooled score.
- TEST dates are 16302, 16303 and 16304; TEST is read once after VAL selection.

## Target scale and zero baseline

The pooled TEST target energy is the sum of the per-date denominators in the
frozen summary: 41,654,049.36 + 40,276,949.74 + 44,762,939.40 =
126,693,938.50. The zero-departure reference therefore has \(R^2=0\) by
construction. A negative \(R^2\) is worse than predicting the TRAIN phase
mean; it is not a zero-energy or undefined-score artifact.

The source-free/linear reference is 0.1258 for ridge and 0.1262 for POD.

## Frozen model results

| Model | VAL \(R^2\) | TEST mean | TEST sd | TEST min–max | TEST per date |
|---|---:|---:|---:|---:|---|
| MLP, source | −0.0794 | −0.0697 | 0.0046 | −0.0740…−0.0631 | −0.0428, −0.0937, −0.0730 |
| MLP, source + road | −0.0407 | −0.0194 | 0.0126 | −0.0333…−0.0009 | 0.0080, −0.0435, −0.0230 |
| U-Net, source | 0.0178 | 0.0120 | 0.0041 | 0.0049…0.0149 | 0.0183, 0.0088, 0.0091 |
| U-Net, source + road | 0.0220 | 0.0125 | 0.0030 | 0.0074…0.0146 | 0.0162, 0.0094, 0.0117 |
| Fourier operator, source | 0.0004 | 0.0003 | 0.0002 | 0.0001…0.0005 | 0.0006, −0.00002, 0.0003 |
| Fourier operator, source + road | 0.0225 | 0.0069 | 0.0022 | 0.0044…0.0094 | 0.0140, 0.0006, 0.0059 |
| Ridge | — | 0.1258 | — | — | — |

The MLP is below the zero baseline on VAL as well as TEST. This rules out a
TEST-only evaluation error or a test-time leakage explanation. The U-Net is
positive on both VAL and TEST under the same target, split, mask, P0 projection
and standardizer, and the Fourier arm is positive on VAL and near zero on
TEST. The anomaly is therefore model-specific, not a shared scoring or field
construction failure.

## Implementation checks

- `h_common.grid_info` asserts the frozen face order (east family followed by
  north family); `load_city` round-trips the target through the grid views and
  aborts on a mismatch.
- `h_train.prepare` uses TRAIN-only standardization and aborts if the road block
  collapses to a constant; the corrected `fixed` road standardization is
  active.
- `h_train.evaluate` applies the same dense \(P_0\) to every arm; the ridge
  reference reproduces the frozen 0.1258 score.
- `h_train` uses VAL-only early stopping and does not read TEST for selection.
- The same pipeline produces positive U-Net/FNO scores, so a global mask,
  face-orientation or projection defect is not consistent with the pattern.

## What can and cannot be concluded

**Supported:** the Chengdu MLP is a genuine small-sample model failure at the
frozen Porto-selected configuration; its VAL and TEST scores are negative, and
the road-input variant remains negative. The linear ridge is a stronger
small-sample baseline in this city.

**Not established:** whether the failure is optimization, capacity, or a
specific architecture/data interaction. The archived aggregate summaries do
not include per-epoch histories or prediction tensors in this workspace, so
those mechanisms cannot be separated without a rerun. No implementation error
was found, and no number in Table 1 was changed.

**Presentation consequence:** the MLP rows are retained with the zero-baseline
note, but the paper does not describe this as evidence that deep models cannot
learn the map. The U-Net and Fourier results are reported separately; no
universal deep-learning conclusion is drawn.
