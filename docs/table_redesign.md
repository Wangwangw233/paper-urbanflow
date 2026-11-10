# Table 1 redesign

## Final main-table organization

Table 1 is a single D/FACE pooled-\(R^2\) table with three sections:

1. **Linear references and road-structure contrast**
   - linear ridge, POD, UrbanFlow road-supported operator;
   - Ridge + E2 road block (Porto only);
   - Ridge + E2 and two matched blocks (Porto only).
2. **Neural arms, frozen configurations**
   - MLP, U-Net, GraphGPS and Fourier operator;
   - source-only and source+road versions;
   - five-seed mean (s.d.); dashes for configurations not run.
3. **Neural arms, VAL-reselected (Porto)**
   - separately labeled because these are not the frozen Task-H configurations.

## Decisions

- UrbanFlow is included through its frozen D/FACE score (0.411/0.209/0.085),
  not through the ROAD-support D scores. This resolves the previous apparent
  contradiction between Fig. 2 and Table 1.
- The linear ridge and POD reference is described as the strongest
  **source-only** reference, not the strongest model overall. The Porto
  `Ridge + E2 road block` result (0.5405) remains visible.
- The road-block and matched-block rows are labeled as linear contrasts, not
  new model families; no selective bold or omitted strong baseline is used.
- GraphGPS is marked Porto-only by dashes. San Francisco and Chengdu are not
  presented as GraphGPS replications.
- The zero-departure reference is stated in the footnote (\(R^2=0\)); negative
  entries are explicitly interpretable as worse than the TRAIN phase mean.
- P/FACE, Graph WaveNet, Gravity, OD estimation and network-loading values are
  moved intact to Supplementary Table S2; no P/FACE value is placed in a
  D/FACE cell.
- The old Fig. 4d--e controls are retained in Supplementary Table S4 and
  Fig. S5, so moving them does not delete frozen results.

## Exclusions and limitations

- UrbanFlow is a single frozen fit; no seed distribution is invented.
- GraphGPS has no SF/Chengdu result; no value is extrapolated.
- The VAL-reselected Porto rows are separated from the frozen Task-H rows; their
  selection protocol differs and they are not pooled with them.
- The MLP Chengdu result is retained despite being negative; the zero-baseline
  footnote and the separate Chengdu diagnostic prevent overinterpretation.
