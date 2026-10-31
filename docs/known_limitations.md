# Known limitations

- The road-supported subspace is specific against rank-matched random
  conservation subspaces, but no rotated or rewired road-network intervention
  was run. The result does not show that road geometry causes the coupling.
- The fitted operator is an effective, observation-dependent object. Source
  and target are constructed from the same trajectories; this is not a
  forecast from independently observed demand.
- Chengdu has three held-out dates in the primary protocol. Its D interval
  and pairing null are descriptive. The matched-capacity sweep uses four
  alternate-hour dates and is likewise a small-sample summary.
- The exact Porto, San Francisco and Chengdu release identifiers and download
  dates were not recorded. They are marked UNKNOWN; the Chengdu aligned-field
  hash is recorded.
- The public repository does not include raw trajectories, model weights or
  large prediction files. Level C raw end-to-end reconstruction is not
  claimed.
- External aligned fields, road bases, freeze files and GIS rasters are
  required for Level B and are not bundled.
- The archived Graph WaveNet adaptation is a specific 300-epoch protocol, not
  a general claim about neural models.
- The Chengdu nonlinear comparison is exploratory and underpowered; it
  supports neither the presence nor the absence of nonlinearity.
