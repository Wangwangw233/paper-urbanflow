# Datasets

No individual-level trajectory records are bundled. The repository contains
aggregate fields, frozen scores and small display products only.

| City | Source | Release identifier | Native unit | Analysis grid | Dates used |
|---|---|---|---|---|---|
| Porto | ECML/PKDD taxi trajectories | UNKNOWN | taxi GPS records | 32 x 32, 250 m | 252 |
| San Francisco | Cabspotting | UNKNOWN | taxi GPS records | 32 x 88, 250 m | 15 |
| Chengdu | DiDi ride-hailing sample | UNKNOWN | trip records | 60 x 60, 250 m | 14 |

The exact Porto, San Francisco and Chengdu release identifiers and download
dates were not recorded in the archived analysis. They are marked UNKNOWN
rather than reconstructed from memory. The Chengdu aligned field used in the
final analysis has SHA-256
`c9330b54e8fbcf589e8b63d480b4db877b7d48291df941fb1348662a9dfb0289`.

Road geometry is derived from OpenStreetMap. Reuse must retain the
OpenStreetMap attribution and comply with the applicable OpenStreetMap and
source-dataset terms. No raw GPS coordinates, trip identifiers or vehicle
identifiers are included in this repository.
