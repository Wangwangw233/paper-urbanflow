# Matched-capacity media

The matched-capacity sweep compares road, mobility-inferred and
permuted-mobility media at equal nominal block count. Frozen outputs are in
`results/frozen/anisotropy/partF_summary.json`; the figure panel is generated
by `scripts/reproduce_figures/fig5_redesign.py`.

The sweep requires external aligned fields, date-freeze files and GIS rasters.
`archive/legacy/matched_capacity/` retains the original construction modules
for provenance; they import private dependencies and are not a standalone
public pipeline. Equal nominal capacity is not equal information.
