r"""The two bottom-row statistics, for THREE cities, from frozen readers.

Panels (e) and (f) of Figure 5 are read by the manuscript's own
`fig5_ablation.nulls()` / `sweep()`, which cover Porto and San Francisco.  This
module keeps those readers for those two cities -- they re-assert their frozen
anchors in-process -- and adds Chengdu from this round's two computations:

    results/partA_chengdu.json   the 999-seed rank-matched null of the SAME
                                 estimator (scripts/cd_partA_null.py, host 164),
                                 with the Porto control re-run in the same job.
    results/partF_summary.json   the equal-capacity medium sweep of the SAME
                                 estimator (scripts/cd_partF_sweep.py, host 164),
                                 with Porto and San Francisco REPRODUCED cell by
                                 cell before Chengdu was computed.

Nothing is typed in: every Chengdu number is read from those two files, and
every cross-file consistency relation is asserted here, so a stale or partial
run raises instead of drawing.

NOT SUPPORTED BY THE ARCHIVE, AND NOT CLAIMED: Chengdu has a single
mobility cache (`default`), so its (f) row is on the all-motion channel whereas
San Francisco's is on the occupancy-defined `passenger` channel, and its frozen
split has 4 held-out dates, so its paired-date interval resamples 4 dates.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
REV = HERE.parent
ROOT = pathlib.Path(os.environ.get('URBANFLOW_ROOT', REPO_ROOT))
FIGS = REPO_ROOT / "figures" / "main"
if str(FIGS) not in sys.path:
    sys.path.insert(0, str(FIGS))

import figstyle as FS                     # noqa: E402
import fig5_ablation as A5                # noqa: E402

DATA_REV = ROOT / "results" / "frozen" / "anisotropy"
CD_A = DATA_REV / "partA_chengdu.json"
CD_F = DATA_REV / "partF_summary.json"
CITY3 = ("Porto", "SF", "Chengdu")


def _load(p):
    if not p.exists():
        raise SystemExit("missing %s -- run the 164 job that writes it" % p)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (e) specificity
# ---------------------------------------------------------------------------
def nulls3():
    """dE = E_null - E_road, 999 rank-matched draws, one row per city."""
    base = A5.nulls()                     # Porto, San Francisco (+ its asserts)
    out = {"Porto": base["Porto"], "SF": base["SF"]}

    j = _load(CD_A)
    ctrl, row = j["control"], j["Chengdu"]
    if np.isnan(float(ctrl["max_abs_e_null_diff"])):
        raise SystemExit("partA_chengdu.json is a PILOT (no Porto control)")
    assert float(ctrl["max_abs_e_null_diff"]) < 1e-12, ctrl
    e = np.asarray(row["e_null"], float)
    seeds = np.asarray(row["seeds"], np.int64)
    assert e.size == 999 and seeds.size == 999, e.size
    assert np.all(np.diff(seeds) == 1), "seeds are not contiguous"
    road = float(row["summary"]["e_road"])
    d = e - road
    out["Chengdu"] = dict(de=d, road=road, q01=float(np.percentile(d, 1)),
                          n=int(e.size),
                          p_emp=float(row["summary"]["p_emp"]),
                          n_at_least_as_good=int(
                              row["summary"]["n_at_least_as_good"]),
                          n_dates=int(row["summary"]["n_dates"]),
                          m=int(row["summary"]["m"]),
                          r_road=int(row["summary"]["r_road"]),
                          dim_ker_D=int(row["summary"]["dim_ker_D"]))
    for city in CITY3:
        d = out[city]["de"]
        assert d.min() > 0.0, (city, "the null distribution reaches the road basis")
    print("%-8s E_road %.4f  dE %.4f..%.4f (median %.4f, q01 %.4f)  n %d"
          % ("Chengdu", out["Chengdu"]["road"], d.min(), d.max(),
             float(np.median(out["Chengdu"]["de"])), out["Chengdu"]["q01"],
             out["Chengdu"]["n"]))
    return out


# ---------------------------------------------------------------------------
# (f) incompleteness
# ---------------------------------------------------------------------------
def sweep3():
    """Gain over a constant medium at the matched capacity R = n_int."""
    base = A5.sweep()                     # Porto, San Francisco (+ its asserts)
    out = {"Porto": base["Porto"], "SF": base["San Francisco"]}

    j = _load(CD_F)
    # the reproduction gate the 164 job ran must be in the file, and passed
    for city, label in (("Porto", "Porto"), ("SF", "San Francisco")):
        rep = j[city]["reproduction"]
        assert rep["max_abs_diff"] < 1e-9, (city, rep)
        assert rep["pairwise_max_abs_diff"] < 1e-9, (city, rep)
        # the two independent readers (frozen CSV vs this job) must agree
        FS.assert_close(float(j[city]["D0_R2"]), out[city]["base"], 1e-9,
                        "%s D0 across readers" % city)
    c = j["Chengdu"]
    nint = int(c["n_int"])
    d = {r["source"]: float(r["R2_cond"]) for r in c["results"]
         if int(r["R"]) == nint}
    basev = float(c["D0_R2"])
    # the anchors the job recorded at R = n_int must be the sweep's own rows
    for src, v in c["anchors_R_eq_n_int"].items():
        assert abs(float(v) - d[src]) < 1e-12, (src, v, d[src])
    pw = [p for p in c["pairwise"]
          if p["contrast"] == "mobility - road" and int(p["R"]) == nint][0]
    out["Chengdu"] = dict(nint=float(nint), base=basev,
                          road=d["road"] - basev, mob=d["mobility"] - basev,
                          shuf=d["mobility_shuf"] - basev,
                          mob_bin=d["mobility_bin"] - basev,
                          road_shuf=d["road_shuf"] - basev,
                          gap=float(pw["delta_mean"]),
                          gap_ci=(float(pw["ci95_lo"]), float(pw["ci95_hi"])),
                          gap_p=float(pw["p_one_sided"]),
                          n_dates=int(pw["n_dates"]),
                          channel=c["frozen_registry"]["channel"],
                          basis=c["frozen_registry"]["basis"])
    FS.assert_close(out["Chengdu"]["gap"],
                    out["Chengdu"]["mob"] - out["Chengdu"]["road"], 1e-6,
                    "Chengdu mobility-road gap")
    # held-out date count of the frozen rows, for the reader of the intervals
    import csv
    with open(A5.PW_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["contrast"] != "mobility - road":
                continue
            for key, label in (("Porto", "Porto"), ("SF", "San Francisco")):
                if r["city"] == label and int(r["R"]) == int(out[key]["nint"]):
                    out[key]["n_dates"] = int(r["n_dates"])
    print("%-8s R = %d  road %+.4f  mob %+.4f  shuf %+.4f  mob-road %+.4f "
          "[%+.4f,%+.4f]  (%d dates)"
          % ("Chengdu", nint, out["Chengdu"]["road"], out["Chengdu"]["mob"],
             out["Chengdu"]["shuf"], out["Chengdu"]["gap"],
             out["Chengdu"]["gap_ci"][0], out["Chengdu"]["gap_ci"][1],
             out["Chengdu"]["n_dates"]))
    return out


if __name__ == "__main__":
    nulls3()
    print()
    sweep3()
