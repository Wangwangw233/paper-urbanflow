"""The objects of UrbanFlow Reference v1 -- one table, no side copies.

Each entry fixes the five things a citation has to be able to point at:

    cache    the aligned face-flux field the operator is applied to
    freeze   the date protocol that produced TRAIN / VAL / TEST
    basis    the road-response subspace U (r columns)
    phases   how many clock phases a day has in this object
    family   taxi trajectory vs courier GPS -- not interchangeable

`geometry` names the entry in `um_ctx.npz`; it is what fixes the grid, the
face ordering, `D` and the study area.  Two objects may share it (SF and
Cabspotting do).
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------
P10 = "project10/outputs/p0_l0"
BASIS_DIR = "road_spatial_transfer/results"

# LaDe delivery window: 22 quality-eligible weekdays, chronological 60/20/20
# by count (13 / 4 / 5).  Identical in all five cities -- see Round-7
# `results/date_protocol.json`.  Frozen here so the split cannot drift with
# the code that recomputes it.
LADE_REFROZEN = {
    "train": [19072, 19073, 19074, 19075, 19076, 19079, 19080, 19081,
              19082, 19083, 19086, 19087, 19088],
    "val": [19089, 19090, 19093, 19094],
    "test": [19095, 19096, 19097, 19100, 19101],
    "eligible": [19072, 19073, 19074, 19075, 19076, 19079, 19080, 19081,
                 19082, 19083, 19086, 19087, 19088, 19089, 19090, 19093,
                 19094, 19095, 19096, 19097, 19100, 19101],
}

# Chengdu: 18-phase daytime window.  Five of the frozen held-out dates come
# from a different collection mechanism (9 alternating hours, N == 0 with
# normal flux), so the paper's chronological 60/20/20 is applied to the 14
# dates that are observed continuously between 06 and 23 h.
CHENGDU = {
    "hours": list(range(6, 24)),
    "dates": [16285, 16286, 16287, 16288, 16290, 16293, 16294, 16296,
              16297, 16300, 16301, 16302, 16303, 16304],
    "train": [16285, 16286, 16287, 16288, 16290, 16293, 16294, 16296],
    "val": [16297, 16300, 16301],
    "test": [16302, 16303, 16304],
}


def _traj(name, geometry, cache, freeze, basis, phases=24, folds=1):
    return dict(name=name, family="trajectory", geometry=geometry,
                cache=cache, freeze=freeze, basis=basis, phases=phases,
                hours=None, folds=folds, splits=("frozen",))


def _lade(name):
    return dict(name=name, family="courier_gps", geometry=name,
                cache=f"{P10}/{name.lower()}_aligned_l250_dt60.npz",
                freeze=f"{P10}/{name.lower()}_datefreeze.json",
                basis=dict(path=f"{BASIS_DIR}/road_basis_{name}.npz",
                           u="U", rank="rank"),
                phases=24, hours=None, folds=1,
                splits=("frozen", "refrozen"))


#: the finer time scales of Round 7.  Same city, same road basis, same date
#: protocol; only the field's `dt_min` changes, and with it the number of
#: clock phases per day (96 at 15 min, 48 at 30 min).  These caches are built
#: from raw crossing timestamps, not by splitting the 60-min field, so they
#: are separate fields and must be cited by their own hash.
SCALE_DIR = "reports/lade_temporal_scale_and_conditional_gain/aligned"


def _lade_scale(name, dt):
    s = dict(OBJECTS[name])
    s["name"] = f"{name}@{dt}"
    s["cache"] = f"{SCALE_DIR}/{name.lower()}_aligned_l250_dt{dt}.npz"
    s["phases"] = 24 * 60 // int(dt)
    s["dt_min"] = int(dt)
    return s


OBJECTS = {
    # --- the paper's own two -------------------------------------------
    "Porto": _traj(
        "Porto", "Porto",
        f"{P10}/porto_aligned_l250_dt60.npz",
        f"{P10}/porto_datefreeze.json",
        # the paper basis, not the archive one: every frozen E_topo in the
        # manuscript was computed with it
        dict(path="road_allocation_manifold/artifacts/UR_exact.npz",
             u="U", rank="r"),
    ),
    "SF": _traj(
        "SF", "SF",
        "project10/outputs/round6/sf_passenger_aligned_250m60m.npz",
        f"{P10}/cabspotting_datefreeze.json",
        dict(path=f"{BASIS_DIR}/road_basis_SF.npz", u="U", rank="rank"),
        folds=5,
    ),
    # --- external check ------------------------------------------------
    "Rome": _traj(
        "Rome", "Rome",
        f"{P10}/rome_aligned_l250_dt60.npz",
        "project10/outputs/round3/rome3a_datefreeze.json",
        dict(path=f"{BASIS_DIR}/road_basis_Rome.npz", u="U", rank="rank"),
    ),
    # --- the daytime window --------------------------------------------
    "Chengdu": dict(
        name="Chengdu", family="trajectory", geometry="Chengdu",
        cache=f"{P10}/chengdu_aligned_l250_dt60.npz",
        freeze=f"{P10}/chengdu_datefreeze.json",
        basis=dict(path=f"{BASIS_DIR}/road_basis_Chengdu.npz",
                   u="U", rank="rank"),
        phases=len(CHENGDU["hours"]), hours=CHENGDU["hours"],
        folds=1, splits=("window",),
    ),
    # --- the LaDe courier fleet ----------------------------------------
    "Shanghai": _lade("Shanghai"),
    "Hangzhou": _lade("Hangzhou"),
    "Chongqing": _lade("Chongqing"),
    "Yantai": _lade("Yantai"),
    "Jilin": _lade("Jilin"),
}

LADE_CITIES = ("Shanghai", "Hangzhou", "Chongqing", "Yantai", "Jilin")

#: finer time scales (`Shanghai@15`, ...).  Registered after the base fleet so
#: `ALL_OBJECTS` reads in the order of the paper: manuscript, external check,
#: window, fleet at 60 min, then the scale arms.
for _c in LADE_CITIES:
    for _dt in (15, 30):
        OBJECTS[f"{_c}@{_dt}"] = _lade_scale(_c, _dt)

#: objects whose numbers appear in the manuscript
PAPER_OBJECTS = ("Porto", "SF")
#: everything v1 can evaluate out of the box
ALL_OBJECTS = tuple(OBJECTS)


def spec(name):
    if name not in OBJECTS:
        raise KeyError(f"unknown object {name!r}; v1 has {ALL_OBJECTS}")
    return OBJECTS[name]
