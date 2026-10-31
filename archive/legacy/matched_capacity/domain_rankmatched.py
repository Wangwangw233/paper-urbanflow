r"""Rank-matched diagnostic: is `M_inf`'s advantage its SOURCE or its DOF?

    python domain_rankmatched.py

THE QUESTION
------------
`M_inf` (adapted RECM 2024) reaches a higher held-out R2_cond than both
cross-spectral rows, but it fits 1984 (Porto) / 5512 (SF) medium scalars against
our 0.  Two competing explanations:

    H1 CAPACITY  any medium with enough free per-face conductances fits better,
                 whatever it was derived from;
    H2 SOURCE    a medium inferred from observed mobility carries spatial
                 information the road raster does not.

THE DESIGN
----------
Both sources are given the SAME nested, exactly-matched medium DOF and pushed
through the SAME frozen minimum-energy lift and scored on the SAME held-out
target:

    W_S(R) = mean of W_S over each of R spatial blocks, then the frozen
             build_w normalisation, so W_S(R) is piecewise constant with
             exactly R free conductances.

    S = road       the frozen OSM road raster (external, 0 fitted parameters)
    S = mobility   the RECM Eq.(5) conductance, TRAIN dates only, frozen
    S = *_shuf     the same per-face values spatially permuted before block
                   averaging: same R, same value multiset, NO spatial content

R is swept 1 -> n_int.  At R = 1 every source collapses to a constant medium,
and a constant medium gives T_W = T_0 exactly, i.e. the D0 row.  At R = n_int
the blocks are single faces and the two ends reproduce the frozen anchors
exactly: mobility -> M_inf, road -> D1.  Both endpoints are asserted.

The diagnostic is then the paired contrast at MATCHED R:

    d(R) = R2(mobility, R) - R2(road, R)

H1 predicts d(R) ~ 0 everywhere once R is large and the shuffled rows tracking
the real ones.  H2 predicts d(R) > 0 persistently and shuffled << real.

Read-only on the archive; nothing frozen is modified.
"""
from __future__ import annotations

import csv
import json

import numpy as np

import bench_lib as B
import de_lib as DE
import domain_lib as DL

RUNS = (("Porto_allmotion", "Porto"), ("SF_passenger", "San Francisco"))
ANCHORS = {"Porto": {"mob": 0.704736, "road": 0.622108},
           "San Francisco": {"mob": 0.828449, "road": 0.753284}}
BLOCK_SIZES = (32, 16, 8, 4, 2, 1)      # then the identity (one block per face)
SHUF_SEED = 23                          # the frozen BOOT_SEED; only a permutation


# ---------------------------------------------------------------------------
def face_centroids(ctx):
    """Grid coordinates of each interior face's midpoint, from `D` alone."""
    Dm, nx = np.asarray(ctx["D"], np.float64), int(ctx["nx"])
    r, c = np.nonzero(Dm)
    v = Dm[r, c]
    n_int = Dm.shape[1]
    p0 = np.empty(n_int, np.int64)
    p1 = np.empty(n_int, np.int64)
    p0[c[v > 0]] = r[v > 0]
    p1[c[v < 0]] = r[v < 0]
    i0, j0 = np.divmod(p0, nx)
    i1, j1 = np.divmod(p1, nx)
    return 0.5 * (i0 + i1), 0.5 * (j0 + j1)


def block_id(ctx, size):
    """Nested block partition of the faces.  size = 1 -> one block per face."""
    u, w = face_centroids(ctx)
    ny, nx = int(ctx["ny"]), int(ctx["nx"])
    if size == 1:                        # the identity: every face its own block
        return np.arange(len(u))
    nby, nbx = int(np.ceil(ny / size)), int(np.ceil(nx / size))
    bi = np.minimum((u // size).astype(np.int64), nby - 1)
    bj = np.minimum((w // size).astype(np.int64), nbx - 1)
    return bi * nbx + bj


def blockify(v, bid):
    """Piecewise-constant (per-region mean) projection of a per-face vector.

    `bid` must be a dense id range 0..max(bid); empty regions keep mean 0 but are
    never selected, so the EFFECTIVE medium DOF is `len(np.unique(bid))`.
    """
    nb = int(bid.max()) + 1
    tot = np.bincount(bid, weights=v, minlength=nb)
    cnt = np.bincount(bid, minlength=nb).astype(np.float64)
    return (tot / np.maximum(cnt, 1.0))[bid]


def normalise(v):
    """The frozen r7_common.build_w rule, verbatim: mean-1, floor 0.02."""
    w = np.maximum(v, 1e-300)
    return np.maximum(w / w.mean(), 0.02)


def apply_TW(Dm, W, X):
    """T_W X^T for X (n_test, NC):  W * (D^T (D W D^T + J)^-1 X^T).

    Same operator as domain_lib.lift_from_W, evaluated against the right-hand
    sides directly instead of materialising the dense NC x n_int lift.
    Validated against lift_from_W at every block size (assert_tw).
    """
    N = Dm.shape[0]
    J = np.outer(np.ones(N), np.ones(N)) / N
    M = (Dm * W[None, :]) @ Dm.T
    z = np.linalg.solve(M + J, X.T)
    return (W[:, None] * (Dm.T @ z)).T


def assert_tw(ctx, W, xTe):
    fast = apply_TW(np.asarray(ctx["D"], np.float64), W, xTe)
    slow = (DL.lift_from_W(ctx["D"], W) @ xTe.T).T
    return float(np.abs(fast - slow).max())


# ---------------------------------------------------------------------------
def sources(ctx):
    """The two structural sources, plus their spatial nulls and a support-only read.

    `mobility_bin` keeps only the SUPPORT of the mobility medium (the faces its
    conductance puts above the frozen 0.02 floor) at full unit amplitude: it has
    the same DOF as `mobility` but discards every amplitude difference, so it
    separates "which faces are active" from "how conductive each face is".
    """
    road = DE.operator_of(ctx["op"], DE.road_raster(ctx["op"]))[1]
    mob = DL.mobility_conductance(ctx, kind="net")
    rng = np.random.default_rng(SHUF_SEED)
    binr = np.where(mob > 0.02 + 1e-12, 1.0, 0.02)
    return {"road": road, "mobility": mob, "mobility_bin": binr,
            "road_shuf": road[rng.permutation(len(road))],
            "mobility_shuf": mob[rng.permutation(len(mob))]}


def build_fields(ctx, W):
    """Fbar + T_W x_t on every fold's held-out rows, through the frozen routine.

    The operator is factored ONCE per conductance and reused across folds; the
    arithmetic per fold is the same as `apply_TW`.
    """
    Dm = np.asarray(ctx["D"], np.float64)
    N = Dm.shape[0]
    J = np.outer(np.ones(N), np.ones(N)) / N
    A = (Dm * W[None, :]) @ Dm.T + J
    try:
        from scipy.linalg import lu_factor, lu_solve
        lu = lu_factor(A)
        solve = lambda B: lu_solve(lu, B)
    except Exception:                                   # numpy fallback
        solve = lambda B: np.linalg.solve(A, B)
    out = []
    for fo in ctx["folds"]:
        te = np.asarray(fo["te"])
        Fbar = DL._routine(ctx, fo, te)
        xTe = np.asarray(fo["x_te"], np.float64)
        lift = (W[:, None] * (Dm.T @ solve(xTe.T))).T
        out.append(Fbar + lift)
    return out


# ---------------------------------------------------------------------------
def main():
    results, pairwise, meta = [], [], {}
    for run, city in RUNS:
        ctx = B.load(run)
        n_int, ny, nx = int(ctx["n_int"]), int(ctx["ny"]), int(ctx["nx"])
        SRC = sources(ctx)

        # ---- endpoints and operator self-checks ---------------------------
        te0 = np.asarray(ctx["folds"][0]["te"])
        xTe0 = np.asarray(ctx["folds"][0]["x_te"], np.float64)
        for nm in ("road", "mobility"):
            e = assert_tw(ctx, SRC[nm], xTe0)
            assert e < 1e-9, (city, nm, e)
        print(f"[{city}] T_W fast path vs lift_from_W: max err "
              f"{max(assert_tw(ctx, SRC[nm], xTe0) for nm in ('road','mobility')):.1e}")

        g = lambda fields: B.score(ctx, fields, tag="x")["R2_routing"]

        # ---- the sweep ----------------------------------------------------
        # One nested partition per block size, ending at the identity (one block
        # per face).  `R` is the EFFECTIVE medium DOF: the number of distinct
        # regions actually carried by the faces, not the nominal block count.
        # The sweep starts at ONE region (every source collapses to a constant
        # medium there, giving T_W = T_0 exactly) and ends at the identity.
        parts, seen, RS = [(0, np.zeros(n_int, np.int64), 1)], {1}, [1]
        for s in BLOCK_SIZES:
            bid = block_id(ctx, s)
            R = int(len(np.unique(bid)))
            if R in seen:
                continue
            seen.add(R)
            RS.append(R)
            parts.append((s, bid, R))
        assert RS[-1] == n_int, (city, RS[-1], n_int)
        meta[city] = {"R": RS, "n_int": n_int, "ny": ny, "nx": nx,
                      "block_sizes": [s for s, _, _ in parts]}
        print(f"        R sweep: {RS}")

        rec = {}
        for s, bid, R in parts:
            for nm, W in SRC.items():
                r = g(build_fields(ctx, normalise(blockify(W, bid))))
                rec[(nm, R)] = r
                results.append({"city": city, "source": nm, "R": R, "medium_dof": R,
                                "block_size": s, "R2_cond": r["date_mean"], "_rec": r})

        # ---- endpoint anchors, from the sweep itself ----------------------
        # R = 1 collapses every source to a constant medium, which gives
        # T_W = T_0 exactly, i.e. the D0 row.  R = n_int is the identity
        # partition, i.e. the full per-face medium: the frozen M_inf and D1.
        r0 = B.score(ctx, DL.field_D0(ctx), tag="D0")["R2_routing"]
        assert RS[0] == 1 and abs(rec[("road", 1)]["date_mean"] - r0["date_mean"]) < 1e-12
        for nm in ("road", "mobility"):
            assert abs(rec[(nm, 1)]["date_mean"] - r0["date_mean"]) < 1e-12
        for nm, key in (("mobility", "mob"), ("road", "road")):
            got = rec[(nm, n_int)]["date_mean"]
            assert abs(got - ANCHORS[city][key]) < 5e-4, (city, nm, got)
        print(f"        R=1 -> {rec[('road',1)]['date_mean']:.9f} (D0 {r0['date_mean']:.9f}) OK;"
              f"  R={n_int} -> mob {rec[('mobility',n_int)]['date_mean']:.6f}"
              f" road {rec[('road',n_int)]['date_mean']:.6f} OK")

        # ---- matched-R paired contrasts -----------------------------------
        for R in RS:
            a, b = rec[("mobility", R)], rec[("road", R)]
            p = B.paired(a["r2_per_date"], a["dates"], b["r2_per_date"], b["dates"])
            pairwise.append({"city": city, "contrast": "mobility - road", "R": R,
                             "delta_mean": p["delta_mean"], "ci95_lo": p["ci95"][0],
                             "ci95_hi": p["ci95"][1],
                             "frac_dates_A_gt_B": p["frac_dates_A_gt_B"],
                             "p_one_sided": p["p_one_sided"], "n_dates": p["n_dates"],
                             "n_boot": p["B"]})
            sa, sb = rec[("mobility_shuf", R)], rec[("mobility", R)]
            p2 = B.paired(sb["r2_per_date"], sb["dates"], sa["r2_per_date"], sa["dates"])
            pairwise.append({"city": city, "contrast": "mobility - mobility_shuf", "R": R,
                             "delta_mean": p2["delta_mean"], "ci95_lo": p2["ci95"][0],
                             "ci95_hi": p2["ci95"][1],
                             "frac_dates_A_gt_B": p2["frac_dates_A_gt_B"],
                             "p_one_sided": p2["p_one_sided"], "n_dates": p2["n_dates"],
                             "n_boot": p2["B"]})

        # ---- the reading --------------------------------------------------
        print(f"        {'R':>6} {'road':>9} {'mob':>9} {'mob-road':>9}"
              f" {'m_shuf':>9} {'mob-shuf':>9} {'r_shuf':>9} {'mob_bin':>9}")
        for R in RS:
            m = lambda nm: rec[(nm, R)]["date_mean"]
            print(f"        {R:>6} {m('road'):>9.6f} {m('mobility'):>9.6f}"
                  f" {m('mobility')-m('road'):>+9.6f} {m('mobility_shuf'):>9.6f}"
                  f" {m('mobility')-m('mobility_shuf'):>+9.6f}"
                  f" {m('road_shuf'):>9.6f} {m('mobility_bin'):>9.6f}")

    # ---- write ------------------------------------------------------------
    cols = ["city", "source", "R", "medium_dof", "R2_cond"]
    with open(B.ROOT / "benchmark/domain_rankmatched_results.csv",
              "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    pcols = ["city", "contrast", "R", "delta_mean", "ci95_lo", "ci95_hi",
             "frac_dates_A_gt_B", "p_one_sided", "n_dates", "n_boot"]
    with open(B.ROOT / "benchmark/domain_rankmatched_pairwise.csv",
              "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=pcols, extrasaction="ignore")
        w.writeheader()
        for r in pairwise:
            w.writerow(r)
    with open(B.ROOT / "benchmark/scripts/_rankmatched.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "pairwise": pairwise,
                   "results": [{k: v for k, v in r.items() if k != "_rec"}
                               for r in results]}, fh, indent=1)
    print(f"wrote {len(results)} rows and {len(pairwise)} contrasts")


if __name__ == "__main__":
    main()
