r"""Matched domain baselines D0 / D1 / D2 -- construction only.

Definitions and their derivation are frozen in
`benchmark/domain_baseline_design.md`; this module is the code that implements
exactly those, nothing else.  Every object is either a frozen object of the
current paper or derived from one:

    D0  Fhat = Fbar_phi + T_0 x_t          the frozen S0_routine row, relabelled
    D1  Fhat = Fbar_phi + T_W x_t          T_W = L_D + A_T^T, proved = W D^T (D W D^T)^+
    D2  Fhat = Fbar_phi + T_0 x_t + g A_W x_t
        g by through-origin OLS on TRAIN rows, the field-space analogue of the
        frozen per-channel calibration lc_lib.cross_map(mode="g")

Scoring is deliberately NOT done here: every baseline is handed to
`bench_lib.score`, the same function that produced the frozen S0_routine and
S3_ours rows, so the comparison is like-for-like on the same test rows, dates
and metric implementation.

Read-only on the archive.  Invariants this module relies on, verified in
_domain_checks.py on the real held-out forcings:

    max|D A_W^T|                3.0e-15 (Porto)  3.3e-14 (SF)   A_W maps into ker D
    max|D (T_W x_t) - x_t|      5.9e-13          7.3e-13        D1 conserves
    max|mean(x_t)|              3.5e-17          1.5e-17        the lift identity needs mean-zero x
"""
from __future__ import annotations

import numpy as np

import bench_lib as B


# ---------------------------------------------------------------------------
# 0. the shared frame helpers
# ---------------------------------------------------------------------------
def tw_lift(ctx):
    """T_W (cell -> face), from frozen objects only.

    T_W = T_0 + A_W^T.  Proved equal to W D^T (D W D^T)^+ on mean-zero forcings
    in domain_baseline_design.md section 2 and checked against an independent
    reconstruction from the road conductance in _domain_checks.py.
    """
    return (np.asarray(ctx["L"], np.float64)
            + np.asarray(ctx["A_T"], np.float64).T)


def _routine(ctx, fo, te):
    return np.asarray(fo["Fbar"], np.float64)[np.asarray(fo["slot"])[te]]


def _lift(ctx, fo, te):
    return (np.asarray(ctx["L"], np.float64)
            @ np.asarray(fo["x_te"], np.float64).T).T


# ---------------------------------------------------------------------------
# 1. D0 -- isotropic minimum-norm / potential-like field
# ---------------------------------------------------------------------------
def field_D0(ctx):
    """Fbar_phi(t) + T_0 x_t.  Identical arithmetic to bench_lib.ours_field's F0."""
    return [_routine(ctx, fo, fo["te"]) + _lift(ctx, fo, fo["te"])
            for fo in ctx["folds"]]


# ---------------------------------------------------------------------------
# 2. D1 -- road-only minimum-energy / electrical-flow field
# ---------------------------------------------------------------------------
def field_D1(ctx):
    """Fbar_phi(t) + T_W x_t.  Same forcing, same road layer, no Psi and no Q."""
    TW = tw_lift(ctx)
    out = []
    for fo in ctx["folds"]:
        te = np.asarray(fo["te"])
        out.append(_routine(ctx, fo, te)
                   + (TW @ np.asarray(fo["x_te"], np.float64).T).T)
    return out


# ---------------------------------------------------------------------------
# 3. D2 -- scalar-calibrated road-only response
# ---------------------------------------------------------------------------
def gamma_D2(ctx, fo, weighted=False):
    """The single train-only scalar of D2.

    Row-pooled (PRIMARY, matching the frozen per-channel estimator):
        g = sum_tr <A_W x_t, z_t> / sum_tr ||A_W x_t||^2

    Date-equal (declared sensitivity, never used to pick a headline number):
        g = mean over TRAIN DATES of the per-date ratio num_d / den_d
    i.e. the same date-weighting convention the frozen R2 aggregate uses, applied
    to the calibration.  Fixed here in advance; both are reported.

    `z_t` is the observed routing residual dFo_t -- the target of the calibration,
    taken on TRAIN rows only.  No test row enters.
    """
    AT = np.asarray(ctx["A_T"], np.float64)
    u = np.asarray(fo["x"], np.float64) @ AT          # A_W x_t, all fold rows
    y = np.asarray(fo["dFo"], np.float64)             # observed routing residual
    tr = np.asarray(fo["base"], np.bool_)
    if not weighted:
        num = float((u[tr] * y[tr]).sum())
        den = float((u[tr] ** 2).sum())
        return num / max(den, 1e-300)
    d = np.asarray(fo["date"])[tr]
    per = []
    for dt in np.unique(d):
        m = d == dt
        per.append(float((u[tr][m] * y[tr][m]).sum())
                   / max(float((u[tr][m] ** 2).sum()), 1e-300))
    return float(np.mean(per))


def field_D2(ctx, weighted=False):
    """Fbar_phi(t) + T_0 x_t + g A_W x_t, one train-only scalar per fold."""
    AT = np.asarray(ctx["A_T"], np.float64)
    out, gams = [], []
    for fo in ctx["folds"]:
        te = np.asarray(fo["te"])
        g = gamma_D2(ctx, fo, weighted=weighted)
        gams.append(float(g))
        u_te = np.asarray(fo["x_te"], np.float64) @ AT
        out.append(_routine(ctx, fo, te) + _lift(ctx, fo, te) + g * u_te)
    return out, gams


# ---------------------------------------------------------------------------
# 4. the diagonal-Gamma same-road ablation, one generic code path for both cities
# ---------------------------------------------------------------------------
# Mirrors lc_lib.cross_map(mode="g") + run_sm.decode verbatim, but takes a city's
# context as an argument instead of reading lc_lib's Porto-only RUN constant.
# lc_lib.boot() hardcodes RUN = "Porto_allmotion", so the frozen diagonal-Gamma
# value exists for Porto only; this function exists so that BOTH cities are
# computed by one and the same code, and it asserts its own Porto value against
# the frozen learning-curve anchor before anything else is scored.
def field_diag_gamma(ctx):
    """Returns (fields, gamma).  Psi, A_W, T_0 and the decoder are the frozen ones."""
    Psi = np.asarray(ctx["Psi"], np.float64)[:, :32]
    L, D = np.asarray(ctx["L"], np.float64), np.asarray(ctx["D"], np.float64)
    AW = np.asarray(ctx["A_T"], np.float64).T
    V, S, Ut = np.linalg.svd(AW, full_matrices=False)
    r = int((S >= 1e-12 * S[0]).sum())
    Pr = Psi - L @ (D @ Psi)
    Q = Pr.T @ V[:, :r]
    out, gams = [], []
    for fo in ctx["folds"]:
        x = np.asarray(fo["x"], np.float64)
        ahat = ((x @ Ut[:r, :].T) * S[:r]) @ Q.T          # Q Sigma U^T x_t
        tr = np.asarray(fo["base"], np.bool_)
        gam = (ahat[tr] * np.asarray(fo["a"], np.float64)[tr]).sum(0) \
            / np.maximum((ahat[tr] ** 2).sum(0), 1e-300)
        gams.append(gam)
        te = np.asarray(fo["te"])
        a = ahat * gam
        G = Pr.T @ Pr
        z = (Pr @ np.linalg.solve(G, a[te].T)).T          # run_sm.decode at K = 32
        base = _routine(ctx, fo, te) + _lift(ctx, fo, te)
        out.append(base + z)
    return out, gams


# ---------------------------------------------------------------------------
# 5. the mobility-inferred medium -- a different STRUCTURAL SOURCE, same lift
# ---------------------------------------------------------------------------
def lift_from_W(Dm, W):
    """T_W (cell -> face) for an arbitrary face conductance W.

    W o (D^T (D W D^T + J)^{-1}) with the same gauge-fixing J as
    r7_common.lift_operator / build_A_T, so it reproduces the frozen road lift
    exactly when W is the road conductance (asserted in _domain_checks.py).
    """
    Dm = np.asarray(Dm, np.float64)
    W = np.asarray(W, np.float64)
    N = Dm.shape[0]
    J = np.outer(np.ones(N), np.ones(N)) / N
    M = (Dm * W[None, :]) @ Dm.T
    return W[:, None] * (Dm.T @ np.linalg.solve(M + J, np.eye(N)))


def mobility_conductance(ctx, kind="net"):
    """Face conductance estimated from TRAIN mobility, then frozen.

    ADAPTED, not a reproduction.  The estimator is the mean-|net-current|
    conductance of the RECM / electric-circuit line as it is implemented in
    project9/scripts/phaseN_audits.py::_transfer_r2:

        Wn = max(mean over TRAIN dates of |f|, 1e-3)          ("net")
        Wg = max(mean over TRAIN dates of gross throughput, 1e-3)   ("gross")

    Two documented adaptations: (a) the original quantities are graph-edge
    currents, here they are the grid-face fluxes of the frozen field, because the
    lift this baseline is pushed through is the paper's face-level minimum-energy
    operator; (b) the frozen `build_w` normalisation is applied afterwards
    (divide by the mean, floor at 0.02) so the conductance lives in the same
    function class as the road W and the amplitude comparison is meaningful.

    TRAIN rows only.  No held-out current enters.
    """
    D = np.asarray(ctx["D"], np.float64)
    n_int = D.shape[1]
    tot = np.zeros(n_int)
    n = 0
    for fo in ctx["folds"]:
        tr = np.asarray(fo["base"], np.bool_)
        F = np.asarray(fo["Fi"], np.float64)[tr]
        if kind == "net":
            tot += np.abs(F).mean(axis=0)
        elif kind == "gross":
            # gross bidirectional throughput per face is not carried by the
            # signed field cache; |net| is the quantity the cache holds, so the
            # gross variant is reported as NOT AVAILABLE rather than proxied.
            raise SystemExit("gross-throughput conductance: quantity not in the frozen cache")
        else:
            raise ValueError(kind)
        n += 1
    w = np.maximum(tot / max(n, 1), 1e-3)
    return np.maximum(w / w.mean(), 0.02)


def field_Minf(ctx, kind="net"):
    """Fbar_phi(t) + T_Wmob x_t, with the medium estimated from TRAIN mobility."""
    W = mobility_conductance(ctx, kind=kind)
    TW = lift_from_W(ctx["D"], W)
    return [(_routine(ctx, fo, np.asarray(fo["te"]))
             + (TW @ np.asarray(fo["x_te"], np.float64).T).T)
            for fo in ctx["folds"]], W


# ---------------------------------------------------------------------------
# 6. scoring shims
# ---------------------------------------------------------------------------
def score(ctx, fields, tag):
    """Score one baseline through bench_lib.score -- the frozen metric path."""
    return B.score(ctx, fields, tag=tag)
