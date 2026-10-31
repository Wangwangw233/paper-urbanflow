"""Estimators and scores.  The definitions are the paper's, verbatim.

    E(v, v̂) = Σ‖v - v̂‖² / Σ‖v‖²            held-out error
    score    = 1 - E                        ("R²" in the manuscript)

Ridge is relative: `λ = κ · s_max(X X^T)` on the FIT block, so the same κ grid
means the same thing at every rank and every city.  The grid is frozen; κ is
selected on VAL only.
"""
from __future__ import annotations

import numpy as np

#: the κ grid of `cc_lib` / `r4lib` / `m7lib`.  Pre-declared, never extended
#: after a held-out score is seen.
KAPPA = (1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2)

SEED = 20261028
NBOOT = 10000


def E_alloc(Vtrue, Vpred):
    """`Σ‖v - v̂‖² / Σ‖v‖²`."""
    num = float(((Vtrue - Vpred) ** 2).sum())
    den = float((Vtrue ** 2).sum())
    return num / den if den > 0 else float("nan")


def ridge_fit_predict(Xtr, Ttr, Xpred, lam):
    """Ridge `Xtr -> Ttr` evaluated on `Xpred`, dual form."""
    K = Xtr @ Xtr.T
    s, V = np.linalg.eigh(K)
    s = np.maximum(s, 0.0)
    a = V @ ((V.T @ Ttr) / (s + lam)[:, None])
    return (Xpred @ Xtr.T) @ a, dict(df_eff=float((s / (s + lam)).sum()),
                                     lam=float(lam))


def ridge_path(Xtr, Ttr, Xpred, lams):
    """All κ in one eigendecomposition.  Same arithmetic as
    `ridge_fit_predict`: `pred(λ) = (Xpred Xtrᵀ V)((VᵀT)/(s+λ))`."""
    K = Xtr @ Xtr.T
    s, V = np.linalg.eigh(K)
    s = np.maximum(s, 0.0)
    P = V.T @ Ttr
    base = (Xpred @ Xtr.T) @ V
    return [base @ (P / (s + lam)[:, None]) for lam in lams], s


def lams_from(Xtr, kappa_grid=KAPPA):
    """`(lams, smax)` for the relative ridge grid."""
    smax = float(np.maximum(np.linalg.eigvalsh(Xtr @ Xtr.T), 0.0).max())
    return [float(k) * smax for k in kappa_grid], smax


def date_bootstrap(values, n_boot=NBOOT, seed=SEED):
    """Date-level bootstrap of the mean.  The unit is an independent TEST
    date, not a row: rows inside a day are not independent draws."""
    d = np.asarray([x for x in values if np.isfinite(x)], np.float64)
    if d.size == 0:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    means = d[idx].mean(1)
    return dict(n=int(d.size), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                ci95=[float(np.quantile(means, 0.025)),
                      float(np.quantile(means, 0.975))],
                values=[float(x) for x in d])


def identity_error(R_D, R_CLIM, R_I):
    """`R_I - R_CLIM = (1 - R_CLIM) R_D`.  A failure here is an implementation
    error, not a finding."""
    lhs = R_I - R_CLIM
    rhs = (1.0 - R_CLIM) * R_D
    return dict(lhs=float(lhs), rhs=float(rhs), abs_err=float(abs(lhs - rhs)))
