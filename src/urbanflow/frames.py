"""The four scoring frames, one implementation.

They differ in exactly two decisions -- *what is centred* and *what the
denominator is* -- and in nothing else.  Writing them as four copies of the
same ridge call is how the published numbers drifted between rounds, so here
each difference is a named argument.

    P      centre TRAIN rows only; predict from raw TEST x; score raw TEST y
           denominator ||y||^2                 (the manuscript's 1-E)
    D      centre every row with TRAIN phase means; score the anomaly
           denominator ||z||^2                 (the conditional response)
    I      D's prediction + TRAIN phase mean, denominator ||y||^2
    CLIM   TRAIN phase mean alone, denominator ||y||^2

The identity `R_I - R_CLIM = (1 - R_CLIM) R_D` holds by construction for the
last three; `identity_error` is checked on every run and a failure is an
implementation error, never a result.

`P` is *not* `D` with a different denominator.  `r4lib.anomaly` centres only
the fit rows, so the TEST target stays raw; `m7lib.apply_means` centres all
rows, so the TEST target is the departure.  Collapsing the two is the single
easiest way to lose 0.2 of score on Porto without noticing.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M


# --------------------------------------------------------------------------
# centring
# --------------------------------------------------------------------------
def phase_means(Amat, ph, m_fit):
    """Per-phase mean over the FIT rows only."""
    return {int(p): Amat[m_fit & (ph == p)].mean(0)
            for p in np.unique(ph[m_fit])}


def centred_on_fit(Amat, m_fit, ph):
    """Centre the FIT rows; every other row is left raw.  (P frame.)"""
    out = np.array(Amat, np.float64, copy=True)
    for p, mu in phase_means(Amat, ph, m_fit).items():
        s = m_fit & (ph == p)
        out[s] -= mu
    return out


def centred_all(Amat, ph, means):
    """Centre every row with the supplied TRAIN means.  (D / I frames.)

    Every phase present in `Amat` must have a TRAIN mean; a phase observed
    only outside TRAIN cannot be centred, and silently leaving it raw would
    change the denominator.
    """
    out = np.array(Amat, np.float64, copy=True)
    for p in np.unique(ph):
        if int(p) not in means:
            raise KeyError(f"phase {int(p)} has no TRAIN mean")
        out[ph == p] -= means[int(p)]
    return out


def expand_means(means, ph, shape):
    """Materialise per-row TRAIN phase means."""
    bar = np.zeros(shape, np.float64)
    for p, mu in means.items():
        bar[ph == p] = mu
    return bar


# --------------------------------------------------------------------------
# the paper's frame (P)
# --------------------------------------------------------------------------
def eval_split(D_, m_tr, m_va, m_te, kappa_grid=M.KAPPA):
    """Held-out score of the manuscript: `1 - E(raw TEST y, prediction)`.

    Fitted on TRAIN departures only; κ chosen on VAL against the same raw
    target; TEST scored once.
    """
    X, Y, ph = D_["X"], D_["Y"], D_["ph"]
    Xa = centred_on_fit(X, m_tr, ph)
    Ya = centred_on_fit(Y, m_tr, ph)
    Xpred = np.vstack([Xa[m_va], Xa[m_te]])
    lams, smax = M.lams_from(Xa[m_tr], kappa_grid)
    preds, s = M.ridge_path(Xa[m_tr], Ya[m_tr], Xpred, lams)
    n_va = int(m_va.sum())
    grid = [dict(kappa=float(k), lam=float(lam),
                 E_va=M.E_alloc(Ya[m_va], pr[:n_va]),
                 df_eff=float((s / (s + lam)).sum()))
            for k, lam, pr in zip(kappa_grid, lams, preds)]
    best = min(grid, key=lambda g: g["E_va"])
    j = grid.index(best)
    pred_va, pred_te = preds[j][:n_va], preds[j][n_va:]
    E_te = M.E_alloc(Ya[m_te], pred_te)

    per, per_d = [], []
    pred_all = np.full_like(Ya, np.nan)
    pred_all[m_te] = pred_te
    for d in np.unique(D_["di"][m_te]):
        srow = m_te & (D_["di"] == d)
        per.append(M.E_alloc(Ya[srow], pred_all[srow]))
        per_d.append(int(D_["order"][d]))

    return dict(kappa_star=best["kappa"], lam=float(best["lam"]),
                E_va=best["E_va"], E_te=E_te, score_te=1.0 - E_te,
                df_eff=float(best["df_eff"]), smax=smax, grid=grid,
                per_date_te=per_d, per_date_score_te=[1.0 - e for e in per],
                _pred_te=pred_te)


# --------------------------------------------------------------------------
# the conditional frames (D / I / CLIM) and P, in one pass
# --------------------------------------------------------------------------
def run_frames(D_, m_tr, m_va, m_te, kappa_grid=M.KAPPA):
    """P, D, I and CLIM for one TRAIN/VAL/TEST split.

    One ridge path serves D, I and CLIM: they share the departure target and
    the same κ.  P is evaluated by `eval_split`, i.e. by the manuscript's own
    definition, and is reported alongside -- never subtracted from, never
    ranked against, D.
    """
    X, Y, ph = D_["X"], D_["Y"], D_["ph"]
    Xmu = phase_means(X, ph, m_tr)
    Ymu = phase_means(Y, ph, m_tr)
    Xa = centred_all(X, ph, Xmu)
    Ya = centred_all(Y, ph, Ymu)
    ybar = expand_means(Ymu, ph, Y.shape)

    lams, smax = M.lams_from(Xa[m_tr], kappa_grid)
    xpred = np.vstack([Xa[m_va], Xa[m_te]])
    preds, s = M.ridge_path(Xa[m_tr], Ya[m_tr], xpred, lams)
    nv = int(m_va.sum())
    evs = [M.E_alloc(Ya[m_va], pr[:nv]) for pr in preds]
    j = int(np.argmin(evs))
    pred_va, pred_te = preds[j][:nv], preds[j][nv:]
    kappa = float(kappa_grid[j])

    out = dict(kappa=kappa, lam=float(lams[j]),
               df_eff=float((s / (s + lams[j])).sum()), E_va=float(evs[j]),
               smax=smax, n_train=int(m_tr.sum()), n_val=nv,
               n_test=int(m_te.sum()), r=int(D_["r"]),
               dim_out=int(Y.shape[1]))

    out["R_D"] = 1.0 - M.E_alloc(Ya[m_te], pred_te)
    out["E_D"] = 1.0 - out["R_D"]
    out["E_CLIM"] = M.E_alloc(Y[m_te], ybar[m_te])
    out["R_CLIM"] = 1.0 - out["E_CLIM"]
    out["E_I"] = M.E_alloc(Y[m_te], pred_te + ybar[m_te])
    out["R_I"] = 1.0 - out["E_I"]
    out["dR_I"] = out["R_I"] - out["R_CLIM"]

    prim = eval_split(D_, m_tr, m_va, m_te, kappa_grid)
    out["R_P"] = prim["score_te"]
    out["kappa_P"] = prim["kappa_star"]
    out["E_P"] = prim["E_te"]

    out["identity"] = M.identity_error(out["R_D"], out["R_CLIM"], out["R_I"])
    out["identity_abs_err"] = out["identity"]["abs_err"]
    out["identity_lhs"] = out["identity"]["lhs"]
    out["identity_rhs"] = out["identity"]["rhs"]

    per = {}
    for d in np.unique(D_["di"][m_te]):
        srow = m_te & (D_["di"] == d)
        y, z = Y[srow], Ya[srow]
        pr = pred_te[(D_["di"][m_te] == d)]
        rec = dict(n_rows=int(srow.sum()),
                   R_D=1.0 - M.E_alloc(z, pr),
                   R_CLIM=1.0 - M.E_alloc(y, ybar[srow]),
                   R_I=1.0 - M.E_alloc(y, pr + ybar[srow]),
                   y_energy=float((y ** 2).sum()))
        rec["dR_I"] = rec["R_I"] - rec["R_CLIM"]
        per[int(D_["order"][d])] = rec
    out["per_date"] = per
    out["boot_dR_I"] = M.date_bootstrap([v["dR_I"] for v in per.values()])
    out["boot_R_D"] = M.date_bootstrap([v["R_D"] for v in per.values()])
    out["boot_R_I"] = M.date_bootstrap([v["R_I"] for v in per.values()])
    out["pooled_dR_I"] = out["dR_I"]
    return out
