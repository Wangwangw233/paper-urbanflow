from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from urbanflow_ref import fields, frames, metrics as M, Reference

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "threecity_redesign" / "results"
EXPECTED = {
    "road_spatial_transfer/results/road_basis_Chengdu.npz":
        "be082e6b1854aa2c498fb2c27b8c34910d761ad55f7071c11ff15a06def1d362",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = Reference.load()
    D_, sp, order = ref.design("Chengdu", "window", fold=0, keep_field=True)
    X, Y, ph = D_["X"], D_["Y"], D_["ph"]
    U = D_["U"]
    m_tr = np.isin(D_["di"], [order.index(d) for d in sp["train"]])
    m_va = np.isin(D_["di"], [order.index(d) for d in sp["val"]])
    m_te = np.isin(D_["di"], [order.index(d) for d in sp["test"]])
    n_te = int(m_te.sum())
    nv = int(m_va.sum())

    Xmu = frames.phase_means(X, ph, m_tr)
    Ymu = frames.phase_means(Y, ph, m_tr)
    Xa = frames.centred_all(X, ph, Xmu)
    Ya = frames.centred_all(Y, ph, Ymu)
    ybar = frames.expand_means(Ymu, ph, Y.shape)
    lams, smax = M.lams_from(Xa[m_tr], M.KAPPA)
    preds, s = M.ridge_path(Xa[m_tr], Ya[m_tr], np.vstack([Xa[m_va], Xa[m_te]]), lams)
    j = int(np.argmin([M.E_alloc(Ya[m_va], pr[:nv]) for pr in preds]))
    pred_d = preds[j][nv:]
    r_d = 1.0 - M.E_alloc(Ya[m_te], pred_d)
    assert abs(r_d - 0.1196004302) < 1e-8, r_d
    assert float(M.KAPPA[j]) == 0.1

    Xp = frames.centred_on_fit(X, m_tr, ph)
    Yp = frames.centred_on_fit(Y, m_tr, ph)
    lams_p, _ = M.lams_from(Xp[m_tr], M.KAPPA)
    preds_p, _ = M.ridge_path(Xp[m_tr], Yp[m_tr], np.vstack([Xp[m_va], Xp[m_te]]), lams_p)
    jp = int(np.argmin([M.E_alloc(Yp[m_va], pr[:nv]) for pr in preds_p]))
    pred_p = preds_p[jp][nv:]
    r_p = 1.0 - M.E_alloc(Yp[m_te], pred_p)
    assert abs(r_p - 0.6059232746) < 1e-8, r_p
    assert float(M.KAPPA[jp]) == 0.001

    obs = Y[m_te] @ U.T
    hat = pred_p @ U.T
    raw = D_["Fc"][m_te]
    obs18 = obs.reshape(3, 18, -1)
    hat18 = hat.reshape(3, 18, -1)
    raw18 = raw.reshape(3, 18, -1)
    test_dates = np.asarray(sp["test"], dtype=np.int64)
    date_errors = np.array([M.E_alloc(Yp[m_te & (D_["di"] == i)], pred_p[D_["di"][m_te] == i])
                            for i in np.unique(D_["di"][m_te])])
    picked_i = int(np.argmin(np.abs(date_errors - np.median(date_errors))))
    picked_phase = 6
    np.savez_compressed(
        OUT / "chengdu_snapshot18.npz",
        date18=obs18, hat18=hat18, raw18=raw18, proj18=obs18,
        test_dates=test_dates, picked_date=test_dates[picked_i],
        picked_phase=picked_phase, picked_i=picked_i,
        per_date_R_P=1.0 - date_errors,
    )

    obs_dft = np.fft.rfft(Y[m_te].reshape(3, 18, -1), axis=1, norm="ortho")
    pred_dft = np.fft.rfft(pred_p.reshape(3, 18, -1), axis=1, norm="ortho")
    a = obs_dft.mean(axis=0)
    b = pred_dft.mean(axis=0)
    inner = np.sum(a.conj() * b, axis=1)
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    cos = np.abs(inner) / np.maximum(an * bn, 1e-300)
    rho = bn / np.maximum(an, 1e-300)
    phase = np.angle(inner)

    fitted_all, _ = M.ridge_path(Xp[m_tr], Yp[m_tr], Xp, lams_p)
    fitted_train = fitted_all[jp][m_tr]
    train_obs_dft = np.fft.rfft(Y[m_tr].reshape(8, 18, -1), axis=1, norm="ortho")
    train_pred_dft = np.fft.rfft(fitted_train.reshape(8, 18, -1), axis=1, norm="ortho")
    at = train_obs_dft.mean(axis=0)
    bt = train_pred_dft.mean(axis=0)
    inner_t = np.sum(at.conj() * bt, axis=1)
    atn = np.linalg.norm(at, axis=1)
    btn = np.linalg.norm(bt, axis=1)
    train_cos = np.abs(inner_t) / np.maximum(atn * btn, 1e-300)
    train_rho = btn / np.maximum(atn, 1e-300)
    train_phase = np.angle(inner_t)
    m = np.arange(10, dtype=np.int64)
    np.savez_compressed(
        OUT / "chengdu_response18.npz",
        m=m, freq_cycles_per_observed_hour=m / 18.0,
        cycles_per_day=4.0 * m / 3.0,
        cos=cos, rho=rho, phase_rad=phase, a_norm=an, b_norm=bn,
        train_cos=train_cos, train_rho=train_rho, train_phase_rad=train_phase,
        test_dates=test_dates,
    )

    inputs = {
        "project10/outputs/p0_l0/chengdu_aligned_l250_dt60.npz",
        "project10/outputs/p0_l0/chengdu_datefreeze.json",
        "routing_urban_medium_constitutive/results/um_ctx.npz",
        "road_spatial_transfer/results/road_basis_Chengdu.npz",
    }
    record = {
        "status": "pass",
        "city": "Chengdu",
        "R_D": float(r_d),
        "R_P": float(r_p),
        "kappa_D": float(M.KAPPA[j]),
        "kappa_P": float(M.KAPPA[jp]),
        "test_dates": test_dates.tolist(),
        "snapshot_rule": "P-frame readout matching the Porto/San Francisco snapshots; TEST date with per-date P error closest to the median; 12:00 (phase 6 of the 06--23 h window)",
        "response": "P-selected operator; unitary 18-point DFT per TEST date; coherent TEST-date mean coefficients; all ten single-sided bins",
        "inputs_sha256": {rel: sha256(ROOT / "external_data" / rel) for rel in sorted(inputs)},
        "expected_basis_sha256": EXPECTED,
        "outputs": ["chengdu_snapshot18.npz", "chengdu_response18.npz"],
    }
    (OUT / "chengdu_frozen_response_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"R_D": r_d, "R_P": r_p,
                      "snapshot_date": int(test_dates[picked_i]),
                      "cos": cos.tolist(), "rho": rho.tolist()}, indent=2))


if __name__ == "__main__":
    main()
