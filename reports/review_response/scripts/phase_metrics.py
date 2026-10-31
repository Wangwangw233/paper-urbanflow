from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("URBANFLOW_ROOT", r"legacy/")

from urbanflow_ref import Reference, frames, metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "review_response" / "results"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = {
    "Porto": dict(which="frozen", folds=[0]),
    "SF": dict(which="frozen", folds=[0, 1, 2, 3, 4]),
    "Chengdu": dict(which="window", folds=[0]),
}


def unitary_dft(field: np.ndarray) -> np.ndarray:
    """One-sided unitary DFT over the phase axis with correct interior weights."""
    n = field.shape[1]
    coeff = np.fft.rfft(field, axis=1, norm="ortho")
    if n % 2 == 0:
        coeff[:, 1:n // 2, :] *= np.sqrt(2.0)
    else:
        coeff[:, 1:(n + 1) // 2, :] *= np.sqrt(2.0)
    return coeff


def fit_p(design, train_dates, val_dates, test_dates):
    X, Y, ph, di, order = (design["X"], design["Y"], design["ph"],
                           design["di"], design["order"])
    pos = {int(d): i for i, d in enumerate(order)}
    m_tr = np.isin(di, [pos[int(d)] for d in train_dates])
    m_va = np.isin(di, [pos[int(d)] for d in val_dates])
    m_te = np.isin(di, [pos[int(d)] for d in test_dates])
    Xa = frames.centred_on_fit(X, m_tr, ph)
    Ya = frames.centred_on_fit(Y, m_tr, ph)
    lams, _ = M.lams_from(Xa[m_tr], M.KAPPA)
    nv = int(m_va.sum())
    preds, _ = M.ridge_path(Xa[m_tr], Ya[m_tr],
                            np.vstack([Xa[m_va], Xa[m_te]]), lams)
    vals = [M.E_alloc(Ya[m_va], p[:nv]) for p in preds]
    j = int(np.argmin(vals))
    pred = preds[j][nv:]
    r_test = 1.0 - float(M.E_alloc(Ya[m_te], pred))
    return (pred, Y[m_te], m_te, di[m_te], float(M.KAPPA[j]),
            1.0 - float(vals[j]), r_test, test_dates)


def main():
    ref = Reference.load()
    records = {}
    for city, spec in CITIES.items():
        D_, sp, order = ref.design(city, spec["which"], 0, keep_field=False)
        n_phase = len(sp["hours"]) if sp.get("hours") else 24
        n_bins = n_phase // 2 + 1
        per_fold = []
        for fold in spec["folds"]:
            if fold == 0:
                split = sp
                design = D_
            else:
                design, split, order = ref.design(city, spec["which"], fold,
                                                  keep_field=False)
            pred, obs, m_te, date_idx, kappa, r_val, r_test, dates = fit_p(
                design, split["train"], split["val"], split["test"])
            A = unitary_dft(obs.reshape(len(split["test"]), n_phase, -1))
            B = unitary_dft(pred.reshape(len(split["test"]), n_phase, -1))
            a = A  # (n_dates, n_bins, r)
            b = B
            num = (np.abs(a - b) ** 2).sum(axis=(0, 2))
            den = (np.abs(a) ** 2).sum(axis=(0, 2))
            r2_bin = 1.0 - num / np.maximum(den, 1e-300)
            inner = np.real((a * np.conj(b)).sum(axis=2))
            inner_abs = np.abs((a * np.conj(b)).sum(axis=2))
            an = np.linalg.norm(a, axis=2)
            bn = np.linalg.norm(b, axis=2)
            signed = inner / np.maximum(an * bn, 1e-300)
            abs_align = inner_abs / np.maximum(an * bn, 1e-300)
            signed_pooled = inner.sum(axis=0) / np.maximum(
                np.sqrt((an ** 2).sum(axis=0) * (bn ** 2).sum(axis=0)), 1e-300)
            date_r2 = 1.0 - (np.abs(a - b) ** 2).sum(axis=2) / np.maximum(
                (np.abs(a) ** 2).sum(axis=2), 1e-300)
            weights = den / den.sum()
            pooled = float((weights * r2_bin).sum())
            assert abs(pooled - r_test) < 1e-9, (city, fold, pooled, r_test)
            per_fold.append(dict(
                fold=fold, kappa=kappa, R_P=r_test,
                R_val=r_val, pooled_from_bins=pooled,
                n_phase=n_phase, n_bins=n_bins, n_dates=len(dates),
                test_dates=[int(x) for x in dates],
                bin_R2=r2_bin.tolist(),
                bin_signed_pooled=signed_pooled.tolist(),
                bin_signed_mean=signed.mean(axis=0).tolist(),
                bin_abs_mean=abs_align.mean(axis=0).tolist(),
                bin_abs_median=np.median(abs_align, axis=0).tolist(),
                bin_signed_median=np.median(signed, axis=0).tolist(),
                bin_signed_q25=np.percentile(signed, 25, axis=0).tolist(),
                bin_signed_q75=np.percentile(signed, 75, axis=0).tolist(),
                bin_energy_share=weights.tolist(),
                date_R2=date_r2.tolist(),
                date_signed=signed.tolist(),
            ))
        records[city] = dict(which=spec["which"], n_phase=n_phase,
                             n_bins=n_bins, folds=per_fold)
        print(city, "folds", len(per_fold), "R_P",
              [round(f["R_P"], 6) for f in per_fold])
        print("  bin R2", [round(x, 4) for x in per_fold[0]["bin_R2"]])
        print("  signed", [round(x, 4) for x in per_fold[0]["bin_signed_mean"]])

    inputs = {}
    for rel in ["urbanflow_ref/manifest.json"]:
        p = ROOT / rel
        inputs[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "phase_metrics.json").write_text(json.dumps(
        dict(definition=dict(
            bin_R2="1 - sum_d ||a_dm-b_dm||^2 / sum_d ||a_dm||^2",
            signed="mean_d Re<a_dm,b_dm>/(||a_dm|| ||b_dm||)",
            dft="unitary one-sided DFT over observed phases; interior bins sqrt(2)",
            source="frozen P-frame held-out predictions; road-coordinate target"),
            inputs=inputs, cities=records), indent=1), encoding="utf-8")
    print("wrote", OUT / "phase_metrics.json")


if __name__ == "__main__":
    main()
