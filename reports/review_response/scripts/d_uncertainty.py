from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("URBANFLOW_ROOT", r".//external_data")

from urbanflow_ref import Reference, frames, metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "review_response" / "results"
OUT.mkdir(parents=True, exist_ok=True)
CITIES = {"Porto": ("frozen", [0], [0.468]),
          "SF": ("frozen", [0, 1, 2, 3, 4], [0.240]),
          "Chengdu": ("window", [0], [0.120])}
N_BOOT = 10000
N_NULL = 199
SEED = 20261003


def fit_curves(design, split, kappas=(1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    X, Y, ph, di, order = (design["X"], design["Y"], design["ph"],
                           design["di"], design["order"])
    pos = {int(d): i for i, d in enumerate(order)}
    m_tr = np.isin(di, [pos[int(d)] for d in split["train"]])
    m_va = np.isin(di, [pos[int(d)] for d in split["val"]])
    m_te = np.isin(di, [pos[int(d)] for d in split["test"]])
    Xmu = frames.phase_means(X, ph, m_tr)
    Xa = frames.centred_all(X, ph, Xmu)
    Xtr = Xa[m_tr]
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    keep = s > 1e-10 * s[0]
    U, s, Vt = U[:, keep], s[keep], Vt[keep]
    A_va = Xa[m_va] @ Vt.T
    A_te = Xa[m_te] @ Vt.T
    return dict(Xa=Xa, Y=Y, ph=ph, di=di, m_tr=m_tr, m_va=m_va, m_te=m_te,
                U=U, s=s, Vt=Vt, A_va=A_va, A_te=A_te, order=order,
                pos={int(d): i for i, d in enumerate(order)},
                n_tr=int(m_tr.sum()), n_va=int(m_va.sum()), n_te=int(m_te.sum()))


def d_score(cur, Ytarget, kappas=(1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    Ymu = frames.phase_means(Ytarget, cur["ph"], cur["m_tr"])
    Ya = frames.centred_all(Ytarget, cur["ph"], Ymu)
    B = cur["U"].T @ Ya[cur["m_tr"]]
    smax2 = float(cur["s"][0] ** 2)
    best = None
    for kappa in kappas:
        lam = kappa * smax2
        K = (cur["s"] / (cur["s"] ** 2 + lam))[:, None] * B
        pv = cur["A_va"] @ K
        ev = M.E_alloc(Ya[cur["m_va"]], pv)
        if best is None or ev < best[0]:
            best = (ev, kappa, K)
    _, kappa, K = best
    pt = cur["A_te"] @ K
    num = ((Ya[cur["m_te"]] - pt) ** 2).sum(axis=1)
    den = (Ya[cur["m_te"]] ** 2).sum(axis=1)
    row_dates = cur["di"][cur["m_te"]]
    udates = np.unique(row_dates)
    num_d = np.asarray([num[row_dates == d].sum() for d in udates])
    den_d = np.asarray([den[row_dates == d].sum() for d in udates])
    return dict(kappa=float(kappa), R_D=1.0 - float(num.sum() / den.sum()),
                num=num_d, den=den_d, dates=udates,
                predictions=pt, targets=Ya[cur["m_te"]])


def permute_split_rows(di, dates, rng, pos):
    idx_dates = [int(pos[int(d)]) for d in dates]
    groups = {d: np.flatnonzero(di == d) for d in idx_dates}
    order = rng.permutation(len(idx_dates))
    mapping = {idx_dates[i]: idx_dates[order[i]] for i in range(len(idx_dates))}
    rows = np.concatenate([groups[d] for d in idx_dates])
    idx = np.empty(rows.size, dtype=np.int64)
    location = {int(r): i for i, r in enumerate(rows)}
    for d in idx_dates:
        target = groups[mapping[d]]
        if len(target) != len(groups[d]):
            raise RuntimeError("unequal phase blocks")
        for src, dst in zip(groups[d], target):
            idx[location[int(src)]] = dst
    return rows, idx


def permuted_targets(Y, cur, split, rng):
    Yp = Y.copy()
    for dates in (split["train"], split["val"], split["test"]):
        rows, idx = permute_split_rows(cur["di"], dates, rng, cur["pos"])
        Yp[rows] = Y[idx]
    return Yp


def main():
    ref = Reference.load()
    out = {}
    for city, (which, folds, expected) in CITIES.items():
        fold_records = []
        for fold in folds:
            design, split, order = ref.design(city, which, fold, keep_field=False)
            cur = fit_curves(design, split)
            real = d_score(cur, cur["Y"])
            fold_records.append(dict(fold=fold, split=split, cur=cur, real=real,
                                     n_test_dates=len(set(real["dates"].tolist()))))
        # city-level pooled real score
        num = sum(r["real"]["num"].sum() for r in fold_records)
        den = sum(r["real"]["den"].sum() for r in fold_records)
        real_city = 1.0 - float(num / den)
        # paired-date bootstrap: resample dates within fold, then pool
        rng = np.random.default_rng([SEED, list(CITIES).index(city)])
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            bn = bd = 0.0
            for rec in fold_records:
                n = len(rec["real"]["dates"])
                take = rng.integers(0, n, n)
                bn += rec["real"]["num"][take].sum()
                bd += rec["real"]["den"][take].sum()
            boot[b] = 1.0 - bn / bd
        ci = np.percentile(boot, [2.5, 97.5]).tolist()
        # pairing null: permute date blocks within each split, refit, re-evaluate
        null = np.empty(N_NULL)
        for it in range(N_NULL):
            nn = nd = 0.0
            for rec in fold_records:
                Yp = permuted_targets(rec["cur"]["Y"], rec["cur"], rec["split"], rng)
                res = d_score(rec["cur"], Yp)
                nn += res["num"].sum()
                nd += res["den"].sum()
            null[it] = 1.0 - nn / nd
        p = float((1 + np.sum(null >= real_city)) / (1 + N_NULL))
        total_test_dates = len(set(
            int(d) for rec in fold_records for d in rec["real"]["dates"].tolist()))
        inferential = total_test_dates >= 10
        out[city] = dict(
            which=which, folds=folds, real_D=real_city,
            expected_D=expected, n_test_dates=[r["n_test_dates"] for r in fold_records],
            total_test_dates=total_test_dates,
            bootstrap_ci95=ci, bootstrap_n=N_BOOT,
            null_n=N_NULL, null_median=float(np.median(null)),
            null_q95=float(np.percentile(null, 95)),
            null_max=float(null.max()),
            pairing_p=(p if inferential else None),
            pairing_p_note=(
                "descriptive only: three TEST dates allow six date permutations"
                if not inferential else None),
            per_fold=[dict(fold=r["fold"], R_D=r["real"]["R_D"],
                           kappa=r["real"]["kappa"],
                           n_test_dates=r["n_test_dates"]) for r in fold_records],
        )
        print(city, out[city]["real_D"], "CI", ci, "null med",
              out[city]["null_median"], "p", p)
    (OUT / "d_uncertainty.json").write_text(json.dumps(dict(
        seed=SEED,
        bootstrap="dates resampled with replacement within each fold, then pooled",
        null="date blocks permuted within TRAIN, VAL and TEST separately; D ridge refitted and re-selected in every draw",
        cities=out), indent=1), encoding="utf-8")
    print("wrote", OUT / "d_uncertainty.json")


if __name__ == "__main__":
    main()
