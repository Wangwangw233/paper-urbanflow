from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("URBANFLOW_ROOT", str(ROOT / "external_data"))
sys.path.insert(0, str(ROOT / "src"))

from urbanflow import Reference, frames, metrics as M, geometry as G  # noqa: E402

OUT = ROOT / "results" / "frozen" / "anisotropy"
OUT.mkdir(parents=True, exist_ok=True)
CITIES = {"Porto": ("frozen", 0, 0.01), "SF": ("frozen", 0, 0.1),
          "Chengdu": ("window", 0, 0.1)}
N_NULL = 999
SEED = 20261031


def tensor_stats(S):
    S = np.asarray(S, float)
    tr = S[:, 0, 0] + S[:, 1, 1]
    d = np.sqrt(np.maximum((S[:, 0, 0] - S[:, 1, 1]) ** 2 + 4 * S[:, 0, 1] ** 2, 0))
    A = np.where(tr > 0, d / np.maximum(tr, 1e-300), 0.0)
    theta = 0.5 * np.arctan2(2 * S[:, 0, 1], S[:, 0, 0] - S[:, 1, 1])
    return A, theta, tr


def tens_from_coeff(Cx, Cy, w2):
    Cxw = Cx * np.sqrt(w2)[None, :]
    Cyw = Cy * np.sqrt(w2)[None, :]
    S = np.empty((Cx.shape[0], 2, 2), float)
    S[:, 0, 0] = (Cxw * Cxw).sum(1)
    S[:, 1, 1] = (Cyw * Cyw).sum(1)
    S[:, 0, 1] = S[:, 1, 0] = (Cxw * Cyw).sum(1)
    return S


def haar(rng, n):
    Q, R = np.linalg.qr(rng.normal(size=(n, n)))
    return Q * np.sign(np.diag(R))[None, :]


def axis_agreement(Ac, Au, tc, tu):
    w = Ac * Au
    ok = w > 0
    return float((w[ok] * np.cos(2 * (tc[ok] - tu[ok]))).sum() / w[ok].sum())


def lift_rows(U, nx, ny, p0, p1, n_int):
    is_x = (p1 - p0) == 1
    is_y = (p1 - p0) == nx
    if not np.all(is_x | is_y):
        raise RuntimeError("unexpected face orientation")
    nrm = np.zeros((n_int, 2))
    nrm[is_x, 0] = 1.0
    nrm[is_y, 1] = 1.0
    NC = nx * ny
    f = np.arange(n_int)
    D = sp.csr_matrix((np.concatenate([np.ones(n_int), -np.ones(n_int)]),
                       (np.concatenate([p0, p1]), np.concatenate([f, f]))),
                      shape=(NC, n_int))
    Iu = abs(D)
    Wx = np.asarray(Iu @ (nrm[:, 0:1] * U), float)
    Wy = np.asarray(Iu @ (nrm[:, 1:2] * U), float)
    cnt_x = np.asarray(Iu @ is_x.astype(float)).ravel()
    cnt_y = np.asarray(Iu @ is_y.astype(float)).ravel()
    return Wx, Wy, cnt_x, cnt_y


def main():
    ref = Reference.load()
    out = {}
    for city, (which, fold, kappa) in CITIES.items():
        D_, sp, order = ref.design(city, which, fold, keep_field=False)
        X, Y, ph, di = D_["X"], D_["Y"], D_["ph"], D_["di"]
        U = D_["U"]
        pos = {int(d): i for i, d in enumerate(order)}
        m_tr = np.isin(di, [pos[int(d)] for d in sp["train"]])
        Xmu = frames.phase_means(X, ph, m_tr)
        Ymu = frames.phase_means(Y, ph, m_tr)
        Xa = frames.centred_all(X, ph, Xmu)
        Ya = frames.centred_all(Y, ph, Ymu)
        Xtr, Ytr = Xa[m_tr], Ya[m_tr]
        lam = kappa * float(np.linalg.svd(Xtr, compute_uv=False)[0] ** 2)
        B = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
        K = np.ascontiguousarray(B.T, float)  # (r, NC)
        Q, sv, Vt = np.linalg.svd(K, full_matrices=False)
        keep = sv > 1e-10 * sv[0]
        Q, sv, Vt = Q[:, keep], sv[keep], Vt[keep]
        rnk = int(keep.sum())

        c = G.ctx(city)
        Wx, Wy, cnt_x, cnt_y = lift_rows(U, c["nx"], c["ny"], c["p0"], c["p1"],
                                        c["n_int"])
        cells = np.flatnonzero((cnt_x == 2) & (cnt_y == 2))
        Wxc, Wyc = Wx[cells], Wy[cells]
        WxQ, WyQ = Wxc @ Q, Wyc @ Q
        SC = tens_from_coeff(WxQ, WyQ, sv ** 2)
        SU = tens_from_coeff(Wxc, Wyc, np.ones(Wxc.shape[1]))
        A_C, th_C, _ = tensor_stats(SC)
        A_U, th_U, _ = tensor_stats(SU)
        G_real = axis_agreement(A_C, A_U, th_C, th_U)

        # source-covariance weighted response tensor
        Z = K @ Xtr.T  # (r, n_train)
        WxZ, WyZ = Wxc @ Z, Wyc @ Z
        Sresp = np.empty((cells.size, 2, 2))
        w = 1.0 / Z.shape[1]
        Sresp[:, 0, 0] = w * (WxZ * WxZ).sum(1)
        Sresp[:, 1, 1] = w * (WyZ * WyZ).sum(1)
        Sresp[:, 0, 1] = Sresp[:, 1, 0] = w * (WxZ * WyZ).sum(1)
        A_R, th_R, _ = tensor_stats(Sresp)

        rng = np.random.default_rng([SEED, 1 + list(CITIES).index(city)])
        G_null = np.empty(N_NULL)
        A_med_null = np.empty(N_NULL)
        for it in range(N_NULL):
            Qr = haar(rng, rnk)
            S = tens_from_coeff(WxQ @ Qr, WyQ @ Qr, sv ** 2)
            A, th, _ = tensor_stats(S)
            G_null[it] = axis_agreement(A, A_U, th, th_U)
            A_med_null[it] = float(np.median(A))
        p = float((1 + np.sum(G_null >= G_real)) / (1 + N_NULL))
        out[city] = dict(
            kappa=kappa, rank=rnk, n_cells=int(cells.size),
            G_theta=G_real, G_null_median=float(np.median(G_null)),
            G_null_q01=float(np.percentile(G_null, 1)),
            G_null_q99=float(np.percentile(G_null, 99)),
            empirical_p=p,
            median_A_C=float(np.median(A_C)), median_A_U=float(np.median(A_U)),
            median_A_response=float(np.median(A_R)),
            A_response_q25=float(np.percentile(A_R, 25)),
            A_response_q75=float(np.percentile(A_R, 75)),
            median_A_null=float(np.median(A_med_null)),
            null_q01=float(np.percentile(A_med_null, 1)),
            null_q99=float(np.percentile(A_med_null, 99)),
            G_null=G_null.tolist(),
        )
        print(city, {k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in out[city].items() if k != "G_null"})

    inputs = {}
    for rel in ["src/urbanflow/manifest.json", "external_data/road_spatial_transfer/results/road_basis_Chengdu.npz"]:
        pth = ROOT / rel
        if pth.is_file():
            inputs[rel] = hashlib.sha256(pth.read_bytes()).hexdigest()
    (OUT / "anisotropy_null.json").write_text(json.dumps(dict(
        definition=dict(
            unit_metric="Sigma_C=L_c C_R C_R^T L_c^T",
            source_metric="Sigma_response=L_c C_R Sigma_xx C_R^T L_c^T with TRAIN source covariance",
            null="spectrum-matched K_0=Q_0 Lambda V^T, Q_0 Haar in ROAD output space",
            G_theta="A_C A_U-weighted cos2(theta_C-theta_U)"),
        inputs=inputs, cities=out), indent=1), encoding="utf-8")
    print("wrote", OUT / "anisotropy_null.json")


if __name__ == "__main__":
    main()
