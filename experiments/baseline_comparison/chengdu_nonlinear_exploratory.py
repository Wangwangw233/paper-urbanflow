r"""Chengdu 18-phase, road-free, full-FACE exploratory A -> D experiment.

Read-only inputs:
  * project10/outputs/p0_l0/chengdu_aligned_l250_dt60.npz
  * routing_urban_medium_constitutive/results/um_ctx.npz
  * src/urbanflow geometry/window definitions

Estimator provenance (no road basis is used):
  * A: empirical_transfer_round/scripts/et_A.py and et_lib.py
  * D: legacy/empirical_transfer_round/scripts/et_D.py
    (legacy read-only source; its archived D JSON is byte-identical to the
    current-tree archived JSON)

Only this file's parent report directory is written.  All primary A/D scores
follow the reference definitions.  `full_face` blocks are an explicitly separate
orthogonal-completion metric: out-of-POD routing energy is retained in both the
numerator and denominator, and is never used to select a hyperparameter.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import pathlib
import platform
import sys
import time

import numpy as np
import scipy
import threadpoolctl

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2]
EXTERNAL = pathlib.Path(os.environ.get("URBANFLOW_ROOT", ROOT / "external_data"))
RESULTS = ROOT / "results" / "frozen" / "baselines" / "chengdu_nonlinear"
LOGS = ROOT / "results" / "logs" / "chengdu_nonlinear"
for _d in (RESULTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "run.log", mode="w", encoding="utf-8")],
)
LOG = logging.getLogger("chengdu-exploratory")

CFG = json.loads((ROOT / "configs" / "experiments" / "chengdu_nonlinear.json").read_text(encoding="utf-8"))
NPHASE = int(CFG["window"]["n_phase"])
NHARM = int(CFG["window"]["n_harm"])
W_H = np.asarray(CFG["window"]["one_sided_weights"], np.float64)
KAPPA = np.asarray(CFG["kappa_grid"], np.float64)
MU_GRID = np.asarray(CFG["A"]["pool_mu_grid"], np.float64)
BOOT_SEED = int(CFG["seed"])
N_BOOT = int(CFG["bootstrap"]["n"])

sys.path.insert(0, str(ROOT / "src"))
os.environ["URBANFLOW_ROOT"] = str(EXTERNAL)
from urbanflow import fields, geometry as G, registry, splits  # noqa: E402


# =============================================================================
# IO / provenance
# =============================================================================
def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jdef(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(type(o).__name__)


def jdump(obj, path: pathlib.Path):
    text = json.dumps(obj, indent=1, sort_keys=True, default=_jdef,
                      allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def iso_date(ordinal):
    return (_dt.date(1970, 1, 1) + _dt.timedelta(days=int(ordinal))).isoformat()


def environment():
    return dict(
        python=sys.version,
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        threadpoolctl=threadpoolctl.__version__,
        blas_thread_env={k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        threadpool_at_record=threadpoolctl.threadpool_info(),
        python_dont_write_bytecode=os.environ.get("PYTHONDONTWRITEBYTECODE"),
        urbanflow_root=os.environ.get("URBANFLOW_ROOT"),
        seed=BOOT_SEED,
        random_generator="numpy.random.default_rng (bootstrap only; all fits deterministic)",
    )


# =============================================================================
# Reference numerical layer, generalized from 24 to 18 phases
# =============================================================================
def dft(traj):
    """(n,18,p) -> (n,10,p), one-sided unitary DFT over observed phase."""
    return np.fft.fft(np.asarray(traj), axis=1)[:, :NHARM, :] / np.sqrt(NPHASE)


def idft(C):
    """Hermitian completion with h=9 Nyquist stored once, then real ifft."""
    C = np.asarray(C)
    assert C.shape[1] == NHARM
    full = np.empty((C.shape[0], NPHASE, C.shape[2]), np.complex128)
    full[:, :NHARM] = C
    # Negative frequencies h=10..17 are conjugates of h=8..1.  h=0 and h=9
    # are not duplicated.
    full[:, NHARM:] = np.conj(C[:, NHARM - 2:0:-1])
    return (np.fft.ifft(full, axis=1) * np.sqrt(NPHASE)).real


def pod_basis(A, r):
    """Reference TRAIN-only POD via the smaller Gram matrix."""
    A = np.asarray(A, np.float64)
    m, p = A.shape
    r = int(min(r, min(m, p)))
    if m >= p:
        w, V = np.linalg.eigh(A.T @ A)
        idx = np.argsort(-w)[:r]
        return np.ascontiguousarray(V[:, idx]), np.sqrt(np.maximum(w[idx], 0.0))
    w, U = np.linalg.eigh(A @ A.T)
    idx = np.argsort(-w)[:r]
    U = U[:, idx]
    s = np.sqrt(np.maximum(w[idx], 0.0))
    V = (A.T @ U) / np.maximum(s, 1e-300)[None, :]
    return np.ascontiguousarray(V), s


def E_flat(T, P):
    den = float((np.abs(T) ** 2).sum())
    num = float((np.abs(T - P) ** 2).sum())
    return num / den if den > 0 else float("nan")


def E_series(Yo, Yp):
    """Reference A per-date Parseval-weighted error in harmonic coordinates."""
    w = W_H[None, :, None]
    num = (np.abs(Yo - Yp) ** 2 * w).sum(axis=(1, 2))
    den = (np.abs(Yo) ** 2 * w).sum(axis=(1, 2))
    return num / np.maximum(den, 1e-300)


def E_pooled(Yo, Yp):
    e = E_series(Yo, Yp)
    den = float((np.abs(Yo) ** 2 * W_H[None, :, None]).sum())
    num = float((np.abs(Yo - Yp) ** 2 * W_H[None, :, None]).sum())
    return (num / den if den > 0 else float("nan")), e, num, den


def E_series_flat(T, P):
    """Reference D per-date error on flattened (date,phase) rows."""
    n = T.shape[0] // NPHASE
    assert T.shape[0] == n * NPHASE
    out = np.empty(n)
    for i in range(n):
        sl = slice(i * NPHASE, (i + 1) * NPHASE)
        den = float((np.abs(T[sl]) ** 2).sum())
        num = float((np.abs(T[sl] - P[sl]) ** 2).sum())
        out[i] = num / den if den > 0 else np.nan
    return out


def cridge_kappa(Xtr, Ytr, Xva, Yva, kappas=KAPPA):
    """Exact reference dual/primal relative-kappa complex ridge."""
    Xtr = np.asarray(Xtr, np.complex128)
    Ytr = np.asarray(Ytr, np.complex128)
    n, p = Xtr.shape
    dual = n <= p
    G = Xtr @ Xtr.conj().T if dual else Xtr.conj().T @ Xtr
    s, V = np.linalg.eigh(G)
    s = np.maximum(s.real, 0.0)
    smax = float(s.max()) if s.size else 1.0
    if dual:
        VhY = V.conj().T @ Ytr
        def _H(A):
            return Xtr.conj().T @ (V @ A)
    else:
        VhY = V.conj().T @ (Xtr.conj().T @ Ytr)
        def _H(A):
            return V @ A
    best = None
    for k in kappas:
        lam = float(k) * smax
        H = _H(VhY / (s + lam)[:, None])
        ev = E_flat(Yva, Xva @ H)
        if best is None or ev < best[0]:
            best = (ev, float(k), float(lam), H)
    nz = s[s > smax * 1e-12]
    return dict(val=best[0], kappa=best[1], lam=best[2], H=best[3], smax=smax,
                df_eff=float((s / (s + best[2])).sum()), n_fit=int(n),
                n_feat=int(p), dual=dual,
                cond=float(smax / max(nz.min(), 1e-300)) if nz.size else float("inf"))


def boot_gain(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Reference paired-date bootstrap of mean(a)-mean(b); negative favors b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    m = d[idx].mean(1)
    return dict(n=int(d.size), mean=float(d.mean()),
                sd=float(d.std(ddof=1)) if d.size > 1 else 0.0,
                ci95=[float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
                frac_improved=float((d < 0).mean()),
                possible_ordered_date_resamples=int(d.size ** d.size))


# =============================================================================
# Data and full-FACE orthogonal-completion scoring
# =============================================================================
def load_data():
    spec = registry.spec("Chengdu")
    sp = splits.resolve(spec, "window", 0)
    for key in ("dates", "train", "val", "test"):
        assert list(sp.get(key, CFG["window"][key])) == list(CFG["window"][key])
    order = list(dict.fromkeys(sp["train"] + sp["val"] + sp["test"]))
    assert order == CFG["window"]["dates"]
    cache = EXTERNAL / CFG["field"]["cache"]
    F, meta = G.load_flux(CFG["field"]["cache"], CFG["field"]["geometry_key"])
    Fr, ph, di = fields.stack_rows(F, meta, order, hours=sp["hours"])
    assert Fr.shape == (len(order) * NPHASE, G.ctx("Chengdu")["n_int"])
    assert np.array_equal(ph.reshape(len(order), NPHASE),
                          np.tile(np.asarray(sp["hours"], np.int64), (len(order), 1)))
    assert np.array_equal(di, np.repeat(np.arange(len(order)), NPHASE))
    D = G.D("Chengdu")
    X = np.ascontiguousarray((D @ Fr.T).T, np.float64)
    R = np.ascontiguousarray(G.P0("Chengdu", Fr), np.float64)
    n_int = Fr.shape[1]
    rec = dict(X24=X.reshape(len(order), NPHASE, -1),
               R24=R.reshape(len(order), NPHASE, n_int),
               F24=Fr.reshape(len(order), NPHASE, n_int),
               order=order, sp=sp, meta=meta, D=D)
    rec["pos"] = {int(d): i for i, d in enumerate(order)}
    rec["itr"] = np.array([rec["pos"][int(d)] for d in sp["train"]], np.int64)
    rec["iva"] = np.array([rec["pos"][int(d)] for d in sp["val"]], np.int64)
    rec["ite"] = np.array([rec["pos"][int(d)] for d in sp["test"]], np.int64)
    rec["ker_resid_maxabs"] = float(np.abs(D @ R.T).max())
    rec["decomposition_conservation_maxabs"] = float(
        np.abs((D @ (Fr - R).T) - X.T).max())
    rec["input_hashes"] = {
        CFG["field"]["cache"]: sha256(cache),
        CFG["field"]["geometry"]: sha256(EXTERNAL / CFG["field"]["geometry"]),
    }
    assert rec["input_hashes"][CFG["field"]["cache"]] == CFG["field"]["cache_sha256"]
    assert rec["input_hashes"][CFG["field"]["geometry"]] == CFG["field"]["geometry_sha256"]
    return rec


def full_face_spectral(Y, Yfull, predictions):
    """Add out-of-POD energy to A errors without changing reference scores."""
    out = ((np.abs(Yfull) ** 2).sum(axis=2)
           - (np.abs(Y) ** 2).sum(axis=2))
    out = np.maximum(out, 0.0)
    out_w = (out * W_H[None, :]).sum(axis=1)
    den_full_date = (np.abs(Yfull) ** 2 * W_H[None, :, None]).sum(axis=(1, 2))
    out_result = {}
    for name, P in predictions.items():
        num_in_date = (np.abs(Y - P) ** 2 * W_H[None, :, None]).sum(axis=(1, 2))
        per = (num_in_date + out_w) / den_full_date
        pooled = float((num_in_date.sum() + out_w.sum()) / den_full_date.sum())
        out_result[name] = dict(E=pooled, per_date=per)
    return out_result, out


def full_face_phase(T, Ffull, predictions):
    """Add out-of-POD energy to D phase-domain errors, per date and pooled."""
    nd = T.shape[0] // NPHASE
    T3 = T.reshape(nd, NPHASE, -1)
    F3 = Ffull.reshape(nd, NPHASE, -1)
    out = (np.abs(F3) ** 2).sum(axis=(1, 2)) - (np.abs(T3) ** 2).sum(axis=(1, 2))
    out = np.maximum(out, 0.0)
    den = (np.abs(F3) ** 2).sum(axis=(1, 2))
    result = {}
    for name, P in predictions.items():
        P3 = P.reshape(nd, NPHASE, -1)
        num = (np.abs(T3 - P3) ** 2).sum(axis=(1, 2))
        per = (num + out) / den
        result[name] = dict(E=float((num.sum() + out.sum()) / den.sum()),
                            per_date=per)
    return result, out


# =============================================================================
# Experiment A (reference definition)
# =============================================================================
def _pred(X, Hs):
    return np.einsum("nhp,hpq->nhq", X, Hs)


def fit_A(PX, PY, itr, iva, rx, ry):
    Xtr = np.ascontiguousarray(PX[itr][:, :, :rx])
    Ytr = np.ascontiguousarray(PY[itr][:, :, :ry])
    Xva = np.ascontiguousarray(PX[iva][:, :, :rx])
    Yva = np.ascontiguousarray(PY[iva][:, :, :ry])
    stack = lambda A: A.reshape(A.shape[0] * A.shape[1], A.shape[2])

    Kf = cridge_kappa(stack(Xtr), stack(Ytr), stack(Xva), stack(Yva))
    K = Kf["H"]
    cx = np.sqrt((np.abs(Xtr) ** 2).mean(axis=(0, 2)))
    cx = np.where(cx > 0, 1.0 / np.maximum(cx, 1e-300), 1.0)
    sc = cx[None, :, None]
    Kbf = cridge_kappa(stack(Xtr * sc), stack(Ytr * sc),
                       stack(Xva * sc), stack(Yva * sc))
    Kb = Kbf["H"]

    Hf = np.empty((NHARM, rx, ry), np.complex128)
    freq_fits = []
    for h in range(NHARM):
        f = cridge_kappa(Xtr[:, h, :], Ytr[:, h, :], Xva[:, h, :], Yva[:, h, :])
        Hf[h] = f["H"]
        freq_fits.append({k: f[k] for k in ("val", "kappa", "lam", "df_eff", "smax")})

    val = {}
    val["M_shared"] = E_pooled(Yva, _pred(Xva, np.broadcast_to(K, Hf.shape)))[0]
    val["M_shared_bal"] = E_pooled(Yva, _pred(Xva, np.broadcast_to(Kb, Hf.shape)))[0]
    val["M_freq"] = E_pooled(Yva, _pred(Xva, Hf))[0]
    best = None
    for mu in MU_GRID:
        Hp = (1.0 - mu) * K[None] + mu * Hf
        ev = E_pooled(Yva, _pred(Xva, Hp))[0]
        if best is None or ev < best[0]:
            best = (ev, float(mu))
    val["M_pool"], mu_star = best
    Hp = (1.0 - mu_star) * K[None] + mu_star * Hf
    return dict(rx=int(rx), ry=int(ry), val=val, mu_star=mu_star,
                kappa_shared=Kf["kappa"], kappa_bal=Kbf["kappa"],
                df_shared=Kf["df_eff"], cond_shared=Kf["cond"],
                shared_fit={k: Kf[k] for k in ("val", "kappa", "lam", "df_eff", "smax")},
                balanced_fit={k: Kbf[k] for k in ("val", "kappa", "lam", "df_eff", "smax")},
                freq_fits=freq_fits, K=K, Kb=Kb, Hf=Hf, Hp=Hp)


def spectral_audit(X):
    out = {}
    for h in range(NHARM):
        Xh = X[:, h]
        Sxx = Xh.conj().T @ Xh / Xh.shape[0]
        sv = np.linalg.svd(Sxx, compute_uv=False)
        smax = float(sv.max())
        nz = sv[sv > smax * 1e-12]
        out[str(h)] = dict(
            n_dates=int(Xh.shape[0]), smax=smax,
            cond=float(smax / max(nz.min(), 1e-300)) if nz.size else float("inf"),
            eff_rank=int((sv > smax * 1e-6).sum()),
            numerical_rank=int(nz.size),
            energy_frac_nonzero=[float(x / max(sv.sum(), 1e-300)) for x in nz])
    return out


def run_A(rec, PX, PY, RF):
    itr, iva, ite = rec["itr"], rec["iva"], rec["ite"]
    rx_grid = CFG["A"]["r_x_grid"]
    ry_grid = CFG["A"]["r_y_grid"]
    grid = []
    LOG.info("A: VAL-only rank grid")
    for rx in rx_grid:
        for ry in ry_grid:
            if ry > rx:
                continue
            f = fit_A(PX, PY, itr, iva, rx, ry)
            row = dict(rx=rx, ry=ry, val=f["val"], mu_star=f["mu_star"],
                       kappa_shared=f["kappa_shared"], kappa_bal=f["kappa_bal"],
                       df_shared=f["df_shared"])
            grid.append(row)
            LOG.info("A grid rx=%d ry=%d M_pool_VAL=%.8f mu=%.1f",
                     rx, ry, row["val"]["M_pool"], row["mu_star"])
    chosen_row = min(grid, key=lambda r: r["val"]["M_pool"])
    rx, ry = int(chosen_row["rx"]), int(chosen_row["ry"])
    LOG.info("A chosen rx=%d ry=%d; refit and read TEST", rx, ry)
    f = fit_A(PX, PY, itr, iva, rx, ry)
    Xte, Yte, Yfull = PX[ite][:, :, :rx], PY[ite][:, :, :ry], RF[ite]
    predictions = dict(
        M_shared=_pred(Xte, np.broadcast_to(f["K"], f["Hf"].shape)),
        M_shared_bal=_pred(Xte, np.broadcast_to(f["Kb"], f["Hf"].shape)),
        M_freq=_pred(Xte, f["Hf"]),
        M_pool=_pred(Xte, f["Hp"]),
        M_zero=np.zeros_like(Yte),
    )
    per_date, pooled, nums, dens = {}, {}, {}, {}
    for name, P in predictions.items():
        e, ev, nu, de = E_pooled(Yte, P)
        pooled[name], per_date[name], nums[name], dens[name] = e, ev, nu, de
    gains = {
        "freq_vs_shared": boot_gain(per_date["M_freq"], per_date["M_shared"]),
        "pool_vs_shared": boot_gain(per_date["M_pool"], per_date["M_shared"]),
        "pool_vs_freq": boot_gain(per_date["M_pool"], per_date["M_freq"]),
        "bal_vs_shared": boot_gain(per_date["M_shared_bal"], per_date["M_shared"]),
    }
    full, out_energy = full_face_spectral(Yte, Yfull, predictions)
    full_per_date = {k: v["per_date"] for k, v in full.items()}
    full_gains = {
        "freq_vs_shared": boot_gain(full_per_date["M_freq"], full_per_date["M_shared"]),
        "pool_vs_shared": boot_gain(full_per_date["M_pool"], full_per_date["M_shared"]),
        "pool_vs_freq": boot_gain(full_per_date["M_pool"], full_per_date["M_freq"]),
        "bal_vs_shared": boot_gain(full_per_date["M_shared_bal"], full_per_date["M_shared"]),
    }
    eta = {}
    per_harmonic = []
    for h in range(NHARM):
        den = float((np.abs(Xte[:, h] @ f["K"]) ** 2).sum())
        num = float((np.abs(Xte[:, h] @ (f["Hf"][h] - f["K"])) ** 2).sum())
        eta[str(h)] = num / den if den > 0 else float("nan")
        row = dict(h=h)
        for name, Hs in (("shared", np.broadcast_to(f["K"], f["Hf"].shape)),
                         ("free", f["Hf"]), ("pool", f["Hp"])):
            P = Xte[:, h] @ Hs[h]
            row[name] = E_flat(Yte[:, h], P)
        per_harmonic.append(row)
    out = dict(
        definition="reference A; inside TRAIN-only POD unless explicitly labelled full_face",
        chosen=dict(rx=rx, ry=ry, selection="minimum M_pool VAL"),
        grid=grid,
        selected_fit=dict(mu_star=f["mu_star"], shared=f["shared_fit"],
                          shared_bal=f["balanced_fit"], per_harmonic=f["freq_fits"]),
        n_test_dates=int(len(ite)), test_dates=[int(rec["order"][i]) for i in ite],
        E=pooled,
        per_date={k: [float(x) for x in v] for k, v in per_date.items()},
        gains=gains, eta_h=eta, per_harmonic=per_harmonic,
        spectral_train=spectral_audit(PX[itr][:, :, :rx]),
        full_face=dict(
            definition="zero prediction outside selected TRAIN routing POD; out-of-POD energy retained",
            E={k: v["E"] for k, v in full.items()},
            per_date={k: [float(x) for x in v["per_date"]] for k, v in full.items()},
            gains=full_gains,
            test_out_of_pod_energy_by_harmonic=out_energy.tolist()),
        interpretation="exploratory only; no frequency-flat verdict with three TEST dates",
    )
    return out


# =============================================================================
# Experiment D (reference definition, compact even-Nyquist phase design)
# =============================================================================
def phase_design(Z, ph):
    """18 real Fourier blocks: DC, cos/sin 1..8, and Nyquist cos 9 only."""
    cols = [Z]
    w = 2.0 * np.pi * np.asarray(ph) / NPHASE
    for k in range(1, NPHASE // 2):
        cols.append(Z * np.cos(k * w)[:, None])
        cols.append(Z * np.sin(k * w)[:, None])
    cols.append(Z * np.cos((NPHASE // 2) * w)[:, None])
    return np.concatenate(cols, axis=1)


def expanded_phase_design_with_zero_nyquist_sine(Z, ph):
    """Direct 18-phase analogue of et_D, used only to verify compactification."""
    cols = [Z]
    w = 2.0 * np.pi * np.asarray(ph) / NPHASE
    for k in range(1, NPHASE // 2 + 1):
        cols.append(Z * np.cos(k * w)[:, None])
        cols.append(Z * np.sin(k * w)[:, None])
    return np.concatenate(cols, axis=1)


def quad_features(Z, q=8):
    A = np.asarray(Z[:, :q], float)
    iu = np.triu_indices(A.shape[1])
    return A[:, iu[0]] * A[:, iu[1]]


def _krr_solve(K, r, alphas):
    w, V = np.linalg.eigh(K)
    Vtr = V.T @ r
    return {a: V @ (Vtr / (w + a)[:, None]) for a in alphas}


def arms_D_at(rx, Ztr, Ttr, Zva, Tva, Zte, Tte, ph_tr, ph_va, ph_te):
    out = {}
    f0 = cridge_kappa(Ztr, Ttr, Zva, Tva)
    Ptr0, Pva0, Pte0 = Ztr @ f0["H"], Zva @ f0["H"], Zte @ f0["H"]
    out["lin0"] = dict(te=Pte0, va=Pva0, tr=Ptr0,
                       fit={k: f0[k] for k in ("val", "kappa", "lam", "df_eff", "smax")})

    Dtr, Dva, Dte = (phase_design(Ztr, ph_tr), phase_design(Zva, ph_va),
                     phase_design(Zte, ph_te))
    fF = cridge_kappa(Dtr, Ttr, Dva, Tva)
    PtrF, PvaF, PteF = Dtr @ fF["H"], Dva @ fF["H"], Dte @ fF["H"]
    out["linF"] = dict(te=PteF, va=PvaF, tr=PtrF, kappa=fF["kappa"],
                       df_eff=fF["df_eff"], n_feat=Dtr.shape[1],
                       fit={k: fF[k] for k in ("val", "kappa", "lam", "df_eff", "smax")})

    Qtr, Qva, Qte = (quad_features(z, CFG["D"]["q_dim"]) for z in (Ztr, Zva, Zte))
    g = cridge_kappa(np.concatenate([Dtr, Qtr], 1), Ttr,
                     np.concatenate([Dva, Qva], 1), Tva)
    out["quadF"] = dict(te=np.concatenate([Dte, Qte], 1) @ g["H"],
                        va=np.concatenate([Dva, Qva], 1) @ g["H"],
                        tr=np.concatenate([Dtr, Qtr], 1) @ g["H"],
                        kappa=g["kappa"], q_dim=int(Qtr.shape[1]),
                        fit={k: g[k] for k in ("val", "kappa", "lam", "df_eff", "smax")})

    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
    Zs = lambda A: (A - mu) / sd
    Ztr_s = Zs(Ztr)
    def sqd(A, B):
        return (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
                - 2.0 * (A @ B.T))
    d2 = sqd(Ztr_s, Ztr_s)
    med = float(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
    Ktr_full = sqd(Ztr_s, Ztr_s)
    Kva_full = sqd(Zs(Zva), Ztr_s)
    Kte_full = sqd(Zs(Zte), Ztr_s)
    resid_tr = Ttr - PtrF
    best = None
    for mult in CFG["D"]["sigma_mult_grid"]:
        sig2 = max(float(mult) * med, 1e-12)
        sols = _krr_solve(np.exp(-Ktr_full / (2 * sig2)), resid_tr,
                          CFG["D"]["alpha_grid"])
        for a, sol in sols.items():
            pv = PvaF + np.exp(-Kva_full / (2 * sig2)) @ sol
            ev = E_flat(Tva, pv)
            if best is None or ev < best[0]:
                best = (ev, float(mult), float(a), sol, sig2)
    krr_val, mult, alpha, sol, sig2 = best
    Ktr = np.exp(-Ktr_full / (2 * sig2))
    out["krrF"] = dict(
        te=PteF + np.exp(-Kte_full / (2 * sig2)) @ sol,
        va=PvaF + np.exp(-Kva_full / (2 * sig2)) @ sol,
        tr=PtrF,  # exact et_D reporting convention
        tr_full=PtrF + Ktr @ sol,
        sigma_mult=mult, alpha=alpha, sigma2=float(sig2), median_d2=med,
        val=krr_val)
    return out


def run_D(rec, PX, PY, R24, Uy, ry):
    itr, iva, ite = rec["itr"], rec["iva"], rec["ite"]
    Zall, Tall = idft(PX), idft(PY[:, :, :ry])
    ph = np.arange(NPHASE)
    def flat(idx, A, rx=None):
        B = A[idx].reshape(-1, A.shape[2])
        return np.ascontiguousarray(B[:, :rx]) if rx is not None else np.ascontiguousarray(B)

    rank_val = []
    for rx in CFG["D"]["r_x_grid"]:
        Dtr = phase_design(flat(itr, Zall, rx), np.tile(ph, len(itr)))
        Dva = phase_design(flat(iva, Zall, rx), np.tile(ph, len(iva)))
        f = cridge_kappa(Dtr, flat(itr, Tall), Dva, flat(iva, Tall))
        rank_val.append(dict(rx=int(rx), val=float(f["val"]), kappa=f["kappa"],
                             df_eff=f["df_eff"], n_feat=int(Dtr.shape[1])))
    best_rx = int(min(rank_val, key=lambda r: r["val"])["rx"])
    LOG.info("D chosen rx=%d (ry=%d from A); VAL rank scan complete", best_rx, ry)

    rank_probe = {}
    for rx in CFG["D"]["r_x_grid"]:
        A = arms_D_at(rx, flat(itr, Zall, rx), flat(itr, Tall),
                      flat(iva, Zall, rx), flat(iva, Tall),
                      flat(ite, Zall, rx), flat(ite, Tall),
                      np.tile(ph, len(itr)), np.tile(ph, len(iva)), np.tile(ph, len(ite)))
        Tte = flat(ite, Tall)
        pb = E_series_flat(Tte, A["linF"]["te"])
        pq = E_series_flat(Tte, A["quadF"]["te"])
        ff, _ = full_face_phase(Tte, R24[ite].reshape(-1, R24.shape[-1]),
                                {"linF": A["linF"]["te"], "quadF": A["quadF"]["te"]})
        rank_probe[str(rx)] = dict(
            E_lin=float(np.nanmean(pb)), E_quad=float(np.nanmean(pq)),
            dR2=float(np.nanmean(pb) - np.nanmean(pq)),
            E_lin_full_face=ff["linF"]["E"], E_quad_full_face=ff["quadF"]["E"],
            dR2_full_face=float(ff["linF"]["E"] - ff["quadF"]["E"]),
            lin_kappa=A["linF"]["kappa"], quad_kappa=A["quadF"]["kappa"],
            krr_sigma_mult=A["krrF"]["sigma_mult"], krr_alpha=A["krrF"]["alpha"])
        LOG.info("D rank probe rx=%d quad_dR2=%+.8f", rx, rank_probe[str(rx)]["dR2"])

    rx = best_rx
    Ztr, Zva, Zte = (flat(i, Zall, rx) for i in (itr, iva, ite))
    Ttr, Tva, Tte = (flat(i, Tall) for i in (itr, iva, ite))
    A = arms_D_at(rx, Ztr, Ttr, Zva, Tva, Zte, Tte,
                  np.tile(ph, len(itr)), np.tile(ph, len(iva)), np.tile(ph, len(ite)))
    arms = {}
    for name in CFG["D"]["arms"]:
        arms[name] = dict(
            tr=E_flat(Ttr, A[name]["tr"]), va=E_flat(Tva, A[name]["va"]),
            te=E_flat(Tte, A[name]["te"]),
            per_date=E_series_flat(Tte, A[name]["te"]).tolist())
    arms["krrF"]["tr_full"] = E_flat(Ttr, A["krrF"]["tr_full"])
    full, out_energy = full_face_phase(Tte, R24[ite].reshape(-1, R24.shape[-1]),
                                       {k: A[k]["te"] for k in CFG["D"]["arms"]})

    pd = {k: np.asarray(arms[k]["per_date"]) for k in CFG["D"]["arms"]}
    summary = {}
    for name in CFG["D"]["arms"]:
        g = boot_gain(pd[name], pd["linF"])
        summary[name] = dict(E_te=float(np.nanmean(pd[name])), dR2=-g["mean"],
                             ci=[-g["ci95"][1], -g["ci95"][0]], n=g["n"],
                             frac_improved=g["frac_improved"],
                             possible_ordered_date_resamples=g["possible_ordered_date_resamples"])
    full_summary = {}
    for name in CFG["D"]["arms"]:
        g = boot_gain(full[name]["per_date"], full["linF"]["per_date"])
        full_summary[name] = dict(E_te=float(np.mean(full[name]["per_date"])),
                                  dR2=-g["mean"], ci=[-g["ci95"][1], -g["ci95"][0]],
                                  n=g["n"], frac_improved=g["frac_improved"])

    n_train_rows = int(Ttr.shape[0])
    capture_date = ((np.abs(Tall) ** 2).sum(axis=(1, 2))
                    / np.maximum((np.abs(R24) ** 2).sum(axis=(1, 2)), 1e-300))
    out = dict(
        definition="reference D nested nonlinear increments; inside TRAIN-only POD unless full_face",
        chosen=dict(rx=rx, ry=int(ry), selection="minimum linF VAL; ry inherited from A"),
        rank_val=rank_val, rank_probe=rank_probe,
        n_train_dates=int(len(itr)), n_test_dates=int(len(ite)),
        n_train_rows=n_train_rows, n_feat_lin=int(A["linF"]["n_feat"]),
        feature_saturation=float(A["linF"]["n_feat"] / n_train_rows),
        lin_kappa=A["linF"]["kappa"], lin_df=A["linF"]["df_eff"],
        quad_kappa=A["quadF"]["kappa"], quad_q_dim=int(A["quadF"]["q_dim"]),
        krr=dict(sigma_mult=A["krrF"]["sigma_mult"], alpha=A["krrF"]["alpha"],
                 sigma2=A["krrF"]["sigma2"], median_squared_distance=A["krrF"]["median_d2"],
                 val=A["krrF"]["val"]),
        arms=arms, summary=summary,
        per_date={k: arms[k]["per_date"] for k in CFG["D"]["arms"]},
        material_margin=CFG["D"]["material_margin"],
        equivalence_band=CFG["D"]["equivalence_band"],
        full_face=dict(
            definition="zero prediction outside selected TRAIN routing POD; out-of-POD energy retained",
            E={k: full[k]["E"] for k in CFG["D"]["arms"]},
            per_date={k: full[k]["per_date"].tolist() for k in CFG["D"]["arms"]},
            summary=full_summary,
            test_out_of_pod_energy_by_date=out_energy.tolist(),
            routing_pod_capture_by_date=capture_date.tolist()),
        decision="UNDERPOWERED_EXPLORATORY",
        interpretation=("three TEST dates cannot support a confirmatory nonlinear or no-nonlinear verdict; "
                        "material/equivalence thresholds are recorded but not used to claim absence"),
    )
    return out


# =============================================================================
# Validation and main
# =============================================================================
def validate(rec, PX, PY, RF, Ux, Uy, A, D, phase_compact_error):
    checks = {}
    def check(name, condition, detail):
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check("input_sha256", all(rec["input_hashes"][k] == CFG["field"][f"{kind}_sha256"]
                              for k, kind in ((CFG["field"]["cache"], "cache"),
                                              (CFG["field"]["geometry"], "geometry"))),
          rec["input_hashes"])
    check("complete_14_by_18_window", rec["X24"].shape[:2] == (14, 18) and
          rec["R24"].shape[:2] == (14, 18), dict(X=list(rec["X24"].shape), R=list(rec["R24"].shape)))
    check("split_8_3_3_disjoint", (len(rec["itr"]), len(rec["iva"]), len(rec["ite"])) == (8, 3, 3)
          and not (set(rec["itr"]) & set(rec["iva"]) or set(rec["itr"]) & set(rec["ite"])
                   or set(rec["iva"]) & set(rec["ite"])),
          dict(train=rec["sp"]["train"], val=rec["sp"]["val"], test=rec["sp"]["test"]))
    scale = max(float(np.abs(rec["F24"]).max()), 1.0)
    check("routing_in_ker_D", rec["ker_resid_maxabs"] <= 1e-8 * scale,
          dict(max_abs=rec["ker_resid_maxabs"], tolerance=1e-8 * scale))
    check("continuity_decomposition", rec["decomposition_conservation_maxabs"] <= 1e-10 * scale,
          dict(max_abs=rec["decomposition_conservation_maxabs"], tolerance=1e-10 * scale))
    check("one_sided_weights", NHARM == 10 and np.array_equal(W_H, np.array([1] + [2] * 8 + [1]))
          and W_H.sum() == NPHASE, W_H.tolist())
    parseval = {}
    for name, V in (("X", rec["X24"][:2]), ("Fcirc", rec["R24"][:2])):
        C = dft(V)
        lhs = float((np.abs(C) ** 2 * W_H[None, :, None]).sum())
        rhs = float((np.abs(V) ** 2).sum())
        parseval[name] = abs(lhs - rhs) / max(rhs, 1e-300)
    check("parseval_X_and_Fcirc", max(parseval.values()) < 1e-11, parseval)
    recon = max(float(np.abs(idft(dft(rec["X24"][:2])) - rec["X24"][:2]).max()),
                float(np.abs(idft(dft(rec["R24"][:2])) - rec["R24"][:2]).max()))
    check("hermitian_idft_reconstruction", recon < 1e-8, dict(max_abs=recon))
    nyq = max(float(np.abs(dft(rec["X24"])[:, 9].imag).max()),
              float(np.abs(RF[:, 9].imag).max()))
    dc = max(float(np.abs(dft(rec["X24"])[:, 0].imag).max()),
             float(np.abs(RF[:, 0].imag).max()))
    check("dc_and_nyquist_real", max(nyq, dc) < 1e-8, dict(dc_imag_max=dc, nyquist_imag_max=nyq))
    check("nyquist_sine_column_removed", phase_compact_error < 1e-12,
          dict(max_abs_compact_vs_expanded=phase_compact_error,
               compact_blocks_per_source_coordinate=18,
               expanded_blocks_including_zero_sine=19))
    orth = max(float(np.abs(Ux.T @ Ux - np.eye(Ux.shape[1])).max()),
               float(np.abs(Uy.T @ Uy - np.eye(Uy.shape[1])).max()))
    check("train_pod_orthonormal", orth < 1e-9, dict(max_abs=orth))
    check("A_rank_selected_on_VAL", A["chosen"]["rx"] == min(A["grid"], key=lambda r: r["val"]["M_pool"])["rx"]
          and A["chosen"]["ry"] == min(A["grid"], key=lambda r: r["val"]["M_pool"])["ry"],
          A["chosen"])
    check("D_rank_selected_on_VAL_and_ry_from_A",
          D["chosen"]["rx"] == min(D["rank_val"], key=lambda r: r["val"])["rx"]
          and D["chosen"]["ry"] == A["chosen"]["ry"], D["chosen"])
    check("A_zero_reference", abs(A["E"]["M_zero"] - 1.0) < 1e-12
          and abs(A["full_face"]["E"]["M_zero"] - 1.0) < 1e-12,
          dict(inside=A["E"]["M_zero"], full_face=A["full_face"]["E"]["M_zero"]))
    dmax = 0.0
    for name in CFG["D"]["arms"]:
        vals = np.asarray(D["per_date"][name])
        dmax = max(dmax, abs(D["summary"][name]["E_te"] - float(vals.mean())),
                   abs(D["summary"][name]["dR2"] -
                       float(np.mean(D["per_date"]["linF"]) - vals.mean())))
    check("D_summary_recomputed_from_per_date", dmax < 1e-12, dict(max_abs=dmax))
    check("three_test_dates_explicitly_underpowered",
          A["n_test_dates"] == D["n_test_dates"] == 3
          and D["decision"] == "UNDERPOWERED_EXPLORATORY",
          dict(n_test=3, decision=D["decision"], no_absence_claim=True))
    return dict(all_pass=True, checks=checks)


def main():
    t0 = time.time()
    LOG.info("Loading Chengdu full-FACE data; no road basis")
    rec = load_data()
    max_rank = max(max(CFG["A"]["r_x_grid"]), max(CFG["A"]["r_y_grid"]))
    itr = rec["itr"]
    LOG.info("TRAIN-only POD: %d rows, max rank %d", len(itr) * NPHASE, max_rank)
    Ux, sx = pod_basis(rec["X24"][itr].reshape(-1, rec["X24"].shape[-1]), max_rank)
    Uy, sy = pod_basis(rec["R24"][itr].reshape(-1, rec["R24"].shape[-1]), max_rank)
    PX = dft(rec["X24"] @ Ux)
    PY = dft(rec["R24"] @ Uy)
    RF = dft(rec["R24"])
    Zprobe = rec["X24"][0, :, :3]
    php = np.arange(NPHASE)
    # Expanded ordering is compact ordering plus a final, identically-zero sine-9 block.
    compact = phase_design(Zprobe, php)
    expanded = expanded_phase_design_with_zero_nyquist_sine(Zprobe, php)
    phase_compact_error = float(np.abs(compact - expanded[:, :compact.shape[1]]).max())

    A = run_A(rec, PX, PY, RF)
    LOG.info("A TEST E: %s", {k: round(v, 8) for k, v in A["E"].items()})
    D = run_D(rec, PX, PY, rec["R24"], Uy, A["chosen"]["ry"])
    LOG.info("D TEST summaries: %s", {k: round(v["dR2"], 8) for k, v in D["summary"].items()})

    checks = validate(rec, PX, PY, RF, Ux, Uy, A, D, phase_compact_error)
    reference_paths = {
        "A_script": EXTERNAL / CFG["references"]["A_script"],
        "shared_script": EXTERNAL / CFG["references"]["shared_script"],
        "D_script_read_only_legacy": pathlib.Path(CFG["references"]["D_script_read_only_legacy"]),
        "D_amendment_read_only_legacy": pathlib.Path(CFG["references"]["D_amendment_read_only_legacy"]),
        "urbanflow_reference_manifest": ROOT / "src/urbanflow/manifest.json",
        "urbanflow_geometry": ROOT / "src/urbanflow/geometry.py",
        "urbanflow_fields": ROOT / "src/urbanflow/fields.py",
        "urbanflow_registry": ROOT / "src/urbanflow/registry.py",
        "this_script": _HERE,
        "config": ROOT / "configs" / "experiments" / "chengdu_nonlinear.json",
    }
    references = {k: dict(path=str(p), sha256=sha256(p)) for k, p in reference_paths.items()}
    manifest = json.loads((ROOT / "src/urbanflow/manifest.json").read_text(encoding="utf-8"))
    data_audit = dict(
        shape_X=list(rec["X24"].shape), shape_Fcirc=list(rec["R24"].shape),
        NC=int(rec["X24"].shape[-1]), n_int=int(rec["R24"].shape[-1]),
        order=rec["order"], order_iso=[iso_date(d) for d in rec["order"]],
        train=rec["sp"]["train"], val=rec["sp"]["val"], test=rec["sp"]["test"],
        hours=rec["sp"]["hours"], ker_D_maxabs=rec["ker_resid_maxabs"],
        decomposition_conservation_maxabs=rec["decomposition_conservation_maxabs"],
        train_pod_singular_values_X=sx, train_pod_singular_values_Fcirc=sy,
        input_hashes=rec["input_hashes"],
        frequency_bins=[dict(h=h, cycles_per_observed_hour=h / NPHASE,
                             period_observed_hours=(NPHASE / h if h else None),
                             parseval_weight=float(W_H[h])) for h in range(NHARM)],
        clock_caveat=("DFT is over the observed 06--23 block. h=1 is an 18-observed-hour bin, "
                      "not a once-per-24-hour harmonic; periodic extension does not infer night flow."))
    env = environment()
    result = dict(
        schema="chengdu_nonlinear_exploratory_v1",
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        status="COMPLETE_VALIDATED_EXPLORATORY",
        SI_use=("allowed only as an explicitly exploratory SI numerical supplement with code, hashes, "
                "full-face/inside-POD distinction and the three-TEST-date underpowered caveat; "
                "not allowed as evidence of no nonlinearity or as a confirmatory frequency-flat verdict"),
        no_absence_of_nonlinearity_claim=True,
        runtime_seconds=None,
        script=references["this_script"], config=references["config"],
        config_protocol=CFG,
        inputs=data_audit,
        reference_code=references,
        urbanflow_reference=dict(id=manifest["payload"]["reference_id"],
                                 version=manifest["payload"]["version"], digest=manifest["digest"]),
        environment=env,
        A=A, D=D,
        validation=dict(all_pass=checks["all_pass"], detail_file="validation.json"),
    )
    result["runtime_seconds"] = float(time.time() - t0)
    result_path = jdump(result, RESULTS / "chengdu_nonlinear_exploratory.json")
    jdump(env | dict(script=references["this_script"], inputs=rec["input_hashes"]),
          RESULTS / "environment.json")
    validation = dict(
        schema="chengdu_nonlinear_exploratory_validation_v1",
        created_utc=result["created_utc"], all_pass=True,
        result_file=str(result_path), result_sha256=sha256(result_path),
        config_sha256=sha256(ROOT / "configs" / "experiments" / "chengdu_nonlinear.json"), script_sha256=sha256(_HERE),
        checks=checks["checks"],
        statistical_limitation=dict(
            n_test_dates=3, status="UNDERPOWERED",
            bootstrap_draws=N_BOOT, possible_ordered_date_resamples=27,
            note=("9999 bootstrap draws do not create independent information; with three dates "
                  "there are only 27 ordered resamples. CIs are descriptive reference-defined intervals."),
            absence_of_nonlinearity_supported=False,
            confirmatory_frequency_flatness_supported=False),
    )
    jdump(validation, RESULTS / "validation.json")
    LOG.info("All validation checks passed; wrote %s", result_path)
    LOG.info("Runtime %.2f s", result["runtime_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
