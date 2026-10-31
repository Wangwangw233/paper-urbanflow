"""Small frozen-input adapters for the two public supplementary figures.

Only functions used by those entry points are included.  No aligned field or
model fitting is hidden behind this module.
"""
from __future__ import annotations
import json
import numpy as np
import figstyle as FS

NH = 13

def _jload(path):
    return json.loads(path.read_text(encoding="utf-8"))

def source_spectrum(city, kmax=8, rec=None):
    if kmax != 8:
        raise ValueError("the compact public product stores kmax=8 only")
    with np.load(FS.ROOT / "frozen/source_spectra.npz", allow_pickle=False) as z:
        q = np.asarray(z[f"{city}__q"], dtype=np.float64)
        return dict(q=q, pk=q, P=q.sum(axis=1), share=q.sum(axis=1),
                    kept=float(q.sum()), kept_hi=float(q[:2].sum()),
                    c0_err=float(z[f"{city}__c0_err"]), kmax=8,
                    n_modes=int(z[f"{city}__n_modes"]))

def transfer(city, rec=None, series=None, arms=None):
    a = _jload(FS.ET_RES / "A_frequency_arms.json")[city]
    return dict(eta=np.asarray([a["eta_h"][str(h)] for h in range(NH)], dtype=float))

def closure(city):
    c = _jload(FS.ET_RES / "C_spectral_closure.json")[city]
    h = np.arange(NH)
    return dict(h=h,
                trace=np.asarray([c[str(x)]["trace_ratio"] for x in h]),
                overlap=np.asarray([c[str(x)]["overlap_k3"] for x in h]),
                canon=np.asarray([c[str(x)]["canon_load"] for x in h]))

def nonlinear(city):
    d = _jload(FS.ET_RES / "D_nonlinear_increment.json")[city]
    return {a: dict(d=d["summary"][a]["dR2"], lo=d["summary"][a]["ci"][0],
                    hi=d["summary"][a]["ci"][1], e=d["summary"][a]["E_te"])
            for a in ("lin0", "linF", "quadF", "krrF")}

def bicoherence(city):
    z = _jload(FS.ET_RES / "E_bicoherence.json")[city]
    return {k: dict(pairs=[r["pair"] for r in z[k]["rows"]],
                    obs=np.asarray([r["observed"] for r in z[k]["rows"]]),
                    q99=np.asarray([r["surrogate_q99"] for r in z[k]["rows"]]),
                    p=np.asarray([r["p_one_sided"] for r in z[k]["rows"]]),
                    n=int(z[k]["n_dates"]))
            for k in ("source", "routing", "residual")}
