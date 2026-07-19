"""
Bayesian model comparison via PSIS-LOO / WAIC (ArviZ).

The pooled hierarchical fit uses a pm.Potential likelihood, which ArviZ cannot
decompose into pointwise terms. For comparison we refit with an equivalent
pm.CustomDist likelihood that records the pointwise log-likelihood, then use
arviz.compare. This answers your roadmap's cross-validation item: is the hand t0
better estimated or fixed, and (per saccade cell) single vs mixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._speeds import PHYSIO_FLOOR
from .models.wald import _build_units

_LOG2PI = float(np.log(2.0 * np.pi))


def _fit_with_loglik(rt, uidx, minrt, n_units, floor, t0_mode="estimate", mu_z_mean=0.5,
                     draws=600, tune=600, chains=2, target_accept=0.95, cores=1, seed=7):
    """Pooled hierarchical single-Wald with a CustomDist likelihood (records log_likelihood)."""
    import pymc as pm
    import pytensor.tensor as pt

    def wald_logp(value, v, a, t0):
        tau = value - t0
        safe = pt.maximum(tau, 1e-6)
        lp = pt.log(a) - 0.5 * _LOG2PI - 1.5 * pt.log(safe) - (a - v * safe) ** 2 / (2 * safe)
        return pt.switch(tau > 1e-6, lp, -1e30)

    with pm.Model() as m:
        mu_lv = pm.Normal("mu_lv", np.log(10), 0.5); s_lv = pm.HalfNormal("s_lv", 0.5)
        mu_la = pm.Normal("mu_la", np.log(1.0), 0.5); s_la = pm.HalfNormal("s_la", 0.5)
        lv = mu_lv + s_lv * pm.Normal("lv_raw", 0, 1, shape=n_units)
        la = mu_la + s_la * pm.Normal("la_raw", 0, 1, shape=n_units)
        v = pm.Deterministic("v", pt.exp(lv))
        a = pm.Deterministic("a", pt.exp(la))
        if t0_mode == "estimate":
            mu_z = pm.Normal("mu_z", mu_z_mean, 1.0); s_z = pm.HalfNormal("s_z", 1.0)
            z = mu_z + s_z * pm.Normal("z_raw", 0, 1, shape=n_units)
            t0u = pm.Deterministic("t0", floor + (minrt - floor) * pm.math.sigmoid(z))
        else:
            t0u = pm.Deterministic("t0", pt.as_tensor_variable(np.full(n_units, floor)))
        pm.CustomDist("obs", v[uidx], a[uidx], t0u[uidx], logp=wald_logp, observed=rt)
        idata = pm.sample(draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": True})
    return idata


def compare_t0_modes(df: pd.DataFrame, effector: str, draws=600, tune=600, chains=2,
                     cores=1, status=lambda m: None) -> dict:
    """
    Fit the effector with t0 estimated vs t0 fixed at the floor and compare by
    LOO. Returns the ArviZ comparison table and the preferred model.
    """
    import arviz as az
    floor = PHYSIO_FLOOR[effector]
    mu_z_mean = 0.5 if effector == "hand" else 0.0
    units, rt, uidx, minrt = _build_units(df, effector)
    idatas = {}
    for mode in ("estimate", "fixed"):
        status(f"{effector}: fitting t0-{mode} model for LOO")
        idatas[f"t0_{mode}"] = _fit_with_loglik(rt, uidx, minrt, len(units), floor,
                                                t0_mode=mode, mu_z_mean=mu_z_mean,
                                                draws=draws, tune=tune, chains=chains,
                                                cores=cores)
    cmp = az.compare(idatas)
    tbl = cmp.reset_index().rename(columns={"index": "model"})
    keep = [c for c in ["model", "rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]
            if c in tbl.columns]
    return {"table": tbl[keep], "preferred": str(cmp.index[0]),
            "note": "Higher elpd_loo is better; elpd_diff is relative to the best model."}
