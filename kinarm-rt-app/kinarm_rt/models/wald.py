"""
Single-boundary shifted-Wald models, matching the SNL-RT-Research pipeline.

Likelihood (single-barrier diffusion first-passage time, unit noise), exactly as
in the repository's Bayesian scripts and CODE_REFERENCE.md:

    log f(tau; v, a) = log a - 0.5*log(2*pi) - 1.5*log(tau) - (a - v*tau)^2 / (2*tau)

with tau = RT - t0. Parameters are drift v, boundary a, and non-decision time t0.

Three fitting paths mirror the pipeline:
  * fit_pooled_hierarchical  -- one hierarchical model with partial pooling across
                                participant x speed units (used for the hand across
                                all units, and for unimodal saccade cells per speed)
  * fit_mixture_cell         -- a two-component express/regular Wald mixture for a
                                bimodal saccade cell
  * mle_preview              -- fast per-cell maximum likelihood, for an instant look

All Bayesian entry points import PyMC lazily, so the module (and the app's preview
and LATER features) still work when PyMC is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._speeds import (SPEEDS, PHYSIO_FLOOR, V_MAX, A_MAX, MIX_SHIFT_FLOOR)

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# NumPy Wald density / CDF (for previews and goodness-of-fit)
# --------------------------------------------------------------------------- #
def wald_pdf(t, v, a):
    t = np.maximum(np.asarray(t, float), 1e-9)
    return (a / np.sqrt(2 * np.pi * t ** 3)) * np.exp(-(a - v * t) ** 2 / (2 * t))


def wald_cdf(t, v, a):
    from scipy import stats
    t = np.maximum(np.asarray(t, float), 1e-9)
    return stats.invgauss.cdf(t, mu=1.0 / (v * a), scale=a ** 2)


# --------------------------------------------------------------------------- #
# Bimodality detection (express vs regular saccades)
# --------------------------------------------------------------------------- #
def detect_bimodal(rts_s: np.ndarray) -> bool:
    """
    True if a cell's RTs look bimodal (an express + a regular population).
    Prefers Hartigan's dip test; falls back to a Gaussian-mixture BIC comparison;
    if neither library is present, returns False (treat as unimodal).
    """
    x = np.asarray(rts_s, float)
    x = x[np.isfinite(x)]
    if x.size < 40:
        return False
    try:
        import diptest
        return bool(diptest.diptest(x * 1000.0)[1] < 0.05)
    except Exception:
        pass
    try:
        from sklearn.mixture import GaussianMixture
        xr = (x * 1000.0).reshape(-1, 1)
        b1 = GaussianMixture(1, random_state=0).fit(xr).bic(xr)
        b2 = GaussianMixture(2, n_init=4, random_state=0).fit(xr).bic(xr)
        return bool(b2 < b1 - 10)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Fast per-cell MLE preview
# --------------------------------------------------------------------------- #
def _fit_cell_mle(rt: np.ndarray, floor: float, contamination: float = 0.0):
    """Fast (v, a, t0) MLE for one cell; bounds follow the frequentist DDM fit."""
    from scipy.optimize import minimize

    rtmin = float(rt.min())
    Tr = float(rt.max() - rt.min()) or 1e-3
    t0_hi = max(np.percentile(rt, 3) - 0.002, floor + 1e-3)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    def unpack(p):
        v = float(np.clip(np.exp(p[0]), 0.1, V_MAX))
        a = float(np.clip(np.exp(p[1]), 0.05, A_MAX))
        t0 = floor + (t0_hi - floor) * sig(p[2])
        return v, a, t0

    def nll(p):
        v, a, t0 = unpack(p)
        tau = rt - t0
        if np.any(tau <= 1e-6):
            return 1e12
        w = wald_pdf(tau, v, a)
        d = (1 - contamination) * w + (contamination / Tr if contamination > 0 else 0.0)
        if np.any(d <= 0) or not np.all(np.isfinite(d)):
            return 1e12
        return float(-np.sum(np.log(d)))

    best = None
    for t0f in (0.3, 0.6, 0.85):
        x0 = [np.log(6.0), np.log(0.8), np.log(t0f / (1 - t0f))]
        try:
            r = minimize(nll, x0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    return unpack(best.x)


def mle_preview(df: pd.DataFrame, effector: str, contamination: float = 0.0) -> dict:
    """Per-cell MLE preview across participant x speed cells (seconds, no sampling)."""
    sub = df[df["effector"] == effector]
    floor = PHYSIO_FLOOR[effector]
    rows = []
    for (p, c), g in sub.groupby(["participant", "condition"]):
        rt = g["rt"].values
        if len(rt) < 15:
            continue
        v, a, t0 = _fit_cell_mle(rt, floor, contamination)
        rows.append({"participant": p, "condition": int(c), "speed": SPEEDS[int(c)],
                     "v": v, "a": a, "t0_ms": t0 * 1000.0,
                     "floored": t0 * 1000.0 <= floor * 1000.0 + 1.0})
    cell = pd.DataFrame(rows)
    group = (cell.groupby("condition")
                 .agg(v=("v", "mean"), a=("a", "mean"), t0_ms=("t0_ms", "mean"),
                      pct_floored=("floored", "mean")).reset_index())
    group["speed"] = group["condition"].map(lambda c: SPEEDS[int(c)])
    group["pct_floored"] *= 100
    return {"cell": cell, "group": group, "floor_ms": floor * 1000.0}


# --------------------------------------------------------------------------- #
# Hierarchical Bayesian (pooled single-Wald)
# --------------------------------------------------------------------------- #
def _build_units(df: pd.DataFrame, effector: str, speeds=None):
    """Return (units, rt, uidx, minrt) pooled across the requested speeds."""
    sub = df[df["effector"] == effector]
    speeds = SPEEDS if speeds is None else speeds
    units, rt, uidx, minrt, u = [], [], [], [], 0
    for c in range(len(SPEEDS)):
        if SPEEDS[c] not in speeds:
            continue
        for pid in sorted(sub["participant"].unique()):
            x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
            x = x[np.isfinite(x)]
            if len(x) < 15:
                continue
            units.append((pid, int(c)))
            minrt.append(float(x.min()))
            rt.extend(x.tolist())
            uidx.extend([u] * len(x))
            u += 1
    return units, np.array(rt), np.array(uidx), np.array(minrt)


def fit_pooled_hierarchical(rt, uidx, minrt, n_units, floor, mu_z_mean=0.5,
                            draws=1000, tune=1000, chains=4, target_accept=0.95,
                            cores=1, seed=7, contamination=0.0, progressbar=False):
    """
    Hierarchical single-Wald with partial pooling across units, matching the
    repository's `hierarchical_single`. Returns (idata, per-unit posterior arrays).
    """
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as m:
        mu_lv = pm.Normal("mu_lv", np.log(10), 0.5); s_lv = pm.HalfNormal("s_lv", 0.5)
        mu_la = pm.Normal("mu_la", np.log(1.0), 0.5); s_la = pm.HalfNormal("s_la", 0.5)
        mu_z = pm.Normal("mu_z", mu_z_mean, 1.0); s_z = pm.HalfNormal("s_z", 1.0)
        lv = mu_lv + s_lv * pm.Normal("lv_raw", 0, 1, shape=n_units)
        la = mu_la + s_la * pm.Normal("la_raw", 0, 1, shape=n_units)
        z = mu_z + s_z * pm.Normal("z_raw", 0, 1, shape=n_units)
        v = pm.Deterministic("v", pt.exp(lv))
        a = pm.Deterministic("a", pt.exp(la))
        t0 = pm.Deterministic("t0", floor + (minrt - floor) * pm.math.sigmoid(z))
        tau = rt - t0[uidx]
        logp = (pt.log(a[uidx]) - 0.5 * _LOG2PI - 1.5 * pt.log(tau)
                - (a[uidx] - v[uidx] * tau) ** 2 / (2 * tau))
        if contamination > 0:
            Tr = float(rt.max() - rt.min())
            logp = pt.logaddexp(np.log1p(-contamination) + logp,
                                np.log(contamination) - np.log(Tr))
        pm.Potential("lik", logp.sum())
        idata = pm.sample(draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar,
                          idata_kwargs={"log_likelihood": False})
    po = idata.posterior
    arr = {
        "v": po["v"].mean(("chain", "draw")).values,
        "a": po["a"].mean(("chain", "draw")).values,
        "t0": po["t0"].mean(("chain", "draw")).values * 1000.0,
        "t0_lo": po["t0"].quantile(0.025, ("chain", "draw")).values * 1000.0,
        "t0_hi": po["t0"].quantile(0.975, ("chain", "draw")).values * 1000.0,
    }
    return idata, arr


def fit_mixture_cell(rts, draws=1000, tune=1000, chains=4, target_accept=0.95,
                     cores=1, seed=11, progressbar=False):
    """
    Two-component express/regular Wald mixture for one bimodal saccade cell,
    matching the repository's `bayesian_mixture`. Returns a summary dict.
    """
    import pymc as pm
    import pytensor.tensor as pt

    minrt = float(rts.min())

    def wlog(tau, v, a):
        return pt.log(a) - 0.5 * _LOG2PI - 1.5 * pt.log(tau) - (a - v * tau) ** 2 / (2 * tau)

    with pm.Model() as m:
        pi = pm.Beta("pi", 2, 2)
        v1 = pm.LogNormal("v1", np.log(15), 0.6); a1 = pm.LogNormal("a1", np.log(1.0), 0.6)
        v2 = pm.LogNormal("v2", np.log(12), 0.6); a2 = pm.LogNormal("a2", np.log(1.2), 0.6)
        z1 = pm.Normal("z1", 0, 1); z2 = pm.Normal("z2", 0, 1)
        t01 = pm.Deterministic("t01", MIX_SHIFT_FLOOR + (minrt - MIX_SHIFT_FLOOR) * pm.math.sigmoid(z1) * 0.98)
        t02 = pm.Deterministic("t02", MIX_SHIFT_FLOOR + (minrt - MIX_SHIFT_FLOOR) * pm.math.sigmoid(z2) * 0.98)
        lp1 = pt.log(pi) + wlog(rts - t01, v1, a1)
        lp2 = pt.log(1 - pi) + wlog(rts - t02, v2, a2)
        pm.Potential("lik", pt.sum(pm.math.logsumexp(pt.stack([lp1, lp2], axis=0), axis=0)))
        idata = pm.sample(draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar,
                          idata_kwargs={"log_likelihood": False})

    po = idata.posterior
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0

    def cd(n):
        return po[n].values
    pi_, v1_, a1_, t01_, v2_, a2_, t02_ = [cd(n) for n in ["pi", "v1", "a1", "t01", "v2", "a2", "t02"]]
    m1 = t01_ + a1_ / v1_
    m2 = t02_ + a2_ / v2_
    exp_is1 = m1 <= m2
    em = np.where(exp_is1, m1, m2).reshape(-1) * 1000.0
    rm = np.where(exp_is1, m2, m1).reshape(-1) * 1000.0
    pe = np.where(exp_is1, pi_, 1 - pi_).reshape(-1)

    def ci(x):
        return (round(float(np.mean(x)), 3), round(float(np.percentile(x, 2.5)), 3),
                round(float(np.percentile(x, 97.5)), 3))
    pim, pil, pih = ci(pe)
    emm, eml, emh = ci(em)
    rmm, rml, rmh = ci(rm)
    return {"model": "mixture", "conv_div": div,
            "pi": pim, "pi_lo95": pil, "pi_hi95": pih,
            "express_mode": round(emm), "express_mode_lo95": round(eml), "express_mode_hi95": round(emh),
            "reg_mode": round(rmm), "reg_mode_lo95": round(rml), "reg_mode_hi95": round(rmh)}


# --------------------------------------------------------------------------- #
# Orchestration: fit a whole effector the way the pipeline does
# --------------------------------------------------------------------------- #
def fit_effector(df: pd.DataFrame, effector: str, draws=1000, tune=1000, chains=4,
                 target_accept=0.95, cores=1, contamination=0.0, use_mixture=True,
                 progressbar=False, status=lambda msg: None):
    """
    Fit one effector following the repository's structure and return a result dict:

        units       : DataFrame [participant, condition, speed, v, a, t0_ms, t0_lo95,
                      t0_hi95, model]
        group       : per-speed means of v, a, t0 (and % floored)
        mixture     : DataFrame of express/regular mixture cells (saccades only)
        idata       : the hand posterior, or {speed: posterior} for saccades
        convergence : {max_rhat, n_divergences, converged}

    Hand: one pooled hierarchical fit across all units.
    Saccade: per speed, a pooled hierarchical fit over unimodal cells plus a
    two-component mixture for each bimodal cell.
    """
    import arviz as az
    floor = PHYSIO_FLOOR[effector]
    mu_z_mean = 0.5 if effector == "hand" else 0.0
    sub = df[df["effector"] == effector]

    unit_rows, mixture_rows = [], []
    idata_store = {}
    rhats, divs = [], []

    def summarize_units(units, arr, model="single"):
        for i, (pid, c) in enumerate(units):
            unit_rows.append({
                "participant": pid, "condition": int(c), "speed": SPEEDS[int(c)],
                "v": round(float(arr["v"][i]), 3), "a": round(float(arr["a"][i]), 4),
                "t0_ms": round(float(arr["t0"][i])), "t0_lo95": round(float(arr["t0_lo"][i])),
                "t0_hi95": round(float(arr["t0_hi"][i])), "model": model})

    if effector == "hand":
        units, rt, uidx, minrt = _build_units(sub, effector)
        status(f"Hand: pooled hierarchical fit over {len(units)} units, {len(rt)} trials")
        idata, arr = fit_pooled_hierarchical(rt, uidx, minrt, len(units), floor,
                                             mu_z_mean=mu_z_mean, draws=draws, tune=tune,
                                             chains=chains, target_accept=target_accept,
                                             cores=cores, contamination=contamination,
                                             progressbar=progressbar)
        summarize_units(units, arr, "single")
        idata_store["all"] = idata
        rh = az.rhat(idata, var_names=["v", "a", "t0"])
        rhats.append(float(max(rh["v"].max(), rh["a"].max(), rh["t0"].max())))
        divs.append(int(idata.sample_stats["diverging"].sum()))
    else:
        for c in range(len(SPEEDS)):
            spd = SPEEDS[c]
            pids = sorted(sub["participant"].unique())
            uni, bim = [], []
            for pid in pids:
                x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
                x = x[np.isfinite(x)]
                if len(x) < 15:
                    continue
                (bim if (use_mixture and detect_bimodal(x)) else uni).append(pid)
            # unimodal: one pooled hierarchical fit for this speed
            if uni:
                units, rt, uidx, minrt = [], [], [], []
                for u_i, pid in enumerate(uni):
                    x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
                    x = x[np.isfinite(x)]
                    units.append((pid, c)); minrt.append(float(x.min()))
                    rt.extend(x.tolist()); uidx.extend([u_i] * len(x))
                status(f"Saccade {int(spd)} deg/s: {len(uni)} unimodal cells, {len(rt)} trials")
                idata, arr = fit_pooled_hierarchical(
                    np.array(rt), np.array(uidx), np.array(minrt), len(units), floor,
                    mu_z_mean=mu_z_mean, draws=draws, tune=tune, chains=chains,
                    target_accept=target_accept, cores=cores, contamination=contamination,
                    progressbar=progressbar)
                summarize_units(units, arr, "single")
                idata_store[int(spd)] = idata
                rh = az.rhat(idata, var_names=["v", "a", "t0"])
                rhats.append(float(max(rh["v"].max(), rh["a"].max(), rh["t0"].max())))
                divs.append(int(idata.sample_stats["diverging"].sum()))
            # bimodal: a mixture per cell
            for pid in bim:
                x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
                x = x[np.isfinite(x)]
                status(f"Saccade {int(spd)} deg/s: express/regular mixture for {pid}")
                mres = fit_mixture_cell(x, draws=draws, tune=tune, chains=chains,
                                        target_accept=target_accept, cores=cores,
                                        progressbar=progressbar)
                mres.update({"participant": pid, "condition": c, "speed": spd, "n": len(x)})
                mixture_rows.append(mres)
                divs.append(mres["conv_div"])

    units_df = pd.DataFrame(unit_rows)
    if len(units_df):
        group = (units_df.groupby("condition")
                 .agg(v=("v", "mean"), a=("a", "mean"), t0_ms=("t0_ms", "mean")).reset_index())
        group["speed"] = group["condition"].map(lambda c: SPEEDS[int(c)])
        group["t0_floor_ms"] = floor * 1000.0
        group["pct_floored"] = [
            100.0 * np.mean(np.abs(units_df[units_df.condition == c]["t0_ms"] - floor * 1000) < 2)
            for c in group["condition"]]
    else:
        group = pd.DataFrame()
    max_rhat = float(max(rhats)) if rhats else float("nan")
    n_div = int(sum(divs))
    return {"effector": effector, "units": units_df, "group": group,
            "mixture": pd.DataFrame(mixture_rows), "idata": idata_store,
            "convergence": {"max_rhat": max_rhat, "n_divergences": n_div,
                            "converged": (rhats and max_rhat < 1.01 and n_div == 0)}}
