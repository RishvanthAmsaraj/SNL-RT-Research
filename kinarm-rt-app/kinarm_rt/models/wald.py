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

import os


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
# Frequentist per-cell fit -- a literal port of the repository's DDM_fit.py
#
# These three functions are the same code the standalone script runs, so the app
# reports the same numbers as running DDM_fit.py in an IDE. Keep them in step: the
# optimiser, its seeds, iteration cap and tolerance, the bounds, and the mixture
# acceptance rule all affect the fitted values.
# --------------------------------------------------------------------------- #
DE_SEEDS = (42, 7)
DE_MAXITER = 400
DE_TOL = 1e-9
DE_POPSIZE = 12
MIX_DE_POPSIZE = 14
MIN_TRIALS = 15
KS_ACCEPT = 0.10          # a single Wald is kept unless its KS exceeds this
MIX_MIN_PI, MIX_MAX_PI = 0.10, 0.90
MIX_MIN_MODE_GAP_MS = 30.0


def ddm_fit_single(rts, floor, contam=0.0):
    """Single shifted-Wald by differential evolution. Returns (x, nll, ks).

    Port of DDM_fit.py::fit_single.
    """
    from scipy import stats
    from scipy.optimize import differential_evolution

    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min())

    def nll(p):
        v, a, t0 = p
        adj = rts - t0
        if np.any(adj <= 0):
            return 1e10
        w = wald_pdf(adj, v, a)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            return 1e10
        d = (1 - contam) * w + (contam / Tr if contam > 0 else 0.0)
        if np.any(d <= 0):
            return 1e10
        return -np.sum(np.log(d))

    b = [(0.1, V_MAX), (0.05, A_MAX),
         (floor, max(np.percentile(rts, 3) - 0.002, floor + 1e-3))]
    best = None
    for s in DE_SEEDS:
        r = differential_evolution(nll, b, seed=s, maxiter=DE_MAXITER, tol=DE_TOL,
                                   popsize=DE_POPSIZE, polish=True)
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    adj = rts - x[2]
    ks = float(stats.kstest(adj, lambda z: wald_cdf(z, x[0], x[1])).statistic)
    return x, float(best.fun), ks


def _mix_nll(p, rts, Tr, contam):
    """Port of DDM_fit.py::_mix_nll."""
    pi, ve, ae, t0e, vr, ar, t0r = p
    ee = rts - t0e
    rr = rts - t0r
    if np.any(ee <= 0) or np.any(rr <= 0):
        return 1e10
    if (t0e + ae / ve) >= (t0r + ar / vr):      # express mean must precede regular
        return 1e10
    core = pi * wald_pdf(ee, ve, ae) + (1 - pi) * wald_pdf(rr, vr, ar)
    d = (1 - contam) * core + (contam / Tr if contam > 0 else 0.0)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return 1e10
    return -np.sum(np.log(d))


def _moments_to_wald(mean, sd, floor):
    """Port of DDM_fit.py::_moments_to_wald."""
    t0 = min(max(floor, mean - 2.5 * sd), mean - 1e-3)
    mu = mean - t0
    var = max(sd ** 2, 1e-6)
    lam = mu ** 3 / var
    a = np.sqrt(lam)
    v = a / mu
    return [np.clip(v, 0.1, V_MAX), np.clip(a, 0.05, A_MAX), np.clip(t0, floor, mean - 1e-3)]


def ddm_fit_mixture(rts, floor, contam=0.0):
    """Two-component express/regular Wald mixture. Returns (x, nll, ks).

    Port of DDM_fit.py::fit_mixture.
    """
    from scipy import stats
    from scipy.optimize import differential_evolution, minimize

    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min())
    mn = float(rts.min())
    cands = []
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(2, n_init=8, random_state=0).fit(rts.reshape(-1, 1))
        o = np.argsort(gm.means_.ravel())
        m1, m2 = gm.means_.ravel()[o]
        s1, s2 = np.sqrt(gm.covariances_.ravel()[o])
        w1 = gm.weights_[o][0]
        cands.append([np.clip(w1, 0.05, 0.95)]
                     + _moments_to_wald(m1, s1, floor) + _moments_to_wald(m2, s2, floor))
    except Exception:
        pass
    for pe, pr, piw in [(15, 65, 0.5), (25, 70, 0.4), (10, 55, 0.6), (30, 75, 0.5)]:
        me, mr = np.percentile(rts, pe), np.percentile(rts, pr)
        if mr - me < 0.005:
            continue
        cands.append([piw] + _moments_to_wald(me, (mr - me) / 3, floor)
                     + _moments_to_wald(mr, (rts.max() - mr) / 3, floor))
    b = [(0.02, 0.98), (0.1, V_MAX), (0.05, A_MAX), (floor, mn - 1e-3),
         (0.1, V_MAX), (0.05, A_MAX), (floor, mn - 1e-3)]
    best = None
    for c in cands:
        try:
            r = minimize(_mix_nll, c, args=(rts, Tr, contam), method="L-BFGS-B", bounds=b)
            if r.fun < 1e9 and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass
    rde = differential_evolution(_mix_nll, b, args=(rts, Tr, contam), seed=42,
                                 maxiter=DE_MAXITER, tol=DE_TOL, popsize=MIX_DE_POPSIZE,
                                 polish=True)
    if best is None or rde.fun < best.fun:
        best = rde
    x = best.x
    pi = x[0]

    def mc(z):
        return pi * wald_cdf(z - x[3], x[1], x[2]) + (1 - pi) * wald_cdf(z - x[6], x[4], x[5])

    ks = float(stats.kstest(rts, mc).statistic)
    return x, float(best.fun), ks


def ddm_select_srt(rts, floor, contam=0.0):
    """Choose single vs express/regular mixture for one saccade cell.

    Port of the selection rule in DDM_fit.py::main: the mixture is only attempted
    where the single Wald fails (KS > 0.10), and is only accepted if it fits well,
    splits the trials sensibly, and separates the two modes by at least 30 ms.
    Returns a dict with the chosen model and both fits' diagnostics.
    """
    xs, _, ks_s = ddm_fit_single(rts, floor, contam)
    out = {"ks_single": ks_s, "model": "single", "v": xs[0], "a": xs[1], "t0": xs[2],
           "ks": ks_s, "mixture": None}
    if ks_s <= KS_ACCEPT:
        return out
    try:
        xm, _, ks_m = ddm_fit_mixture(rts, floor, contam)
    except Exception:
        return out
    pi = xm[0]
    em = (xm[3] + xm[2] / xm[1]) * 1000.0
    rm = (xm[6] + xm[5] / xm[4]) * 1000.0
    if (ks_m < KS_ACCEPT) and (MIX_MIN_PI <= pi <= MIX_MAX_PI) \
            and ((rm - em) >= MIX_MIN_MODE_GAP_MS):
        out.update({"model": "mixture", "ks": ks_m,
                    "mixture": {"pi": pi, "ve": xm[1], "ae": xm[2], "t0e": xm[3],
                                "vr": xm[4], "ar": xm[5], "t0r": xm[6],
                                "express_mode": em, "reg_mode": rm}})
    return out


# --------------------------------------------------------------------------- #
# Fast per-cell MLE preview
# --------------------------------------------------------------------------- #
def _fit_cell_mle(rt: np.ndarray, floor: float, contamination: float = 0.0):
    """(v, a, t0) for one cell, using the repository's DDM_fit.py optimiser."""
    x, _, _ = ddm_fit_single(rt, floor, contamination)
    return float(x[0]), float(x[1]), float(x[2])


# --------------------------------------------------------------------------- #
# Running many cell fits
#
# Each participant x speed cell is fitted independently and the optimiser is
# seeded per cell, so the result does not depend on how the work is scheduled.
# That makes the loop safe to run across cores, which matters because the
# two-component saccade fit costs roughly nine times a single fit.
# --------------------------------------------------------------------------- #
def _cell_worker(item):
    """Fit one cell. Module-level so it can be sent to a worker process."""
    key, rt, floor, contam, mode = item
    if mode == "select":
        return key, ddm_select_srt(rt, floor, contam)
    if mode == "mixture":
        x, nll, ks = ddm_fit_mixture(rt, floor, contam)
        return key, {"x": x, "nll": nll, "ks": ks}
    x, nll, ks = ddm_fit_single(rt, floor, contam)
    return key, {"model": "single", "v": x[0], "a": x[1], "t0": x[2],
                 "ks": ks, "ks_single": ks, "nll": nll, "mixture": None}


def default_jobs() -> int:
    """
    How many workers to fit cells with.

    Threads, not processes. Cell fits are independent, so they parallelise cleanly,
    but joblib's process backend starts fresh interpreters that re-import the main
    module -- and under `streamlit run` on Windows that re-runs the app script
    inside each worker, which has been seen both to hang a run and to briefly
    duplicate the interface. Threads cannot do either. Much of the optimiser's time
    is inside NumPy and SciPy, which release the interpreter lock, so threads still
    help; the result is identical to running the cells one after another.
    """
    try:
        return max(1, min(8, (os.cpu_count() or 1)))
    except Exception:
        return 1


def map_cells(items, n_jobs: int = -1, progress=None, offset: int = 0,
              grand_total: int | None = None):
    """
    Fit a list of cells, reporting progress as each finishes.

    `progress` is called as progress(done, total). `offset` and `grand_total` let a
    caller report against a larger span than this batch, so a bar does not restart
    when work is split into several calls.

    Cells are independent and seeded individually, so how the work is scheduled
    never affects the result; any failure falls back to running them in order.
    """
    items = list(items)
    total = len(items)
    span = grand_total if grand_total is not None else total
    out = {}
    if total == 0:
        return out
    if n_jobs is None or n_jobs < 0:
        n_jobs = default_jobs()

    if n_jobs > 1 and total > 1:
        try:
            from joblib import Parallel, delayed
            done = 0
            gen = Parallel(n_jobs=min(n_jobs, total), prefer="threads",
                           return_as="generator_unordered")(
                delayed(_cell_worker)(it) for it in items)
            for key, res in gen:
                out[key] = res
                done += 1
                if progress:
                    progress(offset + done, span)
            if len(out) == total:
                return out
            items = [it for it in items if it[0] not in out]
        except Exception:
            out = {}

    done = len(out)
    for it in items:
        key, res = _cell_worker(it)
        out[key] = res
        done += 1
        if progress:
            progress(offset + done, span)
    return out


def select_srt_cells(items, n_jobs: int = -1, progress=None, status=lambda m: None):
    """
    Choose single versus two-component for a batch of saccade cells, in two passes.

    Deciding cell by cell means a cheap single fit and an expensive two-component
    fit are interleaved unpredictably, and in real data the cells that need two
    components are clustered rather than spread out. That makes any progress bar
    based on cells misleading: the pace measured over the early cells does not
    describe the rest.

    Splitting the work fixes that without changing anything about the outcome. Pass
    one fits a single Wald to every cell; pass two fits a two-component model only
    to the cells the single fit failed. Every cell in a pass costs about the same,
    so the count and the time estimate mean what they appear to mean. The
    acceptance rule is applied exactly as in ddm_select_srt.
    """
    items = list(items)
    single_items = [(k, rt, fl, ct, "single") for (k, rt, fl, ct, _m) in items]
    status(f"pass 1 of 2 — single fits, {len(single_items)} cells")
    singles = map_cells(single_items, n_jobs=n_jobs, progress=progress)

    by_key = {k: (rt, fl, ct) for (k, rt, fl, ct, _m) in items}
    out = {}
    needs_mix = []
    for k, r in singles.items():
        out[k] = {"ks_single": r["ks"], "model": "single", "v": r["v"], "a": r["a"],
                  "t0": r["t0"], "ks": r["ks"], "mixture": None}
        if r["ks"] > KS_ACCEPT:
            needs_mix.append(k)

    if needs_mix:
        status(f"pass 2 of 2 — two-component fits, {len(needs_mix)} cells")
        mix_items = [(k, *by_key[k], "mixture") for k in needs_mix]
        mixes = map_cells(mix_items, n_jobs=n_jobs, progress=progress)
        for k, m in mixes.items():
            xm, ks_m = m["x"], m["ks"]
            pi = xm[0]
            em = (xm[3] + xm[2] / xm[1]) * 1000.0
            rm = (xm[6] + xm[5] / xm[4]) * 1000.0
            if (ks_m < KS_ACCEPT) and (MIX_MIN_PI <= pi <= MIX_MAX_PI) \
                    and ((rm - em) >= MIX_MIN_MODE_GAP_MS):
                out[k].update({"model": "mixture", "ks": ks_m,
                               "mixture": {"pi": pi, "ve": xm[1], "ae": xm[2], "t0e": xm[3],
                                           "vr": xm[4], "ar": xm[5], "t0r": xm[6],
                                           "express_mode": em, "reg_mode": rm}})
    return out


def mle_preview(df: pd.DataFrame, effector: str, contamination: float = 0.0,
                use_mixture: bool = True, n_jobs: int = -1, progress=None,
                selection: dict | None = None, status=lambda m: None) -> dict:
    """
    Per-cell maximum-likelihood fit across participant x speed cells (no sampling).

    This is Method A: the same optimiser, bounds and model-selection rule as
    DDM_fit.py. Hand cells get a single Wald; saccade cells get a single Wald
    unless it fits poorly, in which case the express/regular mixture is tried and
    accepted on the pipeline's criteria. For a mixture cell the reported t0 is the
    regular component's shift, which is the value NDT_barchart.py compares across
    cells.
    """
    sub = df[df["effector"] == effector]
    floor = PHYSIO_FLOOR[effector]
    mode = "select" if (effector == "eye" and use_mixture) else "single"
    items = []
    for (p, c), g in sub.groupby(["participant", "condition"]):
        rt = g["rt"].values.astype(float)
        rt = rt[np.isfinite(rt)]
        if len(rt) < MIN_TRIALS:
            continue
        items.append(((p, int(c)), rt, floor, contamination, mode))

    # Method B already ran this selection while deciding which cells need two
    # components. Reusing it avoids repeating the most expensive step of the run.
    if selection:
        fits = {k: selection[k] for (k, *_ ) in items if k in selection}
        missing = [it for it in items if it[0] not in fits]
        if progress:
            progress(len(fits), len(items))
        if missing:
            fits.update(map_cells(missing, n_jobs=n_jobs, progress=progress,
                                  offset=len(fits), grand_total=len(items)))
    elif mode == "select":
        fits = select_srt_cells(items, n_jobs=n_jobs, progress=progress, status=status)
    else:
        fits = map_cells(items, n_jobs=n_jobs, progress=progress)
    rows = []
    for (p, c), sel in fits.items():
        if sel["model"] == "mixture":
            m = sel["mixture"]
            v, a, t0 = m["vr"], m["ar"], m["t0r"]
        else:
            v, a, t0 = sel["v"], sel["a"], sel["t0"]
        rows.append({"participant": p, "condition": int(c), "speed": SPEEDS[int(c)],
                     "v": float(v), "a": float(a), "t0_ms": float(t0) * 1000.0,
                     "ks": float(sel["ks"]), "model": sel["model"],
                     "floored": float(t0) * 1000.0 <= floor * 1000.0 + 1.0})
    cell = pd.DataFrame(rows).sort_values(["participant", "condition"]).reset_index(drop=True) \
        if rows else pd.DataFrame(rows)
    if not len(cell):
        return {"cell": cell, "group": pd.DataFrame(), "floor_ms": floor * 1000.0}
    group = (cell.groupby("condition")
                 .agg(v=("v", "mean"), a=("a", "mean"), t0_ms=("t0_ms", "mean"),
                      median_ks=("ks", "median"), pct_floored=("floored", "mean")).reset_index())
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
                 progressbar=False, status=lambda msg: None,
                 n_jobs: int = -1, progress=None):
    """
    Fit one effector following the repository's structure and return a result dict:

        units       : DataFrame [participant, condition, speed, v, a, t0_ms, t0_lo95,
                      t0_hi95, model]
        group       : per-speed means of v, a, t0 (and % floored)
        mixture     : DataFrame of two-component saccade cells (saccades only)
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
    selection = {}
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
        if progress:
            progress(0, 1)
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
        if progress:
            progress(1, 1)
    else:
        # Which cells get a two-component fit is decided exactly as in the pipeline:
        # Bayesian_SRT_fit.py reads the selection from DDM_srt_fits.csv, i.e. the cells
        # where DDM_fit.py's rule chose a mixture. That rule is recomputed here on the
        # same data, so the app needs no CSV from a previous run but still splits the
        # cells the same way. Hartigan's dip test is only the fallback the script uses
        # when that table is absent.
        #
        # The selection is the expensive part -- a failing cell costs a full
        # two-component fit -- so it is done up front across cores. Cells are
        # independent and seeded individually, so this changes nothing.
        sel_items = []
        for c in range(len(SPEEDS)):
            for pid in sorted(sub["participant"].unique()):
                x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
                x = x[np.isfinite(x)]
                if len(x) >= MIN_TRIALS:
                    sel_items.append(((pid, c), x, floor, contamination, "select"))

        n_sel = len(sel_items)
        selection = {}
        if use_mixture and n_sel:
            status(f"choosing single vs two-component for {n_sel} cells")
            try:
                selection = select_srt_cells(sel_items, n_jobs=n_jobs,
                                             progress=progress, status=status)
            except Exception:
                selection = {}
        n_mix_total = sum(1 for v in selection.values() if v.get("model") == "mixture")
        status(f"{n_mix_total} of {n_sel} cells need two components")

        # Sampling is reported as its own counted phase rather than being folded in
        # with the selection above. The two do very different amounts of work per
        # step, so a single running total would make the time estimate lurch.
        sample_total = len(SPEEDS) + n_mix_total
        sampled = {"n": 0}

        def bump():
            sampled["n"] += 1
            if progress:
                progress(min(sampled["n"], sample_total), sample_total)

        for c in range(len(SPEEDS)):
            spd = SPEEDS[c]
            pids = sorted(sub["participant"].unique())
            uni, bim = [], []
            for pid in pids:
                x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
                x = x[np.isfinite(x)]
                if len(x) < MIN_TRIALS:
                    continue
                if not use_mixture:
                    is_mix = False
                elif (pid, c) in selection:
                    is_mix = selection[(pid, c)].get("model") == "mixture"
                else:
                    try:
                        is_mix = ddm_select_srt(x, floor, contamination)["model"] == "mixture"
                    except Exception:
                        is_mix = detect_bimodal(x)
                (bim if is_mix else uni).append(pid)
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
            bump()
            # cells needing two components: one fit each
            for pid in bim:
                x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
                x = x[np.isfinite(x)]
                status(f"Saccade {int(spd)} deg/s: two-component fit for {pid}")
                mres = fit_mixture_cell(x, draws=draws, tune=tune, chains=chains,
                                        target_accept=target_accept, cores=cores,
                                        progressbar=progressbar)
                mres.update({"participant": pid, "condition": c, "speed": spd, "n": len(x)})
                mixture_rows.append(mres)
                divs.append(mres["conv_div"])
                bump()

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
    converged = bool(rhats) and max_rhat < 1.01 and n_div == 0
    return {"effector": effector, "selection": selection if effector != "hand" else {},
            "units": units_df, "group": group,
            "mixture": pd.DataFrame(mixture_rows), "idata": idata_store,
            "convergence": {"max_rhat": max_rhat, "n_divergences": n_div,
                            "converged": converged}}
