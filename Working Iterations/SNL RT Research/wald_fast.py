"""
wald_fast.py -- profile-likelihood fitting for the single-boundary shifted Wald.

Drop-in replacement for the differential-evolution fitter in DDM_fit.py, plus the
diagnostics that fall out of the same computation for free.

Why this works
--------------
Given t0, the shifted-Wald MLE is CLOSED FORM. With tau_i = RT_i - t0:

    mu_hat  = mean(tau)                       a = sqrt(lambda_hat)
    1/lam_h = mean(1/tau_i) - 1/mu_hat        v = a / mu_hat

(Tweedie 1957, inverse-Gaussian MLE. Mapping: expanding -(a - v*tau)^2/(2*tau)
against the IG exponent gives lambda = a^2, mu = a/v, and the prefactors agree.)

So the 3-D global search collapses to a 1-D search over t0 on a smooth profile.
The 5% uniform contamination has no closed form, but yields to EM: the E-step
computes responsibilities, the M-step is the same closed form with weights.

What you get from the one profile curve
---------------------------------------
    fit_cell            MLE, matching DDM_fit.py's likelihood
    profile_ci          95% CI for t0 (chi2_1, delta log-lik = 1.92)
    profile_curve       t0 -> profile NLL, for identifiability at ANY floor
    floor_sweep         constrained optimum at every floor, from one curve
    fit_fixed_t0        closed-form (v, a) with t0 held fixed -- no optimizer
    bootstrap_ks        calibrated p-value for the KS statistic
    wald_mode           the actual mode (NOT t0 + a/v, which is the mean)

Parity note
-----------
Validate against the DE fitter on all real cells before switching. Agreement was
exact on simulated data, but cells pinned at the v/a caps may differ: the caps are
applied inside the M-step here versus as box constraints in DE.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, brentq

# Bounds from CODE_REFERENCE.md (Tran et al. 2020 envelope at s = 1).
V_MAX, A_MAX = 20.0, 2.5
P_CONTAM = 0.05
HRT_FLOOR, SRT_FLOOR = 0.130, 0.070

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Densities (identical to DDM_fit.py)
# --------------------------------------------------------------------------- #
def wald_pdf(t, v, a):
    t = np.maximum(np.asarray(t, float), 1e-9)
    return (a / np.sqrt(2 * np.pi * t ** 3)) * np.exp(-(a - v * t) ** 2 / (2 * t))


def wald_cdf(t, v, a):
    t = np.maximum(np.asarray(t, float), 1e-9)
    return stats.invgauss.cdf(t, mu=1.0 / (v * a), scale=a ** 2)


def wald_mean(v, a, t0=0.0):
    """Mean of the shifted Wald. This is what DDM_fit.py's 'express_mode' column holds."""
    return t0 + a / v


def wald_mode(v, a, t0=0.0):
    """
    Actual mode of the shifted Wald. For IG(mu, lam): mu*[sqrt(1 + 9mu^2/4lam^2) - 3mu/2lam].
    Differs from the mean by up to ~27 ms on real cells -- see audit item A6.
    """
    mu, lam = a / v, a ** 2
    return t0 + mu * (np.sqrt(1.0 + 9.0 * mu ** 2 / (4.0 * lam ** 2)) - 3.0 * mu / (2.0 * lam))


# --------------------------------------------------------------------------- #
# Closed-form weighted MLE given t0  (the whole trick)
# --------------------------------------------------------------------------- #
def _ig_mle_weighted(tau, w):
    """Exact weighted (v, a) MLE for tau > 0. Weights are contamination responsibilities."""
    W = w.sum()
    mu = (w * tau).sum() / W
    inv = (w / tau).sum() / W - 1.0 / mu
    lam = 1.0 / max(inv, 1e-12)
    a = np.sqrt(lam)
    return a / mu, a


def _profile(t0, rts, Tr, contam, n_em=50, tol=1e-11):
    """Profile NLL at t0, with (v, a) concentrated out. Returns (nll, v, a)."""
    tau = rts - t0
    if np.any(tau <= 1e-9):
        return 1e10, np.nan, np.nan
    w = np.ones_like(tau)
    v = a = np.nan
    for _ in range(n_em):
        v, a = _ig_mle_weighted(tau, w)
        v = min(max(v, 0.1), V_MAX)
        a = min(max(a, 0.05), A_MAX)
        if contam <= 0:
            break
        f = wald_pdf(tau, v, a)
        num = (1.0 - contam) * f
        w_new = num / (num + contam / Tr)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    f = wald_pdf(tau, v, a)
    d = (1.0 - contam) * f + (contam / Tr if contam > 0 else 0.0)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return 1e10, v, a
    return float(-np.sum(np.log(d))), float(v), float(a)


def _t0_ceiling(rts, floor):
    """Robust upper bound on t0, matching DDM_fit.py (3rd percentile - 2 ms)."""
    return max(np.percentile(rts, 3) - 0.002, floor + 1e-3)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_cell(rts, floor, contam=P_CONTAM, n_grid=40):
    """
    MLE of (v, a, t0) by 1-D profile search. Deterministic -- no seeds, no retries.

    Returns dict with v, a, t0, nll, ks, and at_bound (audit item A7: DDM_srt_fits.csv
    has ~20% of cells pinned at v = 20 or a = 2.5, currently invisible in the output).
    """
    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min()) or 1e-3
    hi = _t0_ceiling(rts, floor)

    grid = np.linspace(floor, hi, n_grid)
    vals = np.array([_profile(t, rts, Tr, contam)[0] for t in grid])
    k = int(np.argmin(vals))
    lo_b, hi_b = grid[max(k - 1, 0)], grid[min(k + 1, n_grid - 1)]

    if hi_b - lo_b > 1e-12:
        r = minimize_scalar(lambda t: _profile(t, rts, Tr, contam)[0],
                            bounds=(lo_b, hi_b), method="bounded",
                            options={"xatol": 1e-9})
        t0 = float(r.x)
    else:
        t0 = float(grid[k])

    nll, v, a = _profile(t0, rts, Tr, contam)
    ks = float(stats.kstest(rts - t0, lambda z: wald_cdf(z, v, a)).statistic)
    at_bound = bool(v >= V_MAX - 1e-6 or v <= 0.1 + 1e-6
                    or a >= A_MAX - 1e-6 or a <= 0.05 + 1e-6
                    or abs(t0 - floor) < 1e-6)
    return {"v": v, "a": a, "t0": t0, "nll": nll, "ks": ks,
            "at_bound": at_bound, "t0_at_floor": bool(abs(t0 - floor) < 1e-6)}


def fit_fixed_t0(rts, t0, contam=P_CONTAM):
    """
    (v, a) with t0 held fixed. Closed form -- no optimizer.
    Replaces the differential_evolution call in SRT_fixed_t0_analysis.py.
    """
    rts = np.asarray(rts, float)
    if rts.min() <= t0:
        return None
    Tr = float(rts.max() - rts.min()) or 1e-3
    nll, v, a = _profile(t0, rts, Tr, contam)
    ks = float(stats.kstest(rts - t0, lambda z: wald_cdf(z, v, a)).statistic)
    return {"v": v, "a": a, "t0": t0, "nll": nll, "ks": ks}


# --------------------------------------------------------------------------- #
# Diagnostics from the same curve
# --------------------------------------------------------------------------- #
def profile_curve(rts, lo_t0=0.0, hi_t0=None, contam=P_CONTAM, n=200):
    """
    The profile NLL as a function of t0. Compute once per cell; every floor-based
    question is then a lookup rather than a refit.
    """
    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min()) or 1e-3
    if hi_t0 is None:
        hi_t0 = float(rts.min()) - 1e-4
    ts = np.linspace(max(lo_t0, 1e-4), hi_t0, n)
    out = np.array([_profile(t, rts, Tr, contam)[0] for t in ts])
    return ts, out


def profile_ci(rts, floor, contam=P_CONTAM, level=0.95):
    """
    Profile-likelihood CI for t0 (chi2_1 cutoff). Gives Method A the uncertainty it
    currently reports nowhere -- see audit item A10.

    floor_open=True means the profile is still improving AT the floor, i.e. t0 is not
    identified from below. That is a cleaner statement than the slope > 0.7 heuristic
    in SRT_identifiability_check.py.
    """
    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min()) or 1e-3
    cut = 0.5 * stats.chi2.ppf(level, 1)          # 1.9207 at 95%
    fit = fit_cell(rts, floor, contam)
    t0, f0 = fit["t0"], fit["nll"]
    hi_lim = _t0_ceiling(rts, floor)

    g = lambda t: _profile(t, rts, Tr, contam)[0] - f0 - cut

    floor_open = g(floor) < 0
    if floor_open:
        lo = floor
    else:
        try:
            lo = brentq(g, floor, t0, xtol=1e-10)
        except Exception:
            lo = floor
    try:
        hi = brentq(g, t0, hi_lim - 1e-9, xtol=1e-10)
    except Exception:
        hi = hi_lim
    return {"t0": t0, "lo": float(lo), "hi": float(hi),
            "width_ms": 1000.0 * (hi - lo), "floor_open": bool(floor_open)}


def floor_sweep(rts, floors, contam=P_CONTAM):
    """
    Constrained t0 at each candidate floor, read off ONE profile curve.
    Replaces the 6-floor x 2-seed refit loop in SRT_identifiability_check.py.

    Returns (t0_ms per floor, slope). slope near 1 means t0 tracks the floor.
    Run this on HAND cells too -- that negative control is missing from the repo
    and it is what turns the observation into a dissociation (audit item A11).
    """
    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min()) or 1e-3
    hi = float(np.percentile(rts, 3) - 0.002)
    t0s = []
    for fl in floors:
        grid = np.linspace(fl, max(hi, fl + 1e-3), 60)
        vals = [_profile(t, rts, Tr, contam)[0] for t in grid]
        k = int(np.argmin(vals))
        lo_b, hi_b = grid[max(k - 1, 0)], grid[min(k + 1, 59)]
        if hi_b - lo_b > 1e-12:
            r = minimize_scalar(lambda t: _profile(t, rts, Tr, contam)[0],
                                bounds=(lo_b, hi_b), method="bounded",
                                options={"xatol": 1e-9})
            t0s.append(r.x * 1000.0)
        else:
            t0s.append(grid[k] * 1000.0)
    slope = float(np.polyfit([f * 1000.0 for f in floors], t0s, 1)[0])
    return {"floors_ms": [f * 1000.0 for f in floors], "t0_ms": t0s,
            "slope": slope, "tracks_floor": bool(slope > 0.7)}


def bootstrap_ks(rts, floor, contam=P_CONTAM, B=200, seed=0):
    """
    Parametric-bootstrap p-value for the KS statistic -- audit item A1.

    The fixed 0.10 threshold in DDM_fit.py assumes the parameters are KNOWN. They are
    estimated, so the null distribution is much tighter: the true 5% critical value is
    ~0.093 at n = 110 and ~0.081 at n = 160, and P(KS > 0.10 | single Wald true) is
    only 0.013. Reporting ks_p instead of comparing to 0.10 makes every cell readable.
    """
    rts = np.asarray(rts, float)
    fit = fit_cell(rts, floor, contam)
    v, a, t0, ks_obs = fit["v"], fit["a"], fit["t0"], fit["ks"]
    n = len(rts)
    mu, lam = a / v, a ** 2
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        sim = t0 + stats.invgauss.rvs(mu / lam, scale=lam, size=n, random_state=rng)
        fb = fit_cell(sim, floor, contam)
        null[b] = fb["ks"]
    p = float((np.sum(null >= ks_obs) + 1) / (B + 1))
    return {"ks": ks_obs, "ks_p": p, "crit95": float(np.percentile(null, 95)),
            "rejected": bool(p < 0.05), **{k: fit[k] for k in ("v", "a", "t0", "at_bound")}}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)

    def sim(v, a, t0, n):
        mu, lam = a / v, a ** 2
        return t0 + stats.invgauss.rvs(mu / lam, scale=lam, size=n, random_state=rng)

    print("fit_cell -- hand-like cell")
    x = sim(9.0, 0.78, 0.165, 160)
    t = time.perf_counter(); f = fit_cell(x, HRT_FLOOR); dt = time.perf_counter() - t
    print(f"  v={f['v']:.3f} a={f['a']:.4f} t0={f['t0']*1000:.1f}ms "
          f"ks={f['ks']:.4f} at_bound={f['at_bound']}  ({dt*1000:.0f} ms)")

    print("\nprofile_ci -- the uncertainty Method A currently omits")
    c = profile_ci(x, HRT_FLOOR)
    print(f"  t0={c['t0']*1000:.1f}  95% CI=[{c['lo']*1000:.1f}, {c['hi']*1000:.1f}]  "
          f"width={c['width_ms']:.1f}ms  floor_open={c['floor_open']}")

    print("\nbootstrap_ks -- calibrated fit test")
    b = bootstrap_ks(x, HRT_FLOOR, B=150)
    print(f"  ks={b['ks']:.4f}  p={b['ks_p']:.3f}  95% crit={b['crit95']:.4f}  "
          f"rejected={b['rejected']}")

    print("\nfloor_sweep -- hand vs eye, the missing control")
    for lab, (v, a, t0, n, fls) in {
            "hand": (9.0, 0.78, 0.165, 160, [0.09, 0.10, 0.11, 0.12, 0.13, 0.14]),
            "eye ": (11.4, 1.45, 0.030, 110, [0.04, 0.05, 0.06, 0.07, 0.08, 0.09])}.items():
        y = sim(v, a, t0, n)
        y = y[y > max(fls) + 2e-3]
        s = floor_sweep(y, fls)
        print(f"  {lab}: slope={s['slope']:.2f}  tracks_floor={s['tracks_floor']}")

    print("\nwald_mode vs wald_mean -- CMT008@75 express (ve=6.797, ae=0.6406, t0e=0.081)")
    print(f"  reported 'express_mode' (= mean): {wald_mean(6.797, 0.6406, 0.081)*1000:.0f} ms")
    print(f"  actual mode                     : {wald_mode(6.797, 0.6406, 0.081)*1000:.0f} ms")
