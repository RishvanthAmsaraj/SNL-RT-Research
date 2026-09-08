"""
later.py -- the LATER model family, fit by maximum likelihood.

LATER says the decision variable rises linearly at a rate r drawn from a Gaussian,
so RT = 1/r. Three variants are implemented, because the comparison the professor
asked for is only fair if the parameter counts line up:

    later_mle           RT = 1/r,        r ~ N(mu, sigma)      2 params
    shifted_later_mle   RT = t0 + 1/r,   r ~ N(mu, sigma)      3 params
    truncated_later_mle RT = 1/r,        r ~ N(mu, sigma) | r>0  2 params

WHY THE SHIFTED VARIANT MATTERS
-------------------------------
Your pipeline fits LATER by OLS on reciprobit plotting positions, which is not a
likelihood -- so it cannot be compared to an MLE-fit Wald by AIC, BIC, or a
likelihood ratio. These are all proper MLEs.

More importantly, shifted LATER nests plain LATER at t0 = 0. That makes
"does this effector's RT distribution demand a non-decision term?" a NESTED
likelihood-ratio test inside the LATER family -- an independent re-test of the
project's central question, in a model family that has no Wald assumptions in it.

The MLE is closed form given t0 (rate mean and SD are just the sample mean and SD of
1/(RT - t0)), so the fit is a 1-D profile search, exactly like wald_fast.py.

ON THE TRUNCATION CONCERN
-------------------------
A Gaussian rate puts some mass below zero, which maps to negative RTs. The relevant
quantity is mu/sigma: at mu/sigma > 4 the implied mass is < 3e-5 and truncation is
numerically irrelevant. check_truncation() reports it so the decision is empirical
rather than assumed. Note the ordering is often the opposite of what people expect --
hand data can have a HIGHER mu/sigma than saccade data, making truncation less of an
issue for the arm, not more.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize


# --------------------------------------------------------------------------- #
# Plain LATER  (2 parameters)
# --------------------------------------------------------------------------- #
def later_logpdf(rt, mu, sigma, t0=0.0):
    """log density of RT = t0 + 1/r with r ~ N(mu, sigma). Jacobian |dr/dt| = 1/tau^2."""
    rt = np.asarray(rt, float)
    tau = rt - t0
    out = np.full(tau.shape, -np.inf)
    ok = tau > 1e-12
    if ok.any():
        r = 1.0 / tau[ok]
        out[ok] = norm.logpdf(r, mu, sigma) - 2.0 * np.log(tau[ok])
    return out


def later_cdf(rt, mu, sigma, t0=0.0):
    """P(RT <= t) = P(r >= 1/(t-t0)) = 1 - Phi((1/(t-t0) - mu)/sigma)."""
    rt = np.asarray(rt, float)
    tau = np.maximum(rt - t0, 1e-12)
    return 1.0 - norm.cdf((1.0 / tau - mu) / sigma)


def later_mle(rt):
    """Closed form: mu and sigma are the sample mean and SD of the rate 1/RT."""
    rt = np.asarray(rt, float)
    r = 1.0 / rt
    mu, sigma = float(r.mean()), float(r.std(ddof=0))
    ll = float(np.sum(later_logpdf(rt, mu, sigma)))
    return {"mu": mu, "sigma": sigma, "t0": 0.0, "ll": ll, "k": 2,
            "model": "LATER"}


# --------------------------------------------------------------------------- #
# Shifted LATER  (3 parameters) -- the like-for-like comparison to the Wald
# --------------------------------------------------------------------------- #
def _shift_nll(t0, rt):
    tau = rt - t0
    if np.any(tau <= 1e-9):
        return np.inf, np.nan, np.nan
    r = 1.0 / tau
    mu, sigma = r.mean(), r.std(ddof=0)
    if sigma <= 0 or not np.isfinite(sigma):
        return np.inf, np.nan, np.nan
    ll = np.sum(norm.logpdf(r, mu, sigma) - 2.0 * np.log(tau))
    if not np.isfinite(ll):
        return np.inf, mu, sigma
    return float(-ll), float(mu), float(sigma)


def shifted_later_mle(rt, floor=0.0, n_grid=60):
    """Profile over t0; (mu, sigma) closed form given t0. Same structure as wald_fast."""
    rt = np.asarray(rt, float)
    hi = float(rt.min()) - 1e-5
    if hi <= floor:
        base = later_mle(rt)
        base.update(model="shifted LATER", k=3, t0=0.0, t0_at_floor=True)
        return base
    grid = np.linspace(floor, hi, n_grid)
    vals = np.array([_shift_nll(t, rt)[0] for t in grid])
    k = int(np.argmin(vals))
    lo_b, hi_b = grid[max(k - 1, 0)], grid[min(k + 1, n_grid - 1)]
    if hi_b - lo_b > 1e-12:
        r = minimize_scalar(lambda t: _shift_nll(t, rt)[0], bounds=(lo_b, hi_b),
                            method="bounded", options={"xatol": 1e-10})
        t0 = float(r.x)
    else:
        t0 = float(grid[k])
    nll, mu, sigma = _shift_nll(t0, rt)
    return {"mu": mu, "sigma": sigma, "t0": t0, "ll": float(-nll), "k": 3,
            "model": "shifted LATER",
            "t0_at_floor": bool(abs(t0 - floor) < 1e-7)}


# --------------------------------------------------------------------------- #
# Truncated LATER (rate constrained positive)
# --------------------------------------------------------------------------- #
def truncated_later_mle(rt):
    """Rate ~ N(mu, sigma) restricted to r > 0. Only matters when mu/sigma is small."""
    rt = np.asarray(rt, float)
    r = 1.0 / rt

    def nll(p):
        mu, ls = p[0], p[1]
        sigma = np.exp(ls)
        Z = norm.sf(-mu / sigma)          # P(r > 0)
        if Z <= 1e-12:
            return 1e10
        v = np.sum(norm.logpdf(r, mu, sigma) - np.log(Z) - 2.0 * np.log(rt))
        return -v if np.isfinite(v) else 1e10

    x0 = np.array([r.mean(), np.log(max(r.std(ddof=0), 1e-6))])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 4000})
    mu, sigma = float(res.x[0]), float(np.exp(res.x[1]))
    return {"mu": mu, "sigma": sigma, "t0": 0.0, "ll": float(-res.fun), "k": 2,
            "model": "truncated LATER"}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def check_truncation(fit):
    """
    Implied probability of a negative rate (hence a negative RT). The user-facing
    question is whether a truncated fit is needed at all.
    """
    z = fit["mu"] / fit["sigma"]
    p_neg = float(norm.cdf(-z))
    return {"mu_over_sigma": float(z), "p_rate_below_zero": p_neg,
            "truncation_needed": bool(z < 4.0)}


def reciprobit_coords(rt):
    """
    Reciprobit plotting coordinates: x = -1/RT (so faster is rightward), y = probit
    of the cumulative proportion. A straight line means LATER's Gaussian rate holds.
    """
    rt = np.sort(np.asarray(rt, float))
    n = rt.size
    p = (np.arange(1, n + 1) - 0.5) / n
    return -1.0 / rt, norm.ppf(p)


def reciprobit_r2(rt):
    """The statistic your pipeline currently reports. Kept for continuity ONLY."""
    x, y = reciprobit_coords(rt)
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def later_ks(rt_sorted, fit):
    """KS distance to the fitted LATER CDF."""
    rt_sorted = np.asarray(rt_sorted, float)
    n = rt_sorted.size
    F = later_cdf(rt_sorted, fit["mu"], fit["sigma"], fit.get("t0", 0.0))
    return float(max(np.max(np.arange(1, n + 1) / n - F),
                     np.max(F - np.arange(0, n) / n)))


def later_rvs(fit, n, rng=None):
    """Simulate from a fitted LATER model. Rates <= 0 are resampled."""
    rng = rng or np.random.default_rng()
    out = np.empty(n)
    filled = 0
    while filled < n:
        r = rng.normal(fit["mu"], fit["sigma"], n - filled)
        r = r[r > 1e-9]
        take = min(r.size, n - filled)
        out[filled:filled + take] = fit.get("t0", 0.0) + 1.0 / r[:take]
        filled += take
    return out


# --------------------------------------------------------------------------- #
# The nested test: does this effector's RT distribution demand a shift?
# --------------------------------------------------------------------------- #
def shift_lrt(rt, floor=0.0):
    """
    Likelihood-ratio test of t0 = 0 inside the LATER family.

    t0 = 0 sits on the boundary of the parameter space, so the null distribution is
    a 50:50 mixture of a point mass at zero and chi2_1 (Self & Liang 1987), NOT a
    plain chi2_1. Using the naive chi2_1 would roughly double the p-value.

    Large D on hand cells + small D on saccade cells re-derives the project's core
    dissociation without any Wald assumption.
    """
    rt = np.asarray(rt, float)
    plain = later_mle(rt)
    shifted = shifted_later_mle(rt, floor=floor)
    D = 2.0 * (shifted["ll"] - plain["ll"])
    p = 0.5 * stats.chi2.sf(D, 1) if D > 0 else 1.0
    return {"D": float(max(D, 0.0)), "p": float(p),
            "t0_ms": 1000.0 * shifted["t0"],
            "ll_plain": plain["ll"], "ll_shifted": shifted["ll"],
            "demands_shift": bool(p < 0.05)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def sim_wald(v, a, t0, n):
        mu, lam = a / v, a ** 2
        return t0 + stats.invgauss.rvs(mu / lam, scale=lam, size=n, random_state=rng)

    print("nested shift test inside the LATER family\n")
    print(f"  {'cell':28s} {'t0 (ms)':>9s} {'D':>8s} {'p':>9s} {'demands shift':>15s}")
    for nm, v, a, t0, n in [("HAND  true t0=165", 9.0, 0.78, 0.165, 160),
                            ("HAND  true t0=162", 12.6, 0.82, 0.162, 160),
                            ("SACC  true t0=30", 11.4, 1.45, 0.030, 110),
                            ("SACC  true t0=35", 13.3, 1.49, 0.035, 110)]:
        x = sim_wald(v, a, t0, n)
        r = shift_lrt(x)
        print(f"  {nm:28s} {r['t0_ms']:9.1f} {r['D']:8.2f} {r['p']:9.4f} "
              f"{str(r['demands_shift']):>15s}")

    print("\ntruncation check")
    for nm, v, a, t0, n in [("HAND", 9.0, 0.78, 0.165, 160),
                            ("SACC", 11.4, 1.45, 0.030, 110)]:
        x = sim_wald(v, a, t0, n)
        f = later_mle(x)
        c = check_truncation(f)
        print(f"  {nm}: mu/sigma = {c['mu_over_sigma']:.2f}  "
              f"P(rate<0) = {c['p_rate_below_zero']:.2e}  "
              f"needed = {c['truncation_needed']}")
