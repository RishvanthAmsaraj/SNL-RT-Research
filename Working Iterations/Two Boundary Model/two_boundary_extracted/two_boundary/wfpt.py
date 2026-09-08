"""
wfpt.py  --  Wiener first-passage-time density (two-boundary DDM)

Navarro & Fuss (2009), "Fast and accurate calculations for first-passage times
in Wiener diffusion models", J. Math. Psych. 53:222-230.  doi:10.1016/j.jmp.2009.02.003

Parameterisation (unit diffusion noise, s = 1, matching CODE_REFERENCE.md):

    a  boundary SEPARATION      (absorbing barriers at 0 and a)
    v  drift rate               (sign free: positive drifts toward the upper barrier)
    w  relative start point     z / a, in (0, 1); w = 0.5 is unbiased
    t0 non-decision time        RT = t0 + decision time

The density of hitting the LOWER barrier (0) at decision time tau is

    f_lower(tau | v, a, w) = (1/a^2) * exp(-v*a*w - v^2*tau/2) * g(tau/a^2 | w)

where g is the normalised (a=1, v=0) density, computed by whichever of the
small-time or large-time series needs fewer terms:

    small time: g_s(u|w) = (2*pi*u^3)^(-1/2) * SUM_k (w+2k) * exp(-(w+2k)^2/(2u))
    large time: g_l(u|w) = pi * SUM_{k>=1} k * exp(-k^2*pi^2*u/2) * sin(k*pi*w)

The UPPER barrier density is obtained by the standard reflection
    f_upper(tau | v, a, w) = f_lower(tau | -v, a, 1-w).

This module gives two implementations:
  * numpy_*  -- adaptive term counts, exact NF algorithm. Reference / simulation /
                goodness-of-fit. No PyMC needed.
  * pt_*     -- PyTensor, fixed term counts, fully differentiable. Used inside the
                PyMC model so NUTS gets analytic gradients.

The PyTensor version is validated against the NumPy reference in test_wfpt.py.

Note on scale: `a` here is the separation between two barriers, whereas in the
single-boundary shifted-Wald pipeline `a` is the height of one barrier. Two-boundary
`a` will land at roughly twice the single-boundary value for comparable RTs. The two
are NOT interchangeable and should never be plotted on the same axis without saying so.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "numpy_wfpt_logpdf", "numpy_wfpt_pdf", "numpy_p_upper",
    "simulate_ddm", "pt_wfpt_logpdf",
]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
DEFAULT_ERR = 1e-10


# --------------------------------------------------------------------------- #
# NumPy reference implementation (adaptive number of terms, exact NF algorithm)
# --------------------------------------------------------------------------- #
def _k_small(u: float, err: float) -> float:
    """Number of terms needed by the small-time series (NF eq. 13)."""
    arg = 2.0 * np.sqrt(2.0 * np.pi * u) * err
    if arg < 1.0:
        ks = 2.0 + np.sqrt(-2.0 * u * np.log(arg))
        return max(ks, np.sqrt(u) + 1.0)
    return 2.0


def _k_large(u: float, err: float) -> float:
    """Number of terms needed by the large-time series (NF eq. 14)."""
    arg = np.pi * u * err
    if arg < 1.0:
        kl = np.sqrt(-2.0 * np.log(arg) / (np.pi ** 2 * u))
        return max(kl, 1.0 / (np.pi * np.sqrt(u)))
    return 1.0 / (np.pi * np.sqrt(u))


def _g_norm(u: float, w: float, err: float) -> float:
    """Normalised first-passage density at the lower barrier, a=1, v=0."""
    ks, kl = _k_small(u, err), _k_large(u, err)
    if ks < kl:
        K = int(np.ceil(ks))
        lo = -int(np.floor((K - 1) / 2.0))
        hi = int(np.ceil((K - 1) / 2.0))
        k = np.arange(lo, hi + 1, dtype=float)
        s = np.sum((w + 2.0 * k) * np.exp(-((w + 2.0 * k) ** 2) / (2.0 * u)))
        return s / np.sqrt(2.0 * np.pi * u ** 3)
    K = int(np.ceil(kl))
    k = np.arange(1, K + 1, dtype=float)
    s = np.sum(k * np.exp(-(k ** 2) * (np.pi ** 2) * u / 2.0) * np.sin(k * np.pi * w))
    return np.pi * s


def numpy_wfpt_pdf(tau, v, a, w, upper=False, err: float = DEFAULT_ERR):
    """
    Defective first-passage density at one barrier.

    tau     decision time (RT - t0), > 0. Scalar or array.
    upper   False -> density at the lower barrier (0)
            True  -> density at the upper barrier (a)

    Integrating over tau gives P(that barrier), not 1 -- these are defective
    densities, which is exactly what makes choice proportions informative.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if upper:
        v, w = -v, 1.0 - w
    out = np.zeros_like(tau)
    ok = tau > 0
    for i in np.flatnonzero(ok):
        u = tau[i] / a ** 2
        g = _g_norm(u, w, err)
        out[i] = max(g, 0.0) / a ** 2 * np.exp(-v * a * w - v ** 2 * tau[i] / 2.0)
    return out if out.size > 1 else float(out[0])


def numpy_wfpt_logpdf(tau, v, a, w, upper=False, err: float = DEFAULT_ERR):
    p = np.atleast_1d(numpy_wfpt_pdf(tau, v, a, w, upper=upper, err=err))
    return np.log(np.maximum(p, 1e-300))


def numpy_p_upper(v, a, w):
    """
    Analytic probability of terminating at the upper barrier.
        P(upper) = (1 - exp(-2*v*a*w)) / (1 - exp(-2*v*a))
    with the v -> 0 limit equal to w.
    """
    if abs(v) < 1e-9:
        return float(w)
    x = -2.0 * v * a
    # exp-safe form for large |x|
    return float((1.0 - np.exp(x * w)) / (1.0 - np.exp(x)))


# --------------------------------------------------------------------------- #
# Simulator (posterior predictive checks, parameter recovery, power analysis)
# --------------------------------------------------------------------------- #
def simulate_ddm(n, v, a, w, t0, dt=1e-4, max_t=5.0, rng=None):
    """
    Euler-Maruyama simulation of the two-boundary DDM.
    Returns (rt_seconds, choice) with choice 1 = upper barrier, 0 = lower.
    Non-terminating paths are returned as NaN.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(n)
    x = np.full(n, w * a, dtype=float)
    t = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    choice = np.zeros(n, dtype=int)
    rt = np.full(n, np.nan)
    sd = np.sqrt(dt)
    n_steps = int(max_t / dt)
    for _ in range(n_steps):
        live = ~done
        if not live.any():
            break
        x[live] += v * dt + sd * rng.standard_normal(live.sum())
        t[live] += dt
        hi = live & (x >= a)
        lo = live & (x <= 0.0)
        rt[hi] = t[hi] + t0
        choice[hi] = 1
        rt[lo] = t[lo] + t0
        choice[lo] = 0
        done |= hi | lo
    return rt, choice


# --------------------------------------------------------------------------- #
# PyTensor implementation (fixed term counts, differentiable -> NUTS)
# --------------------------------------------------------------------------- #
def pt_wfpt_logpdf(tau, v, a, w, choice, err: float = DEFAULT_ERR,
                   k_small: int = 7, k_large: int = 10):
    """
    Elementwise log density of (decision time, barrier) under the two-boundary DDM.

    tau     tensor, decision time RT - t0 (must be > 0; caller guarantees this)
    v, a, w tensors broadcastable to tau
    choice  tensor of 0/1 DATA -- 1 = terminated at the upper barrier

    Barrier is handled by the reflection v -> -v, w -> 1-w, written without a
    switch so the gradient stays clean:
        sgn = 1 - 2*choice          (+1 lower, -1 upper)
        v_e = sgn * v
        w_e = choice + sgn * w      (w if lower, 1-w if upper)

    Both series are evaluated with fixed term counts and the one Navarro & Fuss
    would have chosen is selected elementwise. k_small=7 (15 terms) and
    k_large=10 exceed the adaptive counts across the whole plausible parameter
    range for human RT; test_wfpt.py checks this against the adaptive reference.
    """
    import pytensor.tensor as pt

    sgn = 1.0 - 2.0 * choice
    v_e = sgn * v
    w_e = choice + sgn * w

    u = tau / a ** 2                                    # rescaled time

    # --- how many terms would the adaptive algorithm need? ------------------ #
    arg_s = pt.clip(2.0 * pt.sqrt(2.0 * np.pi * u) * err, 1e-300, 1.0 - 1e-12)
    ks = pt.maximum(2.0 + pt.sqrt(-2.0 * u * pt.log(arg_s)), pt.sqrt(u) + 1.0)
    ks = pt.switch(2.0 * pt.sqrt(2.0 * np.pi * u) * err < 1.0, ks, 2.0)

    arg_l = pt.clip(np.pi * u * err, 1e-300, 1.0 - 1e-12)
    kl = pt.maximum(pt.sqrt(-2.0 * pt.log(arg_l) / (np.pi ** 2 * u)),
                    1.0 / (np.pi * pt.sqrt(u)))
    kl = pt.switch(np.pi * u * err < 1.0, kl, 1.0 / (np.pi * pt.sqrt(u)))

    use_small = ks <= kl

    # --- small-time series -------------------------------------------------- #
    ks_idx = pt.as_tensor_variable(
        np.arange(-k_small, k_small + 1, dtype="float64"))          # (2K+1,)
    wk = w_e[:, None] + 2.0 * ks_idx[None, :]
    g_small = pt.sum(wk * pt.exp(-(wk ** 2) / (2.0 * u[:, None])), axis=1)
    g_small = g_small / pt.sqrt(2.0 * np.pi * u ** 3)

    # --- large-time series -------------------------------------------------- #
    kl_idx = pt.as_tensor_variable(
        np.arange(1, k_large + 1, dtype="float64"))                 # (K,)
    g_large = pt.sum(
        kl_idx[None, :]
        * pt.exp(-(kl_idx[None, :] ** 2) * (np.pi ** 2) * u[:, None] / 2.0)
        * pt.sin(kl_idx[None, :] * np.pi * w_e[:, None]),
        axis=1) * np.pi

    g = pt.switch(use_small, g_small, g_large)
    g = pt.maximum(g, 1e-300)                            # tail underflow guard

    return pt.log(g) - 2.0 * pt.log(a) - v_e * a * w_e - (v_e ** 2) * tau / 2.0


# --------------------------------------------------------------------------- #
# Vectorised NumPy density (fixed terms, same series as the PyTensor version).
# Used by the MLE fitter and the goodness-of-fit code, where the per-trial Python
# loop in numpy_wfpt_pdf would be far too slow.
# --------------------------------------------------------------------------- #
def numpy_wfpt_logpdf_vec(tau, v, a, w, choice, err: float = DEFAULT_ERR,
                          k_small: int = 7, k_large: int = 10):
    """Elementwise log density of (decision time, barrier). choice: 1 = upper."""
    tau = np.asarray(tau, dtype=float)
    choice = np.asarray(choice, dtype=float)
    sgn = 1.0 - 2.0 * choice
    v_e = sgn * np.asarray(v, dtype=float)
    w_e = choice + sgn * np.asarray(w, dtype=float)
    a = np.asarray(a, dtype=float)

    bad = ~(tau > 0)
    u = np.where(bad, 1.0, tau) / a ** 2

    arg_s = np.clip(2.0 * np.sqrt(2.0 * np.pi * u) * err, 1e-300, 1.0 - 1e-12)
    ks = np.maximum(2.0 + np.sqrt(-2.0 * u * np.log(arg_s)), np.sqrt(u) + 1.0)
    ks = np.where(2.0 * np.sqrt(2.0 * np.pi * u) * err < 1.0, ks, 2.0)

    arg_l = np.clip(np.pi * u * err, 1e-300, 1.0 - 1e-12)
    kl = np.maximum(np.sqrt(-2.0 * np.log(arg_l) / (np.pi ** 2 * u)),
                    1.0 / (np.pi * np.sqrt(u)))
    kl = np.where(np.pi * u * err < 1.0, kl, 1.0 / (np.pi * np.sqrt(u)))

    ki = np.arange(-k_small, k_small + 1, dtype=float)[None, :]
    wk = np.atleast_1d(w_e)[:, None] + 2.0 * ki
    g_s = np.sum(wk * np.exp(-(wk ** 2) / (2.0 * u[:, None])), axis=1) \
        / np.sqrt(2.0 * np.pi * u ** 3)

    kj = np.arange(1, k_large + 1, dtype=float)[None, :]
    g_l = np.pi * np.sum(kj * np.exp(-(kj ** 2) * (np.pi ** 2) * u[:, None] / 2.0)
                         * np.sin(kj * np.pi * np.atleast_1d(w_e)[:, None]), axis=1)

    g = np.maximum(np.where(ks <= kl, g_s, g_l), 1e-300)
    lp = np.log(g) - 2.0 * np.log(a) - v_e * a * w_e - (v_e ** 2) * np.where(bad, 0.0, tau) / 2.0
    return np.where(bad, -np.inf, lp)


def sample_ddm(n, v, a, w, t0, rng=None, n_grid=3000, t_max=None):
    """
    Fast, unbiased sampler: exact barrier probability + inverse-CDF on the analytic
    defective densities. Preferred over simulate_ddm (Euler-Maruyama) everywhere --
    Euler overshoots the barrier by O(sqrt(dt)) and biases RTs fast, which shows up
    in posterior predictive KS values at realistic n.

    Returns (rt_seconds, choice) with choice 1 = upper barrier.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(n)
    p_up = numpy_p_upper(v, a, w)
    choice = (rng.random(n) < p_up).astype(int)

    if t_max is None:
        # mean decision time is a*(w - P_up) / v for v != 0, a^2*w*(1-w) for v -> 0;
        # 25x that is comfortably into the tail for either.
        m = (a * (w - p_up) / v) if abs(v) > 1e-6 else (a ** 2 * w * (1.0 - w))
        t_max = max(6.0 * abs(m), 2.0)
    grid = np.linspace(1e-5, t_max, n_grid)

    rt = np.empty(n)
    for up in (0, 1):
        idx = np.flatnonzero(choice == up)
        if idx.size == 0:
            continue
        dens = np.exp(numpy_wfpt_logpdf_vec(grid, v, a, w,
                                            np.full(grid.shape, float(up))))
        cdf = np.cumsum(dens) * (grid[1] - grid[0])
        if cdf[-1] <= 0:
            rt[idx] = np.nan
            continue
        cdf = cdf / cdf[-1]
        rt[idx] = np.interp(rng.random(idx.size), cdf, grid)
    return rt + t0, choice
