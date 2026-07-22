"""
Parity with the standalone pipeline.

The app exists so the analysis can be run without an IDE -- not so it can run a
different analysis. These tests embed the reference implementations copied out of
the standalone scripts and assert the app reproduces them exactly, so any future
edit that changes a fitted number fails here instead of silently disagreeing with
the published CSVs.

Reference sources:
    DDM_fit.py         -- fit_single, _mix_nll, _moments_to_wald, fit_mixture, and
                          the single-vs-mixture selection rule in main()
    Bayesian_HRT_fit.py / Bayesian_SRT_fit.py -- priors, t0 link, log-likelihood
    LATER_analysis.py  -- recipro, fit_line
    CODE_REFERENCE.md  -- bounds, floors, filters
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import differential_evolution, minimize

from kinarm_rt import data, filters
from kinarm_rt._speeds import (SPEEDS, PHYSIO_FLOOR, FILTER_WINDOWS, V_MAX, A_MAX,
                               P_CONTAM, MIX_SHIFT_FLOOR)
from kinarm_rt.models import wald as W
from kinarm_rt.models import later as L

# --------------------------------------------------------------------------- #
# Reference implementation -- copied from DDM_fit.py
# --------------------------------------------------------------------------- #
REF_A_MAX, REF_V_MAX = 2.5, 20.0
REF_HRT_FLOOR, REF_SRT_FLOOR = 0.130, 0.070
REF_P_CONTAM = 0.05


def ref_wald_pdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return (a / np.sqrt(2 * np.pi * t ** 3)) * np.exp(-(a - v * t) ** 2 / (2 * t))


def ref_wald_cdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return stats.invgauss.cdf(t, mu=1 / (v * a), scale=a ** 2)


def ref_fit_single(rts, floor, contam):
    Tr = rts.max() - rts.min()

    def nll(p):
        v, a, t0 = p
        adj = rts - t0
        if np.any(adj <= 0):
            return 1e10
        w = ref_wald_pdf(adj, v, a)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            return 1e10
        d = (1 - contam) * w + (contam / Tr if contam > 0 else 0.0)
        if np.any(d <= 0):
            return 1e10
        return -np.sum(np.log(d))

    b = [(0.1, REF_V_MAX), (0.05, REF_A_MAX),
         (floor, max(np.percentile(rts, 3) - 0.002, floor + 1e-3))]
    best = None
    for s in [42, 7]:
        r = differential_evolution(nll, b, seed=s, maxiter=400, tol=1e-9,
                                   popsize=12, polish=True)
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    adj = rts - x[2]
    return x, best.fun, stats.kstest(adj, lambda z: ref_wald_cdf(z, x[0], x[1])).statistic


def ref_mix_nll(p, rts, Tr, contam):
    pi, ve, ae, t0e, vr, ar, t0r = p
    ee = rts - t0e
    rr = rts - t0r
    if np.any(ee <= 0) or np.any(rr <= 0):
        return 1e10
    if (t0e + ae / ve) >= (t0r + ar / vr):
        return 1e10
    core = pi * ref_wald_pdf(ee, ve, ae) + (1 - pi) * ref_wald_pdf(rr, vr, ar)
    d = (1 - contam) * core + (contam / Tr if contam > 0 else 0.0)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return 1e10
    return -np.sum(np.log(d))


def ref_moments_to_wald(mean, sd, floor):
    t0 = min(max(floor, mean - 2.5 * sd), mean - 1e-3)
    mu = mean - t0
    var = max(sd ** 2, 1e-6)
    lam = mu ** 3 / var
    a = np.sqrt(lam)
    v = a / mu
    return [np.clip(v, 0.1, REF_V_MAX), np.clip(a, 0.05, REF_A_MAX),
            np.clip(t0, floor, mean - 1e-3)]


def ref_fit_mixture(rts, floor, contam):
    from sklearn.mixture import GaussianMixture
    Tr = rts.max() - rts.min()
    mn = rts.min()
    cands = []
    try:
        gm = GaussianMixture(2, n_init=8, random_state=0).fit(rts.reshape(-1, 1))
        o = np.argsort(gm.means_.ravel())
        m1, m2 = gm.means_.ravel()[o]
        s1, s2 = np.sqrt(gm.covariances_.ravel()[o])
        w1 = gm.weights_[o][0]
        cands.append([np.clip(w1, 0.05, 0.95)]
                     + ref_moments_to_wald(m1, s1, floor)
                     + ref_moments_to_wald(m2, s2, floor))
    except Exception:
        pass
    for pe, pr, piw in [(15, 65, 0.5), (25, 70, 0.4), (10, 55, 0.6), (30, 75, 0.5)]:
        me, mr = np.percentile(rts, pe), np.percentile(rts, pr)
        if mr - me < 0.005:
            continue
        cands.append([piw] + ref_moments_to_wald(me, (mr - me) / 3, floor)
                     + ref_moments_to_wald(mr, (rts.max() - mr) / 3, floor))
    b = [(0.02, 0.98), (0.1, REF_V_MAX), (0.05, REF_A_MAX), (floor, mn - 1e-3),
         (0.1, REF_V_MAX), (0.05, REF_A_MAX), (floor, mn - 1e-3)]
    best = None
    for c in cands:
        try:
            r = minimize(ref_mix_nll, c, args=(rts, Tr, contam), method="L-BFGS-B", bounds=b)
            if r.fun < 1e9 and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            pass
    rde = differential_evolution(ref_mix_nll, b, args=(rts, Tr, contam), seed=42,
                                 maxiter=400, tol=1e-9, popsize=14, polish=True)
    if best is None or rde.fun < best.fun:
        best = rde
    x = best.x
    pi = x[0]

    def mc(z):
        return pi * ref_wald_cdf(z - x[3], x[1], x[2]) + (1 - pi) * ref_wald_cdf(z - x[6], x[4], x[5])

    return x, best.fun, stats.kstest(rts, mc).statistic


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def kept():
    raw = data.simulate_dataset(n_participants=6, trials_per_cell=90, seed=11)
    tidy = data.load_trials(raw, participant_col="Participant",
                            hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                            speed_col="Speed_deg_per_s",
                            blocktype_col="BlockType", blocktype_keep="I", rt_units="ms")
    out, _ = filters.apply_windows(tidy, FILTER_WINDOWS)
    return out


def cells(kept, effector, limit=6):
    sub = kept[kept["effector"] == effector]
    out = []
    for (_, _), g in list(sub.groupby(["participant", "condition"]))[:limit]:
        rt = g["rt"].values.astype(float)
        if len(rt) >= 15:
            out.append(rt)
    return out


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_constants_match_code_reference():
    """Bounds, floors, filters and contamination share the published values."""
    assert (V_MAX, A_MAX) == (REF_V_MAX, REF_A_MAX)
    assert PHYSIO_FLOOR["hand"] == REF_HRT_FLOOR
    assert PHYSIO_FLOOR["eye"] == REF_SRT_FLOOR
    assert P_CONTAM == REF_P_CONTAM
    assert MIX_SHIFT_FLOOR == 0.040
    assert FILTER_WINDOWS["hand"] == (0.150, 0.800)     # HRT filter, 150-800 ms
    assert FILTER_WINDOWS["eye"] == (0.080, 0.600)      # SRT filter, 80-600 ms
    assert tuple(SPEEDS) == (0.0, 75.0, 150.0)
    assert W.MIN_TRIALS == 15
    assert W.KS_ACCEPT == 0.10
    assert (W.MIX_MIN_PI, W.MIX_MAX_PI) == (0.10, 0.90)
    assert W.MIX_MIN_MODE_GAP_MS == 30.0


def test_optimiser_settings_match():
    """The differential-evolution settings drive the fitted values, so pin them."""
    assert W.DE_SEEDS == (42, 7)
    assert W.DE_MAXITER == 400
    assert W.DE_TOL == 1e-9
    assert W.DE_POPSIZE == 12
    assert W.MIX_DE_POPSIZE == 14


# --------------------------------------------------------------------------- #
# Densities
# --------------------------------------------------------------------------- #
def test_wald_density_matches_reference():
    t = np.linspace(0.01, 0.9, 250)
    for v, a in [(5.0, 0.8), (12.0, 1.4), (0.5, 0.2)]:
        assert np.allclose(W.wald_pdf(t, v, a), ref_wald_pdf(t, v, a), rtol=0, atol=0)
        assert np.allclose(W.wald_cdf(t, v, a), ref_wald_cdf(t, v, a), rtol=0, atol=0)


def test_bayesian_loglik_formula_matches_density():
    """
    The Bayesian scripts write the log-likelihood out longhand:
        log a - 0.5*log(2*pi) - 1.5*log(tau) - (a - v*tau)^2 / (2*tau)
    It must equal log of the Wald density the frequentist path uses, or the two
    methods would not be fitting the same model.
    """
    tau = np.linspace(0.02, 0.8, 200)
    for v, a in [(6.0, 0.9), (14.0, 1.3)]:
        longhand = (np.log(a) - 0.5 * np.log(2 * np.pi) - 1.5 * np.log(tau)
                    - (a - v * tau) ** 2 / (2 * tau))
        assert np.allclose(longhand, np.log(W.wald_pdf(tau, v, a)), atol=1e-12)


# --------------------------------------------------------------------------- #
# Frequentist fits
# --------------------------------------------------------------------------- #
def test_single_wald_fit_is_identical(kept):
    """ddm_fit_single must reproduce DDM_fit.py::fit_single exactly."""
    for effector, floor in (("hand", REF_HRT_FLOOR), ("eye", REF_SRT_FLOOR)):
        for rt in cells(kept, effector, limit=4):
            xr, fr, kr = ref_fit_single(rt, floor, REF_P_CONTAM)
            xa, fa, ka = W.ddm_fit_single(rt, floor, REF_P_CONTAM)
            assert np.allclose(xr, xa, rtol=0, atol=0), "parameters differ"
            assert fr == fa, "negative log-likelihood differs"
            assert kr == ka, "KS statistic differs"


def test_mixture_fit_is_identical(kept):
    """ddm_fit_mixture must reproduce DDM_fit.py::fit_mixture exactly."""
    for rt in cells(kept, "eye", limit=3):
        xr, fr, kr = ref_fit_mixture(rt, REF_SRT_FLOOR, REF_P_CONTAM)
        xa, fa, ka = W.ddm_fit_mixture(rt, REF_SRT_FLOOR, REF_P_CONTAM)
        assert np.allclose(xr, xa, rtol=0, atol=0)
        assert fr == fa and kr == ka


def test_moments_to_wald_is_identical():
    for mean, sd in [(0.20, 0.05), (0.35, 0.09), (0.12, 0.02)]:
        assert np.allclose(W._moments_to_wald(mean, sd, REF_SRT_FLOOR),
                           ref_moments_to_wald(mean, sd, REF_SRT_FLOOR), atol=0)


def test_selection_rule_matches_script(kept):
    """
    A saccade cell becomes a mixture only under DDM_fit.py's rule: the single Wald
    must fail (KS > 0.10), and the mixture must fit (KS < 0.10), split the trials
    between 10% and 90%, and separate the modes by at least 30 ms.
    """
    for rt in cells(kept, "eye", limit=4):
        xs, _, ks_s = ref_fit_single(rt, REF_SRT_FLOOR, REF_P_CONTAM)
        expected = "single"
        if ks_s > 0.10:
            xm, _, ks_m = ref_fit_mixture(rt, REF_SRT_FLOOR, REF_P_CONTAM)
            pi = xm[0]
            em = (xm[3] + xm[2] / xm[1]) * 1000
            rm = (xm[6] + xm[5] / xm[4]) * 1000
            if (ks_m < 0.10) and (0.10 <= pi <= 0.90) and ((rm - em) >= 30):
                expected = "mixture"
        got = W.ddm_select_srt(rt, REF_SRT_FLOOR, REF_P_CONTAM)
        assert got["model"] == expected
        assert got["ks_single"] == ks_s


def test_preview_uses_the_same_fit(kept):
    """The in-app preview must be Method A, not a faster approximation of it."""
    prev = W.mle_preview(kept, "hand", contamination=REF_P_CONTAM)
    sub = kept[kept["effector"] == "hand"]
    for _, row in prev["cell"].head(3).iterrows():
        g = sub[(sub.participant == row["participant"]) & (sub.condition == row["condition"])]
        xr, _, _ = ref_fit_single(g["rt"].values.astype(float), REF_HRT_FLOOR, REF_P_CONTAM)
        assert abs(row["v"] - xr[0]) < 1e-9
        assert abs(row["a"] - xr[1]) < 1e-9
        assert abs(row["t0_ms"] - xr[2] * 1000.0) < 1e-6


def test_cells_below_fifteen_trials_are_skipped(kept):
    """The pipeline requires at least 15 trials in a cell; so must the app."""
    small = kept.groupby(["effector", "participant", "condition"], group_keys=False).head(10)
    prev = W.mle_preview(small, "hand")
    assert prev["cell"].empty


# --------------------------------------------------------------------------- #
# Bayesian structure
# --------------------------------------------------------------------------- #
def test_bayesian_t0_link_and_floors():
    """
    Both Bayesian scripts place t0 as  floor + (min_RT - floor) * sigmoid(z), which
    bounds it between the physiological floor and the cell's fastest trial.
    """
    minrt = np.array([0.180, 0.240])
    for floor in (REF_HRT_FLOOR, REF_SRT_FLOOR):
        for z in (-4.0, 0.0, 4.0):
            t0 = floor + (minrt - floor) / (1.0 + np.exp(-z))
            assert np.all(t0 >= floor)
            assert np.all(t0 <= minrt)


def test_saccade_prior_mean_differs_from_hand():
    """
    Bayesian_HRT_fit.py uses mu_z ~ Normal(0.5, 1) and Bayesian_SRT_fit.py uses
    mu_z ~ Normal(0.0, 1). fit_effector must apply the same split.
    """
    import inspect
    src = inspect.getsource(W.fit_effector)
    assert 'mu_z_mean = 0.5 if effector == "hand" else 0.0' in src
    assert inspect.signature(W.fit_pooled_hierarchical).parameters["mu_z_mean"].default == 0.5


def test_hierarchical_defaults_match_scripts():
    """target_accept and the sampler seeds are part of the published settings."""
    import inspect
    p = inspect.signature(W.fit_pooled_hierarchical).parameters
    assert p["target_accept"].default == 0.95
    assert p["seed"].default == 7
    assert inspect.signature(W.fit_mixture_cell).parameters["seed"].default == 11
    assert inspect.signature(W.fit_mixture_cell).parameters["target_accept"].default == 0.95


# --------------------------------------------------------------------------- #
# LATER
# --------------------------------------------------------------------------- #
def test_later_reciprobit_matches_reference(kept):
    """recipro and the central-90% line fit are the LATER_analysis.py versions."""
    from scipy.stats import norm, linregress
    rt = kept[(kept.effector == "eye")]["rt"].values.astype(float)[:400] * 1000.0

    rate = np.sort(1000.0 / rt)
    n = len(rate)
    z = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    a, b = int(0.10 * n), int(0.90 * n)
    sl_r, ic_r, r_r, _, _ = linregress(rate[a:b], z[a:b])

    rate_a, z_a = L._recipro(rt)
    sl_a, ic_a, r2_a = L._fit_line(rate_a, z_a)
    assert np.allclose(rate, rate_a, atol=0) and np.allclose(z, z_a, atol=0)
    assert abs(sl_r - sl_a) < 1e-12 and abs(ic_r - ic_a) < 1e-12
    assert abs(r_r ** 2 - r2_a) < 1e-12


def test_later_thresholds():
    """Express cutoff and the minimum cell size come from LATER_analysis.py."""
    assert L.EXPRESS_CUTOFF_MS == 130.0
    assert L.MIN_TRIALS == 40
