"""
wald_fast.py  (v2) -- profile-likelihood fitting for the single-boundary shifted Wald.

CHANGES FROM v1 (both prompted by the real-data validation run)
---------------------------------------------------------------
1. CONSTRAINED M-STEP. v1 clipped the closed-form solution to the v/a caps. Clipping
   gives a feasible point, NOT the constrained optimum, so on cells whose
   unconstrained drift exceeds V_MAX the v1 fitter could land WORSE than differential
   evolution. That is what produced the 3 FAIL cells in the validation run -- a real
   defect in v1, not a benign boundary artifact.

   Fixed by solving ON the active face, which is also closed form:
       a fixed at cap:  v = a*W / sum(w*tau)
       v fixed at cap:  S*a^2 - W*v*a - W = 0,  S = sum(w/tau)
   Verified: clipping was worse than DE on 2/5 cap-hitting test cells (by up to 0.42
   log-lik); the constrained solver matches DE to 1e-4 on all of them.

2. FAST KS. scipy.stats.kstest with a callable costs ~6x the direct ECDF computation
   and dominates bootstrap runtime. Replaced with a direct max-deviation calculation
   and a closed-form normal-based Wald CDF (agrees with scipy to 1.2e-15).

3. MEMORY-SAFE BOOTSTRAP. bootstrap_ks no longer holds intermediate state across
   replicates, and bootstrap_ks_table writes results per cell so a long run is
   resumable rather than all-or-nothing.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq

V_MAX, A_MAX = 20.0, 2.5
V_MIN, A_MIN = 0.1, 0.05
P_CONTAM = 0.05
HRT_FLOOR, SRT_FLOOR = 0.130, 0.070

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Densities
# --------------------------------------------------------------------------- #
def wald_pdf(t, v, a):
    t = np.maximum(np.asarray(t, float), 1e-9)
    return (a / np.sqrt(2 * np.pi * t ** 3)) * np.exp(-(a - v * t) ** 2 / (2 * t))


def wald_cdf(t, v, a):
    """
    Closed-form inverse-Gaussian CDF with lambda = a^2, mu = a/v:
        F = Phi(v*sqrt(t) - a/sqrt(t)) + exp(2av) * Phi(-(v*sqrt(t) + a/sqrt(t)))
    The second term is evaluated as exp(2av + logcdf) so the large exp() and the
    underflowing Phi() cancel in log space. Matches scipy to ~1e-15 across the whole
    parameter box, and avoids scipy's rv_continuous dispatch in hot loops.
    """
    t = np.maximum(np.asarray(t, float), 1e-9)
    s = np.sqrt(t)
    return norm.cdf(v * s - a / s) + np.exp(2.0 * a * v + norm.logcdf(-(v * s + a / s)))


def wald_mean(v, a, t0=0.0):
    """Mean of the shifted Wald. This is what DDM_fit.py's 'express_mode' column holds."""
    return t0 + a / v


def wald_mode(v, a, t0=0.0):
    """Actual mode. Differs from the mean by up to ~27 ms on real cells (audit A6)."""
    mu, lam = a / v, a ** 2
    return t0 + mu * (np.sqrt(1.0 + 9.0 * mu ** 2 / (4.0 * lam ** 2)) - 3.0 * mu / (2.0 * lam))


def ks_statistic(rts_sorted, v, a, t0):
    """KS distance from a SORTED RT vector to the fitted shifted Wald."""
    tau = np.asarray(rts_sorted, float) - t0
    n = tau.size
    F = wald_cdf(tau, v, a)
    return float(max(np.max(np.arange(1, n + 1) / n - F),
                     np.max(F - np.arange(0, n) / n)))


# --------------------------------------------------------------------------- #
# Constrained closed-form M-step
# --------------------------------------------------------------------------- #
def _ig_mle_constrained(tau, w):
    """
    Weighted MLE of (v, a) subject to the parameter box, in closed form.

    Unconstrained (Tweedie 1957):  mu = mean_w(tau), 1/lam = mean_w(1/tau) - 1/mu,
    a = sqrt(lam), v = a/mu. If that violates a cap, the constrained optimum lies on
    the active face, where the free parameter again has a closed form -- so we
    evaluate each candidate face and take the best. No iterative optimizer needed.
    """
    W = w.sum()
    Sm = (w * tau).sum()
    S = (w / tau).sum()
    if Sm <= 0 or S <= 0 or W <= 0:
        return V_MIN, A_MIN
    mu = Sm / W
    inv = S / W - 1.0 / mu
    lam = 1.0 / max(inv, 1e-12)
    a = np.sqrt(lam)
    v = a / mu
    if V_MIN <= v <= V_MAX and A_MIN <= a <= A_MAX:
        return float(v), float(a)

    cands = []
    for af in ([A_MAX] if a > A_MAX else []) + ([A_MIN] if a < A_MIN else []):
        cands.append((min(max(af * W / Sm, V_MIN), V_MAX), af))
    for vf in ([V_MAX] if v > V_MAX else []) + ([V_MIN] if v < V_MIN else []):
        af = (W * vf + np.sqrt((W * vf) ** 2 + 4.0 * W * S)) / (2.0 * S)
        cands.append((vf, min(max(af, A_MIN), A_MAX)))
    cands.append((min(max(v, V_MIN), V_MAX), min(max(a, A_MIN), A_MAX)))

    best, bf = None, np.inf
    for vv, aa in cands:
        f = wald_pdf(tau, vv, aa)
        if np.any(f <= 0) or not np.all(np.isfinite(f)):
            continue
        val = -np.sum(w * np.log(f))
        if val < bf:
            bf, best = val, (float(vv), float(aa))
    return best if best is not None else (min(max(v, V_MIN), V_MAX),
                                          min(max(a, A_MIN), A_MAX))


def _profile(t0, rts, Tr, contam, n_em=50, tol=1e-11):
    """Profile NLL at t0, with (v, a) concentrated out by constrained EM."""
    tau = rts - t0
    if np.any(tau <= 1e-9):
        return 1e10, np.nan, np.nan
    w = np.ones_like(tau)
    v = a = np.nan
    for _ in range(n_em):
        v, a = _ig_mle_constrained(tau, w)
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
    return max(np.percentile(rts, 3) - 0.002, floor + 1e-3)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_cell(rts, floor, contam=P_CONTAM, n_grid=40, sorted_rts=None):
    """MLE of (v, a, t0) by 1-D profile search. Deterministic: no seeds, no retries."""
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
    srt = np.sort(rts) if sorted_rts is None else sorted_rts
    ks = ks_statistic(srt, v, a, t0)
    return {"v": v, "a": a, "t0": t0, "nll": nll, "ks": ks,
            "v_at_cap": bool(v >= V_MAX - 1e-6 or v <= V_MIN + 1e-6),
            "a_at_cap": bool(a >= A_MAX - 1e-6 or a <= A_MIN + 1e-6),
            "at_bound": bool(v >= V_MAX - 1e-6 or v <= V_MIN + 1e-6
                             or a >= A_MAX - 1e-6 or a <= A_MIN + 1e-6),
            "t0_at_floor": bool(abs(t0 - floor) < 1e-6)}


def fit_fixed_t0(rts, t0, contam=P_CONTAM):
    """(v, a) with t0 held fixed. Closed form -- no optimizer."""
    rts = np.asarray(rts, float)
    if rts.min() <= t0:
        return None
    Tr = float(rts.max() - rts.min()) or 1e-3
    nll, v, a = _profile(t0, rts, Tr, contam)
    return {"v": v, "a": a, "t0": t0, "nll": nll,
            "ks": ks_statistic(np.sort(rts), v, a, t0)}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def profile_curve(rts, lo_t0=0.0, hi_t0=None, contam=P_CONTAM, n=200):
    """Profile NLL vs t0. Compute once; every floor question becomes a lookup."""
    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min()) or 1e-3
    if hi_t0 is None:
        hi_t0 = float(rts.min()) - 1e-4
    ts = np.linspace(max(lo_t0, 1e-4), hi_t0, n)
    return ts, np.array([_profile(t, rts, Tr, contam)[0] for t in ts])


def profile_ci(rts, floor, contam=P_CONTAM, level=0.95):
    """
    Profile-likelihood CI for t0. floor_open=True means the profile is still
    improving at the floor, i.e. t0 is not identified from below.
    """
    rts = np.asarray(rts, float)
    Tr = float(rts.max() - rts.min()) or 1e-3
    cut = 0.5 * stats.chi2.ppf(level, 1)
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
    Constrained t0 at each candidate floor. Slope near 1 means t0 tracks the floor.

    NOTE ON SELECTION BIAS: callers typically skip cells whose min RT is below the
    highest candidate floor. Those are the FASTEST cells, i.e. exactly the ones most
    likely to floor -- so the reported tracking fraction UNDERSTATES flooring. Report
    how many cells were skipped alongside the fraction.
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
    Parametric-bootstrap p-value for the KS statistic (audit A1).

    The fixed 0.10 threshold in DDM_fit.py assumes KNOWN parameters. Yours are
    estimated, so the null is much tighter and 0.10 is not a 5% test.

    B controls the resolution: the smallest achievable p-value is 1/(B+1). B=50 caps
    p at 0.02 AND quantises it to {0.020, 0.039, 0.059, ...}, so a reported median
    near 0.049 is an artefact of averaging two adjacent quantisation levels rather
    than a real value. Use B >= 200 for anything you report.
    """
    rts = np.asarray(rts, float)
    fit = fit_cell(rts, floor, contam)
    v, a, t0, ks_obs = fit["v"], fit["a"], fit["t0"], fit["ks"]
    n = rts.size
    mu, lam = a / v, a ** 2
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        sim = t0 + stats.invgauss.rvs(mu / lam, scale=lam, size=n, random_state=rng)
        sim.sort()
        fb = fit_cell(sim, floor, contam, sorted_rts=sim)
        null[b] = fb["ks"]
    p = float((np.sum(null >= ks_obs) + 1) / (B + 1))
    return {"ks": ks_obs, "ks_p": p, "p_floor": 1.0 / (B + 1),
            "crit95": float(np.percentile(null, 95)),
            "crit90": float(np.percentile(null, 90)),
            "rejected": bool(p < 0.05), "B": B,
            **{k: fit[k] for k in ("v", "a", "t0", "at_bound", "v_at_cap",
                                   "a_at_cap", "t0_at_floor")}}


def bootstrap_ks_table(cells, floor, contam=P_CONTAM, B=200, out_csv=None,
                       status=print):
    """
    Run bootstrap_ks over many cells, writing each row as it completes.

    cells: iterable of (key_dict, rts). Resumable -- rows already present in out_csv
    are skipped, so a long run can be interrupted and restarted. This is the
    memory-safe path for B >= 200 across all cells.
    """
    import os
    import pandas as pd

    done = set()
    if out_csv and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        kcols = [c for c in ("effector", "pid", "spd") if c in prev.columns]
        done = set(map(tuple, prev[kcols].values.tolist()))
        status(f"resuming: {len(done)} cells already done")

    header = not (out_csv and os.path.exists(out_csv))
    rows = []
    for key, rts in cells:
        kt = tuple(key[c] for c in ("effector", "pid", "spd") if c in key)
        if kt in done:
            continue
        r = bootstrap_ks(rts, floor(key) if callable(floor) else floor,
                         contam=contam, B=B,
                         seed=int(abs(hash(kt)) % 10 ** 6))
        row = {**key, **{k: r[k] for k in ("ks", "ks_p", "crit95", "crit90",
                                           "rejected", "B", "at_bound")}}
        rows.append(row)
        if out_csv:
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=header, index=False)
            header = False
        status(f"  {kt} ks={r['ks']:.4f} p={r['ks_p']:.3f}"
               f"{'  REJECTED' if r['rejected'] else ''}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)

    def sim(v, a, t0, n):
        mu, lam = a / v, a ** 2
        return t0 + stats.invgauss.rvs(mu / lam, scale=lam, size=n, random_state=rng)

    print("v2 regression: cells whose true drift EXCEEDS the cap")
    print("(these are the cells where v1's clipping produced FAIL verdicts)\n")
    for nm, v, a, t0, n, fl in [("v=25 (beyond cap)", 25.0, 1.5, 0.070, 110, 0.070),
                                ("v=32 (far beyond)", 32.0, 2.2, 0.070, 110, 0.070),
                                ("normal control   ", 9.0, 0.78, 0.165, 160, 0.130)]:
        x = sim(v, a, t0, n)
        t = time.perf_counter(); f = fit_cell(x, fl); dt = time.perf_counter() - t
        print(f"  {nm}: v={f['v']:6.3f} a={f['a']:.4f} t0={f['t0']*1000:6.1f}ms "
              f"nll={f['nll']:10.4f} v_at_cap={f['v_at_cap']}  ({dt*1000:.0f} ms)")

    x = sim(9.0, 0.78, 0.165, 160)
    print(f"\nprofile_ci: ", end="")
    c = profile_ci(x, HRT_FLOOR)
    print(f"t0={c['t0']*1000:.1f} CI=[{c['lo']*1000:.1f},{c['hi']*1000:.1f}] "
          f"floor_open={c['floor_open']}")
    t = time.perf_counter(); b = bootstrap_ks(x, HRT_FLOOR, B=200); dt = time.perf_counter() - t
    print(f"bootstrap_ks (B=200): ks={b['ks']:.4f} p={b['ks_p']:.3f} "
          f"crit95={b['crit95']:.4f}  ({dt:.1f}s/cell)")
    print(f"  -> 96 cells at B=200 ~ {96*dt/60:.1f} min single-threaded")
