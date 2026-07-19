"""
Analysis suite: the model-free and sensitivity analyses from the repository,
plus a parameter-recovery study. All fast (NumPy/SciPy + the MLE preview), so
they run without waiting on NUTS.

  fixed_t0_sensitivity   -- refit with t0 fixed at several values (SRT_fixed_t0_analysis)
  identifiability_sweep  -- how many cells pin to the floor (SRT_identifiability_check)
  mixture_threshold_sensitivity -- stability of the bimodal-cell count vs the dip cutoff
  vincentiles            -- model-free quantile-averaged RTs by speed (vincentile_figures)
  parameter_recovery     -- simulate from known parameters, refit, check recovery
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import kstest

from ._speeds import SPEEDS, PHYSIO_FLOOR, V_MAX, A_MAX
from .models.wald import wald_pdf, wald_cdf, _fit_cell_mle, detect_bimodal

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
def _fit_va_fixed_t0(rt: np.ndarray, t0: float):
    """MLE of (v, a) with t0 held fixed. Returns (v, a, ks)."""
    tau = rt - t0
    tau = tau[tau > 1e-6]
    if tau.size < 10:
        return np.nan, np.nan, np.nan

    def nll(p):
        v, a = np.exp(p[0]), np.exp(p[1])
        if v > V_MAX or a > A_MAX:
            return 1e12
        w = wald_pdf(tau, v, a)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            return 1e12
        return float(-np.sum(np.log(w)))

    best = None
    for x0 in ([np.log(6), np.log(0.8)], [np.log(10), np.log(1.2)]):
        r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    v, a = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    ks = float(kstest(tau, lambda z: wald_cdf(z, v, a)).statistic)
    return v, a, ks


def fixed_t0_sensitivity(df: pd.DataFrame, effector: str,
                         t0_values_ms=(50, 70, 90)) -> pd.DataFrame:
    """
    Refit each cell with t0 fixed at each value and aggregate drift, boundary, and
    KS by speed. Shows that fixing t0 leaves drift and fit quality essentially
    unchanged -- the point of the saccadic fixed-t0 analysis.
    """
    sub = df[df["effector"] == effector]
    rows = []
    for t0_ms in t0_values_ms:
        t0 = t0_ms / 1000.0
        recs = []
        for (p, c), g in sub.groupby(["participant", "condition"]):
            rt = g["rt"].values.astype(float)
            if len(rt) < 15 or rt.min() <= t0:
                continue
            v, a, ks = _fit_va_fixed_t0(rt, t0)
            if np.isfinite(v):
                recs.append((int(c), v, a, ks))
        rr = pd.DataFrame(recs, columns=["condition", "v", "a", "ks"])
        for c in range(len(SPEEDS)):
            cc = rr[rr.condition == c]
            if len(cc):
                rows.append({"t0_fixed_ms": t0_ms, "condition": c, "speed": SPEEDS[c],
                             "v": cc["v"].mean(), "a": cc["a"].mean(),
                             "median_ks": cc["ks"].median(), "n_cells": len(cc)})
    return pd.DataFrame(rows)


def identifiability_sweep(df: pd.DataFrame, effector: str,
                          floors_ms=(35, 50, 70, 90, 110)) -> pd.DataFrame:
    """
    For each candidate floor, what fraction of cells have their free-t0 MLE below
    that floor (i.e. would be pinned to it)? A high fraction means t0 is
    determined by the floor, not the data.
    """
    sub = df[df["effector"] == effector]
    free_t0 = []
    for (p, c), g in sub.groupby(["participant", "condition"]):
        rt = g["rt"].values.astype(float)
        if len(rt) < 15:
            continue
        v, a, t0 = _fit_cell_mle(rt, floor=0.0)      # unconstrained floor
        free_t0.append((int(c), t0 * 1000.0))
    ft = pd.DataFrame(free_t0, columns=["condition", "t0_ms"])
    rows = []
    for fl in floors_ms:
        for c in range(len(SPEEDS)):
            cc = ft[ft.condition == c]
            if len(cc):
                rows.append({"floor_ms": fl, "condition": c, "speed": SPEEDS[c],
                             "pct_below_floor": 100.0 * float(np.mean(cc["t0_ms"] < fl)),
                             "n_cells": len(cc)})
    return pd.DataFrame(rows)


def mixture_threshold_sensitivity(df: pd.DataFrame, effector: str = "eye",
                                  alphas=(0.01, 0.05, 0.10)) -> pd.DataFrame:
    """Count bimodal cells at several dip-test significance levels (stability check)."""
    try:
        import diptest
    except Exception:
        return pd.DataFrame([{"note": "diptest not installed; install it for this analysis."}])
    sub = df[df["effector"] == effector]
    cells = []
    for (p, c), g in sub.groupby(["participant", "condition"]):
        x = g["rt"].values.astype(float)
        if len(x) >= 40:
            cells.append((int(c), float(diptest.diptest(x * 1000.0)[1])))
    dc = pd.DataFrame(cells, columns=["condition", "dip_p"])
    rows = []
    for a in alphas:
        for c in range(len(SPEEDS)):
            cc = dc[dc.condition == c]
            if len(cc):
                rows.append({"alpha": a, "condition": c, "speed": SPEEDS[c],
                             "n_bimodal": int((cc["dip_p"] < a).sum()), "n_cells": len(cc)})
        rows.append({"alpha": a, "condition": "all", "speed": np.nan,
                     "n_bimodal": int((dc["dip_p"] < a).sum()), "n_cells": len(dc)})
    return pd.DataFrame(rows)


def vincentiles(df: pd.DataFrame, effector: str, n_bins: int = 10) -> pd.DataFrame:
    """
    Model-free vincentiles: average the within-participant quantiles across
    participants, per speed. A distribution-shape summary that needs no model.
    """
    sub = df[df["effector"] == effector]
    qs = (np.arange(1, n_bins + 1) - 0.5) / n_bins
    rows = []
    for c in range(len(SPEEDS)):
        per_p = []
        cc = sub[sub.condition == c]
        for pid, g in cc.groupby("participant"):
            rt = g["rt"].values * 1000.0
            if len(rt) >= n_bins:
                per_p.append(np.quantile(rt, qs))
        if per_p:
            vinc = np.mean(np.vstack(per_p), axis=0)
            for q, val in zip(qs, vinc):
                rows.append({"condition": c, "speed": SPEEDS[c], "quantile": round(float(q), 3),
                             "rt_ms": float(val)})
    return pd.DataFrame(rows)


def parameter_recovery(n_participants: int = 12, trials_per_cell: int = 120,
                       seed: int = 0) -> dict:
    """
    Simulate hand and saccade data from known parameters, refit by MLE, and report
    recovery. Demonstrates quantitatively that hand t0 is identified and saccadic
    t0 is not (it is pushed to / below the floor).
    """
    from . import data, filters
    d = data.simulate_dataset(n_participants=n_participants, trials_per_cell=trials_per_cell,
                              seed=seed, express_fraction_eye=0.0)
    tidy = data.load_trials(d, "Participant", hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                            speed_col="Speed_deg_per_s", blocktype_col="BlockType")
    kept, _ = filters.apply_windows(tidy)
    truth = {"hand": {0: 170, 1: 158, 2: 148}, "eye": {0: 30, 1: 30, 2: 30}}  # true t0 (ms)
    out = {}
    from .models.wald import mle_preview
    for eff in ("hand", "eye"):
        grp = mle_preview(kept, eff)["group"]
        grp = grp.assign(true_t0_ms=[truth[eff][c] for c in grp["condition"]])
        grp["t0_error_ms"] = grp["t0_ms"] - grp["true_t0_ms"]
        out[eff] = grp[["speed", "true_t0_ms", "t0_ms", "t0_error_ms", "pct_floored"]].round(1)
    return out
