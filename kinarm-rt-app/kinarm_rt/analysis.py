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
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from scipy.stats import kstest

from ._speeds import SPEEDS, PHYSIO_FLOOR, V_MAX, A_MAX, P_CONTAM
from .models.wald import (wald_pdf, wald_cdf, detect_bimodal, ddm_fit_single,
                          ddm_select_srt, MIN_TRIALS)

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
def _fit_va_fixed_t0(rt: np.ndarray, t0: float, contam: float = P_CONTAM):
    """
    MLE of (v, a) with t0 held fixed. Returns (v, a, ks).

    Port of SRT_fixed_t0_analysis.py::fit_va -- differential evolution over the
    same bounds, with the uniform contamination component. The cell is rejected
    (rather than trimmed) if any trial falls at or below the fixed t0, exactly as
    in the script: dropping those trials would change the sample being fitted.
    """
    rt = np.asarray(rt, float)
    adj = rt - t0
    if np.any(adj <= 0):
        return np.nan, np.nan, np.nan
    Tr = float(rt.max() - rt.min())

    def nll(p):
        v, a = p
        d = (1 - contam) * wald_pdf(adj, v, a) + (contam / Tr if contam > 0 else 0.0)
        if np.any(d <= 0) or not np.all(np.isfinite(d)):
            return 1e10
        return -np.sum(np.log(d))

    r = differential_evolution(nll, [(0.1, V_MAX), (0.05, A_MAX)], seed=42,
                               maxiter=300, tol=1e-9, popsize=12, polish=True)
    ks = float(kstest(adj, lambda z: wald_cdf(z, r.x[0], r.x[1])).statistic)
    return float(r.x[0]), float(r.x[1]), ks


def fixed_t0_sensitivity(df: pd.DataFrame, effector: str = "eye",
                         t0_values_ms=(50, 70, 90)) -> pd.DataFrame:
    """
    Refit each cell with t0 fixed at each value and aggregate drift, boundary, and
    KS by speed.

    Port of SRT_fixed_t0_analysis.py: fixing the saccadic non-decision time removes
    the floor-piling artifact, and the drift pattern across speeds and the fit
    quality are both essentially unchanged, so the conclusions do not depend on
    which value is assumed.
    """
    sub = df[df["effector"] == effector]
    rows = []
    for t0_ms in t0_values_ms:
        t0 = t0_ms / 1000.0
        recs = []
        for (p, c), g in sub.groupby(["participant", "condition"]):
            rt = g["rt"].values.astype(float)
            rt = rt[np.isfinite(rt)]
            if len(rt) < MIN_TRIALS or rt.min() <= t0:
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
                             "median_ks": cc["ks"].median(), "mean_ks": cc["ks"].mean(),
                             "n_cells": len(cc)})
    return pd.DataFrame(rows)


def identifiability_sweep(df: pd.DataFrame, effector: str = "eye",
                          floors_ms=(40, 50, 60, 70, 80, 90),
                          slope_threshold: float = 0.7) -> pd.DataFrame:
    """
    Refit each cell at a range of imposed non-decision floors and measure how
    closely the fitted t0 follows the floor.

    Port of SRT_identifiability_check.py. A cell whose t0 moves one-for-one with
    the floor (slope near 1) is not identified by the data -- the floor is setting
    it. A cell whose t0 stays put regardless of the floor is genuinely identified.
    One row per cell, with the fitted t0 at each floor and the fitted slope.
    """
    sub = df[df["effector"] == effector]
    floor_s = [f / 1000.0 for f in floors_ms]
    rows = []
    for (p, c), g in sub.groupby(["participant", "condition"]):
        rt = g["rt"].values.astype(float)
        rt = rt[np.isfinite(rt)]
        if len(rt) < MIN_TRIALS:
            continue
        # only single-component cells are diagnostic here, matching the script,
        # which reads the single/mixture split from the DDM fit table
        if effector == "eye" and ddm_select_srt(rt, PHYSIO_FLOOR[effector])["model"] == "mixture":
            continue
        t0s = []
        for fl in floor_s:
            x, _, _ = ddm_fit_single(rt, fl, P_CONTAM)
            t0s.append(float(x[2]) * 1000.0)
        slope = float(np.polyfit(list(floors_ms), t0s, 1)[0])
        row = {"participant": p, "condition": int(c), "speed": SPEEDS[int(c)],
               "slope": slope, "tracks_floor": bool(slope > slope_threshold)}
        for f, t in zip(floors_ms, t0s):
            row[f"t0_at_{int(f)}"] = t
        rows.append(row)
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
