"""
LATER model for saccadic latencies (Carpenter & Williams, 1995), matching the
repository's LATER_analysis.py.

Promptness (1/latency) is Gaussian, so the cumulative latency distribution is a
straight line on a reciprobit plot. The line is fit to the central 10-90% of the
distribution to avoid the express tail. LATER has no non-decision parameter, so
nothing can floor -- the saccade-native complement to the shifted-Wald fit.

Pure NumPy/SciPy; runs in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, linregress, kstest

from .._speeds import SPEEDS

EXPRESS_CUTOFF_MS = 130.0
MIN_TRIALS = 40


def _recipro(rt_ms: np.ndarray):
    """Return (sorted promptness in 1/s, probit of plotting positions)."""
    rate = np.sort(1000.0 / rt_ms)
    n = rate.size
    p = (np.arange(1, n + 1) - 0.5) / n
    return rate, norm.ppf(p)


def _fit_line(rate, z, lo=0.10, hi=0.90):
    """Fit the LATER line to the central population (avoids the express tail)."""
    n = len(rate)
    a, b = int(lo * n), int(hi * n)
    sl, ic, r, _, _ = linregress(rate[a:b], z[a:b])
    return float(sl), float(ic), float(r ** 2)


def fit_later(df_eye: pd.DataFrame, express_dominant_frac: float = 0.25) -> dict:
    """
    Fit LATER per participant x speed cell (matching the repository) and summarise.

    Returns dict with per_cell, per_participant, per_condition tables, plotting
    coords, and the headline median reciprobit r-squared.
    """
    rows, coords = [], {}
    for (pid, cond), sub in df_eye.groupby(["participant", "condition"]):
        rt = sub["rt"].values.astype(float) * 1000.0
        if len(rt) < MIN_TRIALS:
            continue
        rate = 1000.0 / rt
        mu, sd = float(rate.mean()), float(rate.std())
        r_s, z_s = _recipro(rt)
        _, _, r2 = _fit_line(r_s, z_s)
        ks = float(kstest(rate, lambda q: norm.cdf(q, mu, sd))[0])
        rows.append({"participant": pid, "condition": int(cond), "speed": SPEEDS[int(cond)],
                     "mu_rate": mu, "sd_rate": sd, "median_lat_ms": 1000.0 / mu,
                     "reciprobit_r2": r2, "ks": ks,
                     "express_frac": float((rt < EXPRESS_CUTOFF_MS).mean())})
        coords.setdefault(pid, {})[int(cond)] = {"rate": r_s, "z": z_s}

    per_cell = pd.DataFrame(rows)
    if per_cell.empty:
        return {"per_cell": per_cell, "per_participant": per_cell,
                "per_condition": per_cell, "coords": coords, "median_r2": float("nan")}

    per_participant = (per_cell.groupby("participant")
                       .agg(reciprobit_r2=("reciprobit_r2", "mean"),
                            express_frac=("express_frac", "mean"),
                            median_lat_ms=("median_lat_ms", "mean")).reset_index())
    per_participant["express_dominant"] = per_participant["express_frac"] >= express_dominant_frac

    per_condition = (per_cell.groupby("condition")
                     .agg(median_lat_ms=("median_lat_ms", "mean"),
                          reciprobit_r2=("reciprobit_r2", "median")).reset_index())
    per_condition["speed"] = per_condition["condition"].map(lambda c: SPEEDS[int(c)])

    return {"per_cell": per_cell, "per_participant": per_participant,
            "per_condition": per_condition, "coords": coords,
            "median_r2": float(per_cell["reciprobit_r2"].median())}
