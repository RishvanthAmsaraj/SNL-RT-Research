"""
Frequentist Method A: per-cell shifted-Wald MLE by differential evolution, with a
uniform contamination component -- matching the repository's DDM_fit.py.

This is the frequentist counterpart to the hierarchical Bayesian fit. Running both
lets you show Method A and Method B agree (a standard robustness check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution

from ._speeds import SPEEDS, PHYSIO_FLOOR, V_MAX, A_MAX, P_CONTAM
from .models.wald import wald_pdf, wald_cdf


def fit_single_cell(rts: np.ndarray, floor: float, contam: float = P_CONTAM, maxiter: int = 200):
    """Differential-evolution MLE of (v, a, t0) with contamination. Returns (v, a, t0, ks)."""
    Tr = float(rts.max() - rts.min()) or 1e-3

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
        return float(-np.sum(np.log(d)))

    hi = max(np.percentile(rts, 3) - 0.002, floor + 1e-3)
    b = [(0.1, V_MAX), (0.05, A_MAX), (floor, hi)]
    best = None
    for s in (42, 7):
        r = differential_evolution(nll, b, seed=s, maxiter=maxiter, tol=1e-8,
                                   popsize=12, polish=True)
        if best is None or r.fun < best.fun:
            best = r
    v, a, t0 = best.x
    adj = rts - t0
    ks = float(stats.kstest(adj, lambda z: wald_cdf(z, v, a)).statistic)
    return float(v), float(a), float(t0), ks


def fit_ddm(df: pd.DataFrame, effector: str, contamination: float = P_CONTAM,
            maxiter: int = 200, status=lambda m: None) -> dict:
    """Per-cell frequentist fit across participant x speed cells; returns cell + group tables."""
    sub = df[df["effector"] == effector]
    floor = PHYSIO_FLOOR[effector]
    rows = []
    cells = list(sub.groupby(["participant", "condition"]))
    for i, ((p, c), g) in enumerate(cells):
        rt = g["rt"].values.astype(float)
        if len(rt) < 15:
            continue
        if i % 10 == 0:
            status(f"{effector}: frequentist MLE cell {i+1}/{len(cells)}")
        v, a, t0, ks = fit_single_cell(rt, floor, contamination, maxiter)
        rows.append({"participant": p, "condition": int(c), "speed": SPEEDS[int(c)],
                     "v": round(v, 3), "a": round(a, 4), "t0_ms": round(t0 * 1000.0),
                     "ks": round(ks, 4), "floored": t0 * 1000.0 <= floor * 1000.0 + 2})
    cell = pd.DataFrame(rows)
    group = (cell.groupby("condition").agg(v=("v", "mean"), a=("a", "mean"),
                                           t0_ms=("t0_ms", "mean"), median_ks=("ks", "median"),
                                           pct_floored=("floored", "mean")).reset_index())
    group["speed"] = group["condition"].map(lambda c: SPEEDS[int(c)])
    group["pct_floored"] *= 100
    return {"cell": cell, "group": group, "floor_ms": floor * 1000.0, "method": "frequentist"}
