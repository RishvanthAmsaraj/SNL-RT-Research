"""
Diagnostics: goodness-of-fit (KS per cell) and a convergence summary.

Convergence (R-hat, divergences) is computed during fitting and carried on the
result dict from models.wald.fit_effector; `convergence_summary` just formats it.
Goodness of fit is computed here from the fitted per-cell (v, a, t0) table using
the same Wald CDF as the repository's DDM fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._speeds import SPEEDS
from .models.wald import wald_cdf


def goodness_of_fit(df: pd.DataFrame, effector: str, units_df: pd.DataFrame) -> dict:
    """KS distance between each cell's empirical RTs and its fitted shifted-Wald."""
    if units_df is None or units_df.empty:
        return {"cell": pd.DataFrame(), "by_condition": pd.DataFrame(), "median_ks": float("nan")}
    sub = df[df["effector"] == effector]
    params = units_df.set_index(["participant", "condition"])
    rows = []
    for (p, c), g in sub.groupby(["participant", "condition"]):
        key = (p, int(c))
        if key not in params.index:
            continue
        pr = params.loc[key]
        v, a, t0 = float(pr["v"]), float(pr["a"]), float(pr["t0_ms"]) / 1000.0
        rt = np.sort(g["rt"].values)
        tau = rt - t0
        ok = tau > 0
        if ok.sum() < 5:
            continue
        n = int(ok.sum())
        Fmodel = wald_cdf(tau[ok], v, a)
        Fhi = np.arange(1, n + 1) / n
        Flo = np.arange(0, n) / n
        ks = float(np.max(np.maximum(np.abs(Fhi - Fmodel), np.abs(Flo - Fmodel))))
        rows.append({"participant": p, "condition": int(c), "speed": SPEEDS[int(c)],
                     "n": n, "ks": ks})
    cell = pd.DataFrame(rows)
    if cell.empty:
        return {"cell": cell, "by_condition": pd.DataFrame(), "median_ks": float("nan")}
    by_cond = (cell.groupby("condition").agg(median_ks=("ks", "median"),
                                             max_ks=("ks", "max")).reset_index())
    by_cond["speed"] = by_cond["condition"].map(lambda c: SPEEDS[int(c)])
    return {"cell": cell, "by_condition": by_cond, "median_ks": float(cell["ks"].median())}


def convergence_summary(result: dict) -> dict:
    """Format the convergence info already attached to a fit_effector result."""
    conv = result.get("convergence", {})
    return {"max_rhat": conv.get("max_rhat", float("nan")),
            "n_divergences": conv.get("n_divergences", 0),
            "converged": conv.get("converged", False)}
