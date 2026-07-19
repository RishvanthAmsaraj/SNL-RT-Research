"""
Statistical tests for the non-decision-time-by-speed dissociation.

The headline result is that hand t0 changes with target speed while saccadic t0
does not (it is pinned at the floor). These tests support that claim with a
repeated-measures Friedman test, a participant-resampling bootstrap of the
speed effect, and a within-participant permutation test.

Each takes a tidy table with columns [participant, condition, t0_ms].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


def _wide(t0_table: pd.DataFrame) -> pd.DataFrame:
    """Participants (rows) x condition (cols) t0, keeping only complete participants."""
    w = t0_table.pivot_table(index="participant", columns="condition", values="t0_ms",
                             aggfunc="mean")
    return w.dropna()


def friedman_t0_by_speed(t0_table: pd.DataFrame) -> dict:
    w = _wide(t0_table)
    if w.shape[0] < 3 or w.shape[1] < 2:
        return {"n": int(w.shape[0]), "statistic": float("nan"), "p_value": float("nan"),
                "note": "Need >=3 participants present in all speeds."}
    stat, p = friedmanchisquare(*[w[c].values for c in w.columns])
    return {"n": int(w.shape[0]), "statistic": float(stat), "p_value": float(p),
            "conditions": [float(c) for c in w.columns]}


def bootstrap_speed_effect(t0_table: pd.DataFrame, lo_cond=0, hi_cond=2,
                           n_boot: int = 5000, seed: int = 0) -> dict:
    """
    Bootstrap the mean t0 difference (slow minus fast speed) by resampling
    participants. Returns the point estimate and a 95% percentile interval.
    """
    w = _wide(t0_table)
    if lo_cond not in w.columns or hi_cond not in w.columns or w.shape[0] < 3:
        return {"note": "Not enough data for the requested conditions."}
    diff = (w[hi_cond] - w[lo_cond]).values          # per participant
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "mean_diff_ms": float(diff.mean()), "ci95_ms": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "direction": f"speed {int(w.columns.max())} vs {int(w.columns.min())}"}


def permutation_test(t0_table: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> dict:
    """
    Within-participant permutation test: shuffle the speed labels for each
    participant, recompute the Friedman statistic, and compare to observed.
    """
    w = _wide(t0_table)
    if w.shape[0] < 3 or w.shape[1] < 2:
        return {"note": "Need >=3 participants present in all speeds."}
    mat = w.values
    obs, _ = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        pm = np.apply_along_axis(rng.permutation, 1, mat)
        stat, _ = friedmanchisquare(*[pm[:, j] for j in range(pm.shape[1])])
        if stat >= obs:
            count += 1
    return {"n": int(w.shape[0]), "observed_statistic": float(obs),
            "p_value": float((count + 1) / (n_perm + 1))}


def dissociation_report(units_hand: pd.DataFrame | None, units_eye: pd.DataFrame | None,
                        n_boot: int = 3000) -> dict:
    """Run the full battery for whichever effectors are available."""
    out = {}
    for name, u in [("hand", units_hand), ("eye", units_eye)]:
        if u is None or u.empty or "t0_ms" not in u:
            continue
        t = u[["participant", "condition", "t0_ms"]]
        out[name] = {"friedman": friedman_t0_by_speed(t),
                     "bootstrap": bootstrap_speed_effect(t, n_boot=n_boot),
                     "permutation": permutation_test(t, n_perm=n_boot)}
    return out
