"""
Physiological inclusion windows, matching the repository (CODE_REFERENCE.md).

  * Hand: keep 150-800 ms. Anticipation removal (Whelan 2008) and lapse removal
    (Luce 1986).
  * Saccades (eye): keep 80-600 ms. The 80 ms lower bound is the physiological
    minimum saccadic latency (Fischer & Weber 1993); the upper bound trims lapses.

Trials outside the window are removed. A few genuine anticipations that survive
inside the window are handled by the model, not by tightening the window.
"""

from __future__ import annotations

import pandas as pd

from ._speeds import FILTER_WINDOWS as DEFAULT_WINDOWS  # (low, high) in seconds


def apply_windows(df: pd.DataFrame, windows: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, report). `report` has one row per effector with the counts removed."""
    windows = {**DEFAULT_WINDOWS, **(windows or {})}
    kept_parts, rep_rows = [], []
    for eff, g in df.groupby("effector"):
        lo, hi = windows.get(eff, (0.0, 1e9))
        below = int((g["rt"] < lo).sum())
        above = int((g["rt"] > hi).sum())
        keep = g[(g["rt"] >= lo) & (g["rt"] <= hi)]
        kept_parts.append(keep)
        rep_rows.append({
            "effector": eff, "window_lo_ms": lo * 1000, "window_hi_ms": hi * 1000,
            "n_in": len(g), "removed_below": below, "removed_above": above,
            "n_kept": len(keep), "pct_kept": 100 * len(keep) / max(len(g), 1),
        })
    kept = pd.concat(kept_parts).reset_index(drop=True) if kept_parts else df
    return kept, pd.DataFrame(rep_rows)
