"""
Frequentist Method A -- the repository's DDM_fit.py, driven from the app.

The fitting itself lives in `models.wald` (`ddm_fit_single`, `ddm_fit_mixture`,
`ddm_select_srt`) and is a literal port of the standalone script: the same
differential-evolution optimiser, seeds, bounds, contamination term, and the same
rule for when a saccade cell is fitted as an express/regular mixture. This module
only walks the cells and assembles the tables, so running Method A here gives the
same numbers as running DDM_fit.py in an IDE.

This is the frequentist counterpart to the hierarchical Bayesian fit. Running both
lets you show Method A and Method B agree (a standard robustness check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._speeds import SPEEDS, PHYSIO_FLOOR, P_CONTAM
from .models.wald import ddm_fit_single, ddm_select_srt, MIN_TRIALS


def fit_single_cell(rts: np.ndarray, floor: float, contam: float = P_CONTAM):
    """MLE of (v, a, t0) with contamination for one cell. Returns (v, a, t0, ks)."""
    x, _, ks = ddm_fit_single(rts, floor, contam)
    return float(x[0]), float(x[1]), float(x[2]), float(ks)


def fit_ddm(df: pd.DataFrame, effector: str, contamination: float = P_CONTAM,
            status=lambda m: None) -> dict:
    """
    Per-cell frequentist fit across participant x speed cells.

    Hand cells are always fitted with a single Wald. Saccade cells go through the
    pipeline's selection rule: a single Wald unless it fits poorly (KS > 0.10), in
    which case an express/regular mixture is tried and accepted only if it fits
    well, splits the trials sensibly, and separates the modes by at least 30 ms.
    """
    sub = df[df["effector"] == effector]
    floor = PHYSIO_FLOOR[effector]
    rows = []
    cells = list(sub.groupby(["participant", "condition"]))
    for i, ((p, c), g) in enumerate(cells):
        rt = g["rt"].values.astype(float)
        rt = rt[np.isfinite(rt)]
        if len(rt) < MIN_TRIALS:
            continue
        if i % 5 == 0:
            status(f"{effector}: Method A cell {i + 1}/{len(cells)}")
        row = {"participant": p, "condition": int(c), "speed": SPEEDS[int(c)], "n": len(rt)}
        if effector == "hand":
            v, a, t0, ks = fit_single_cell(rt, floor, contamination)
            row.update({"model": "single", "v": round(v, 3), "a": round(a, 4),
                        "t0_ms": round(t0 * 1000.0), "ks": round(ks, 4),
                        "ks_single": round(ks, 4),
                        "floored": t0 * 1000.0 <= floor * 1000.0 + 2})
        else:
            sel = ddm_select_srt(rt, floor, contamination)
            row.update({"model": sel["model"], "ks": round(sel["ks"], 4),
                        "ks_single": round(sel["ks_single"], 4)})
            if sel["model"] == "mixture":
                m = sel["mixture"]
                # the regular component carries the comparable non-decision time
                row.update({"v": np.nan, "a": np.nan, "t0_ms": np.nan,
                            "pi": round(m["pi"], 3),
                            "ve": round(m["ve"], 3), "ae": round(m["ae"], 4),
                            "t0e_ms": round(m["t0e"] * 1000.0),
                            "vr": round(m["vr"], 3), "ar": round(m["ar"], 4),
                            "t0r_ms": round(m["t0r"] * 1000.0),
                            "express_mode": round(m["express_mode"]),
                            "reg_mode": round(m["reg_mode"]),
                            "floored": m["t0r"] * 1000.0 <= floor * 1000.0 + 2})
            else:
                row.update({"v": round(sel["v"], 3), "a": round(sel["a"], 4),
                            "t0_ms": round(sel["t0"] * 1000.0),
                            "floored": sel["t0"] * 1000.0 <= floor * 1000.0 + 2})
        rows.append(row)

    cell = pd.DataFrame(rows)
    if not len(cell):
        return {"cell": cell, "group": pd.DataFrame(), "floor_ms": floor * 1000.0,
                "method": "frequentist"}

    # Group means use the non-decision time that is comparable across cells: the
    # single-Wald t0, or the regular component's shift for a mixture cell -- which
    # is what NDT_barchart.py does when it builds its saccadic t0 table.
    if "t0r_ms" in cell.columns:
        t0_group = cell["t0_ms"].where(cell["model"] == "single", cell.get("t0r_ms"))
    else:
        t0_group = cell["t0_ms"]
    cell = cell.assign(_t0_group=t0_group)
    group = (cell.groupby("condition")
             .agg(v=("v", "mean"), a=("a", "mean"), t0_ms=("_t0_group", "mean"),
                  median_ks=("ks", "median"), pct_floored=("floored", "mean"),
                  n_mixture=("model", lambda s: int((s == "mixture").sum()))).reset_index())
    group["speed"] = group["condition"].map(lambda c: SPEEDS[int(c)])
    group["pct_floored"] *= 100
    cell = cell.drop(columns=["_t0_group"])
    return {"cell": cell, "group": group, "floor_ms": floor * 1000.0,
            "method": "frequentist"}
