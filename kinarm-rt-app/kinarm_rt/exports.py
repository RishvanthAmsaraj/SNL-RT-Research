"""
Export fitted results to the repository's exact CSV schemas, so the app's output
is a drop-in replacement for Bayesian_hrt_fits.csv / Bayesian_srt_fits.csv and can
feed the repository's downstream figure and NDT scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_hrt_fits_csv(result: dict) -> pd.DataFrame:
    """Match Bayesian_hrt_fits.csv: pid, spd, v, a, t0, t0_lo95, t0_hi95, conv_div, conv_rhat."""
    u = result.get("units")
    conv = result.get("convergence", {})
    if u is None or u.empty:
        return pd.DataFrame(columns=["pid", "spd", "v", "a", "t0", "t0_lo95", "t0_hi95",
                                     "conv_div", "conv_rhat"])
    out = pd.DataFrame({
        "pid": u["participant"], "spd": u["speed"].astype(int),
        "v": u["v"], "a": u["a"], "t0": u["t0_ms"].astype(int),
        "t0_lo95": u.get("t0_lo95", np.nan), "t0_hi95": u.get("t0_hi95", np.nan),
        "conv_div": conv.get("n_divergences", 0), "conv_rhat": round(conv.get("max_rhat", float("nan")), 3),
    })
    return out.sort_values(["spd", "pid"]).reset_index(drop=True)


def to_srt_fits_csv(result: dict) -> pd.DataFrame:
    """Match Bayesian_srt_fits.csv (single + mixture rows)."""
    cols = ["pid", "spd", "n", "model", "conv_div", "conv_rhat", "v", "a", "t0",
            "t0_lo95", "t0_hi95", "pi", "pi_lo95", "pi_hi95",
            "express_mode", "express_mode_lo95", "express_mode_hi95",
            "reg_mode", "reg_mode_lo95", "reg_mode_hi95"]
    conv = result.get("convergence", {})
    rows = []
    u = result.get("units")
    if u is not None and len(u):
        for _, r in u.iterrows():
            rows.append({"pid": r["participant"], "spd": int(r["speed"]), "n": np.nan,
                         "model": "single", "conv_div": conv.get("n_divergences", 0),
                         "conv_rhat": round(conv.get("max_rhat", float("nan")), 3),
                         "v": r["v"], "a": r["a"], "t0": int(r["t0_ms"]),
                         "t0_lo95": r.get("t0_lo95", np.nan), "t0_hi95": r.get("t0_hi95", np.nan)})
    mix = result.get("mixture")
    if mix is not None and len(mix):
        for _, r in mix.iterrows():
            rows.append({"pid": r["participant"], "spd": int(r["speed"]), "n": r.get("n", np.nan),
                         "model": "mixture", "conv_div": r.get("conv_div", 0),
                         "conv_rhat": r.get("conv_rhat", np.nan),
                         "pi": r.get("pi"), "pi_lo95": r.get("pi_lo95"), "pi_hi95": r.get("pi_hi95"),
                         "express_mode": r.get("express_mode"), "express_mode_lo95": r.get("express_mode_lo95"),
                         "express_mode_hi95": r.get("express_mode_hi95"), "reg_mode": r.get("reg_mode"),
                         "reg_mode_lo95": r.get("reg_mode_lo95"), "reg_mode_hi95": r.get("reg_mode_hi95")})
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].sort_values(["spd", "pid"]).reset_index(drop=True) if len(df) else pd.DataFrame(columns=cols)
