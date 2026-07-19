"""
Publication-style figures matching the repository's house style: the per-speed
palette (green / red-pink / blue with darker matched lines), Arial where present,
editable PDF text (fonttype 42), and 300 DPI on save.

Each function returns a Matplotlib Figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress

from ._speeds import SPEEDS, SPEED_RGB
from .models.wald import wald_pdf

# Arial if available, else a clean sans fallback.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 110,
})

FILL = {c: SPEED_RGB[int(SPEEDS[c])] for c in range(len(SPEEDS))}


def _line(c):
    r, g, b = FILL[c]
    return (r * 0.55, g * 0.55, b * 0.55)


def _spd(c):
    return f"{int(SPEEDS[c])} deg/s"


def fit_overlay(df: pd.DataFrame, effector: str, group: pd.DataFrame):
    """Pooled RT histogram per condition with the fitted shifted-Wald density."""
    sub = df[df["effector"] == effector]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for c in range(len(SPEEDS)):
        ax = axes[c]
        rt = sub[sub["condition"] == c]["rt"].values * 1000.0
        if rt.size:
            ax.hist(rt, bins=40, density=True, color=FILL[c], edgecolor="white", linewidth=0.3)
        row = group[group["condition"] == c]
        if len(row):
            v, a, t0 = row.iloc[0]["v"], row.iloc[0]["a"], row.iloc[0]["t0_ms"] / 1000.0
            xs = np.linspace(max(rt.min() if rt.size else 100, 1), rt.max() if rt.size else 700, 400) / 1000.0
            pdf = np.where(xs > t0, wald_pdf(np.maximum(xs - t0, 1e-6), v, a), 0.0)
            ax.plot(xs * 1000.0, pdf / 1000.0, color=_line(c), linewidth=1.8)
            ax.axvline(t0 * 1000.0, color=_line(c), linestyle=":", linewidth=1.0)
        ax.set_title(_spd(c), fontsize=10)
        ax.set_xlabel("RT (ms)", fontsize=9)
    axes[0].set_ylabel("density", fontsize=9)
    fig.suptitle(f"{effector.capitalize()} RT: data and fitted shifted-Wald", fontsize=11)
    fig.tight_layout()
    return fig


def ndt_dots(units_df: pd.DataFrame, group: pd.DataFrame, effector: str, floor_ms: float):
    """Non-decision time by speed: participant dots + group mean, on a zoomed axis."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    rng = np.random.default_rng(0)
    for c in range(len(SPEEDS)):
        pts = units_df[units_df["condition"] == c]["t0_ms"].values
        if pts.size:
            x = c + rng.uniform(-0.13, 0.13, size=len(pts))
            ax.scatter(x, pts, s=20, color=FILL[c], edgecolor=_line(c), linewidth=0.5, zorder=2)
        row = group[group["condition"] == c]
        if len(row):
            gm = row.iloc[0]["t0_ms"]
            ax.plot([c - 0.22, c + 0.22], [gm, gm], color=_line(c), linewidth=2.6, zorder=3)
    ax.axhline(floor_ms, color="0.4", linestyle="--", linewidth=1.0)
    ax.text(2.42, floor_ms + 1.5, "physiological floor", fontsize=8, color="0.4", ha="right")
    ax.set_xticks(range(len(SPEEDS)))
    ax.set_xticklabels([_spd(c) for c in range(len(SPEEDS))])
    ax.set_ylabel("non-decision time (ms)", fontsize=9)
    ax.set_title(f"{effector.capitalize()} non-decision time by speed", fontsize=11)
    fig.tight_layout()
    return fig


def reciprobit(later_result: dict, df_eye: pd.DataFrame):
    """Per-participant LATER lines (central 10-90% fit), express-dominant highlighted."""
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    pp = later_result["per_participant"].set_index("participant")
    xt_lat = [500, 300, 200, 150, 120, 100]
    xt_pos = [1000.0 / t for t in xt_lat]
    for pid, sub in df_eye.groupby("participant"):
        rt = sub["rt"].values.astype(float) * 1000.0
        rt = rt[(rt >= 80) & (rt <= 600)]
        if rt.size < 40:
            continue
        rate = np.sort(1000.0 / rt)
        n = rate.size
        z = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        express = bool(pp.loc[pid, "express_dominant"]) if pid in pp.index else False
        col = "#b00" if express else "0.6"
        a, b = int(0.10 * n), int(0.90 * n)
        sl, ic, _, _, _ = linregress(rate[a:b], z[a:b])
        xs = np.linspace(rate[a], rate[b - 1], 40)
        ax.plot(xs, ic + sl * xs, color=col, alpha=0.9 if express else 0.5,
                linewidth=1.3 if express else 0.9, zorder=3 if express else 2)
    ax.set_xticks(xt_pos); ax.set_xticklabels(xt_lat)
    ax.set_xlabel("saccadic latency (ms, reciprocal scale)", fontsize=9)
    yt = [norm.ppf(p) for p in [.05, .25, .5, .75, .95]]
    ax.set_yticks(yt); ax.set_yticklabels(["5%", "25%", "50%", "75%", "95%"])
    ax.set_ylabel("cumulative probability (probit)", fontsize=9)
    ax.grid(True, ls="--", alpha=0.3)
    ax.plot([], [], color="#b00", label="express-dominant")
    ax.plot([], [], color="0.6", label="other participants")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"LATER reciprobit  (median $R^2$ = {later_result['median_r2']:.2f})", fontsize=11)
    fig.tight_layout()
    return fig


def why_floors(df: pd.DataFrame):
    """Flooring diagnostic: skew/CV per effector against the pure-Wald value of 3."""
    from .data import cell_summary
    cs = cell_summary(df)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    marker = {"hand": "o", "eye": "s"}
    for eff in cs["effector"].unique():
        for _, r in cs[cs["effector"] == eff].iterrows():
            c = int(r["condition"])
            ax.scatter(r["skew_over_cv"], r["skew"], s=70, color=FILL[c],
                       edgecolor="0.3", linewidth=0.6, marker=marker.get(eff, "o"), zorder=3)
    ax.axvline(3.0, color="0.3", linestyle="--", linewidth=1.0)
    ax.text(3.1, ax.get_ylim()[1] * 0.92, "pure Wald (skew = 3 CV)", fontsize=8, color="0.3")
    ax.set_xlabel("skew / CV", fontsize=9); ax.set_ylabel("skewness", fontsize=9)
    ax.plot([], [], marker="o", color="0.5", linestyle="none", label="hand")
    ax.plot([], [], marker="s", color="0.5", linestyle="none", label="eye")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Why saccadic $t_0$ floors: distribution shape", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figures for the advanced analyses
# --------------------------------------------------------------------------- #
def vincentile_plot(vinc_df: pd.DataFrame, effector: str = ""):
    """Model-free vincentiles: quantile-averaged RT vs quantile, one line per speed."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for c in range(len(SPEEDS)):
        g = vinc_df[vinc_df["condition"] == c]
        if len(g):
            ax.plot(g["quantile"], g["rt_ms"], "-o", ms=4, color=_line(c), label=_spd(c))
    ax.set_xlabel("within-participant quantile", fontsize=9)
    ax.set_ylabel("RT (ms)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Vincentiles{' — ' + effector if effector else ''} (model-free)", fontsize=11)
    fig.tight_layout()
    return fig


def fixed_t0_plot(sens_df: pd.DataFrame, effector: str = ""):
    """Drift and KS as t0 is fixed at several values -- shows the fit is stable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for c in range(len(SPEEDS)):
        g = sens_df[sens_df["condition"] == c].sort_values("t0_fixed_ms")
        if len(g):
            ax1.plot(g["t0_fixed_ms"], g["v"], "-o", ms=4, color=_line(c), label=_spd(c))
            ax2.plot(g["t0_fixed_ms"], g["median_ks"], "-o", ms=4, color=_line(c), label=_spd(c))
    ax1.set_xlabel("fixed $t_0$ (ms)", fontsize=9); ax1.set_ylabel("drift v", fontsize=9)
    ax2.set_xlabel("fixed $t_0$ (ms)", fontsize=9); ax2.set_ylabel("median KS", fontsize=9)
    ax1.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Fixed-$t_0$ sensitivity{' — ' + effector if effector else ''}", fontsize=11)
    fig.tight_layout()
    return fig


def identifiability_plot(sweep_df: pd.DataFrame, effector: str = ""):
    """Fraction of cells pinned below the floor as the floor moves."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for c in range(len(SPEEDS)):
        g = sweep_df[sweep_df["condition"] == c].sort_values("floor_ms")
        if len(g):
            ax.plot(g["floor_ms"], g["pct_below_floor"], "-o", ms=4, color=_line(c), label=_spd(c))
    ax.axhline(50, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("candidate floor (ms)", fontsize=9)
    ax.set_ylabel("% of cells below the floor", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Identifiability{' — ' + effector if effector else ''}: t$_0$ pinned by the floor", fontsize=11)
    fig.tight_layout()
    return fig


def dissociation_plot(diss: dict):
    """Bootstrap mean t0 change (slow minus fast speed) with 95% CI, hand vs eye."""
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ys, labels = [], []
    for i, eff in enumerate(["hand", "eye"]):
        if eff in diss and "bootstrap" in diss[eff] and "mean_diff_ms" in diss[eff]["bootstrap"]:
            b = diss[eff]["bootstrap"]
            m, (lo, hi) = b["mean_diff_ms"], b["ci95_ms"]
            col = "#333" if eff == "hand" else "#b00"
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", ms=9, color=col,
                        capsize=6, lw=2)
            ys.append(i); labels.append(eff)
    ax.axvline(0, color="0.6", ls="--", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("t$_0$ change, slow − fast speed (ms)", fontsize=9)
    ax.set_title("Non-decision-time dissociation (bootstrap 95% CI)", fontsize=11)
    fig.tight_layout()
    return fig
