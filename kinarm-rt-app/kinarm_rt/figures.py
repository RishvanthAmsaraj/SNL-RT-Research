"""
Publication-style figures matching the repository's house style: the per-speed
palette (green / red-pink / blue with darker matched lines), Arial where present,
editable PDF text (fonttype 42), and 300 DPI on save.

Each figure function returns a Matplotlib Figure. The multi-panel suites
(ddm_schematic_figs, vincentile_suite) return a list of (label, Figure) so the app
can show each one and name it in the report bundle.

These reproduce the standalone pipeline scripts one-for-one, driven by the app's
own fitted results instead of the pre-saved CSVs:
    ddm_schematic_figs  <- DDM_conceptual.py     (single-boundary diffusion schematics)
    vincentile_suite    <- vincentile_figures.py (KDE overlay, histograms, vincentiles)
    ndt_by_speed        <- NDT_barchart.py        (HRT/SRT non-decision time, dots + mean)
    why_floors          <- why_saccadic_t0_floors.py (shape-implied t0 vs floor)
    reciprobit          <- LATER_analysis.py       (reciprobit + latency by speed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import norm, linregress, gaussian_kde, skew, friedmanchisquare

from ._speeds import SPEEDS, SPEED_RGB, PHYSIO_FLOOR, EFFECTORS
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

# Saturated per-speed line colours and light fills, matching the standalone scripts.
SPEED_LINE_HEX = {0: "#4a7c59", 75: "#9e4a4a", 150: "#496aa3"}
SPEED_FILL_HEX = {0: "#cfe8cf", 75: "#f3d4d4", 150: "#d3ddef"}
DDM_COL_HEX = {0: "#4a7c59", 75: "#9e5a5a", 150: "#496aa3"}
SRT_C, HRT_C = "#2c7fb8", "#d9772b"   # saccade cool blue, hand warm orange (vincentile figs)
EFF_LABEL = {"hand": "HRT", "eye": "SRT"}
EFF_LONG = {"hand": "HAND  (manual reaction time)", "eye": "EYE  (saccadic reaction time)"}


def _line(c):
    r, g, b = FILL[c]
    return (r * 0.55, g * 0.55, b * 0.55)


def _spd(c):
    return f"{int(SPEEDS[c])} deg/s"


def _rt_ms(df: pd.DataFrame, effector: str, condition: int | None = None) -> np.ndarray:
    """Reaction times in ms for one effector (optionally one condition)."""
    sub = df[df["effector"] == effector]
    if condition is not None:
        sub = sub[sub["condition"] == condition]
    return sub["rt"].values.astype(float) * 1000.0


# --------------------------------------------------------------------------- #
# Data + fitted density overlay (kept)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# DDM conceptual schematic  (port of DDM_conceptual.py :: draw)
# --------------------------------------------------------------------------- #
def _ddm_schematic_one(v, a, t0_ms, med_ms, effector, condition):
    """One annotated single-boundary diffusion schematic (group-mean parameters)."""
    xmax = 600 if effector == "hand" else 500
    color = DDM_COL_HEX[int(SPEEDS[condition])]
    dist_label = f"{EFF_LABEL[effector]} Distribution\n(Interception)"

    fig, ax = plt.subplots(figsize=(11, 6))
    y_base, y_thr = 0.34, 0.72
    ax.add_patch(plt.Rectangle((0, 0.17), t0_ms, 0.66, facecolor="#ededed", edgecolor="none", zorder=0))
    ax.plot([0, xmax], [y_thr, y_thr], color=color, lw=2.6, zorder=4)
    ax.text(xmax, y_thr + 0.012, f"Response threshold ($a$ = {a:.2f})", ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=color)
    ax.plot([t0_ms, xmax], [y_base, y_base], color="#9a9a9a", ls=(0, (5, 4)), lw=1.0, zorder=1)
    ax.text(xmax, y_base - 0.03, "Baseline", ha="right", va="top", fontsize=9, style="italic", color="#9a9a9a")

    tt = np.linspace(t0_ms + 1, xmax, 400)
    dens = wald_pdf((tt - t0_ms) / 1000.0, v, a)
    if dens.max() > 0:
        dens = dens / dens.max()
    band_lo, band_h = y_thr + 0.015, 0.30
    ax.fill_between(tt, band_lo, band_lo + dens * band_h, color=color, alpha=0.22, zorder=2)
    ax.plot(tt, band_lo + dens * band_h, color=color, lw=1.6, zorder=3)
    ax.text(xmax * 0.66, band_lo + band_h * 0.55, dist_label, ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=color)

    # noisy accumulation paths from the starting point to threshold
    rng = np.random.default_rng(42 + condition + (0 if effector == "hand" else 7))
    med = max(med_ms, t0_ms + 20)
    for k in range(4):
        n = 160
        t_end = med * (0.85 + 0.18 * rng.random())
        ttp = np.linspace(t0_ms, t_end, n)
        ramp = np.linspace(y_base, y_thr, n)
        noise = np.cumsum(rng.standard_normal(n)) * 0.030
        noise -= np.linspace(noise[0], noise[-1], n)
        path = np.clip(ramp + noise, y_base - 0.10, y_thr)
        if k == 0:
            ax.plot(ttp, path, color=color, lw=1.7, alpha=0.95, zorder=5, solid_capstyle="round")
        else:
            ax.plot(ttp, path, color=color, lw=0.8, alpha=0.28, zorder=3)

    ax.plot(t0_ms, y_base, "o", color="#1a1a1a", ms=8, zorder=6)
    ax.text(t0_ms - xmax * 0.012, y_base + 0.085, "Starting\npoint ($z$)", ha="right", va="center", fontsize=9.5)
    ax.annotate("", xy=(t0_ms + (med - t0_ms) * 0.42, 0.555), xytext=(t0_ms + 4, y_base + 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#111"), zorder=7)
    ax.text(t0_ms + (med - t0_ms) * 0.46, 0.55, f"Drift rate ($v$ = {v:.2f})", fontsize=10, fontweight="bold", va="center")
    xa = xmax * 0.052
    ax.annotate("", xy=(xa, y_thr), xytext=(xa, y_base), arrowprops=dict(arrowstyle="<->", lw=1.6, color="#1a1a1a"))
    ax.text(xa - xmax * 0.012, (y_base + y_thr) / 2, "Threshold\nheight ($a$)", ha="right", va="center", fontsize=9.5)
    ax.annotate("", xy=(xmax * 1.02, 0.17), xytext=(0, 0.17), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333"))
    for tk in range(0, xmax + 1, 100):
        ax.plot([tk, tk], [0.17, 0.155], color="#333", lw=1)
        ax.text(tk, 0.125, f"{tk}", ha="center", va="top", fontsize=8.5, color="#333")
    ax.text(xmax / 2, 0.06, "Time (ms)", ha="center", va="top", fontsize=10.5)
    ax.plot([t0_ms, t0_ms], [0.17, 0.145], color="#333", lw=1.4)
    ax.text(t0_ms, 0.125, f"{t0_ms:.0f} ms\n($t_0$)", ha="center", va="top", fontsize=8.5, fontweight="bold")
    yb = 0.015
    ax.annotate("", xy=(t0_ms, yb), xytext=(0, yb), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#555"))
    ax.text(t0_ms / 2, yb - 0.055, f"Non-decision time ($t_0$) = {t0_ms:.0f} ms", ha="center", va="top",
            fontsize=9, style="italic", color="#555")
    ax.annotate("", xy=(xmax, yb), xytext=(t0_ms, yb), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#999"))
    ax.text((t0_ms + xmax) / 2, yb - 0.055, "Decision time", ha="center", va="top", fontsize=9, style="italic", color="#999")

    fig.suptitle(f"DDM (Single Boundary, {EFF_LABEL[effector]}) -- {int(SPEEDS[condition])} deg/s",
                 fontsize=12.5, fontweight="bold", y=0.99)
    ax.set_title(f"Group mean:  $v$ = {v:.2f},   $a$ = {a:.2f},   $t_0$ = {t0_ms:.0f} ms", fontsize=11, pad=10)
    ax.set_xlim(-xmax * 0.02, xmax * 1.06)
    ax.set_ylim(-0.10, 1.05)
    ax.axis("off")
    fig.tight_layout()
    return fig


def ddm_schematic_figs(df: pd.DataFrame, group: pd.DataFrame, effector: str):
    """One conceptual diffusion schematic per speed (list of (label, Figure))."""
    out = []
    for c in range(len(SPEEDS)):
        row = group[group["condition"] == c]
        if not len(row):
            continue
        v, a, t0_ms = float(row.iloc[0]["v"]), float(row.iloc[0]["a"]), float(row.iloc[0]["t0_ms"])
        if not np.all(np.isfinite([v, a, t0_ms])):
            continue
        rt = _rt_ms(df, effector, c)
        med = float(np.median(rt)) if rt.size else t0_ms + 120
        fig = _ddm_schematic_one(v, a, t0_ms, med, effector, c)
        out.append((f"DDM schematic — {int(SPEEDS[c])} deg/s", fig))
    return out


# --------------------------------------------------------------------------- #
# Non-decision time by speed  (port of NDT_barchart.py :: panel)
# --------------------------------------------------------------------------- #
def _friedman_p(units_df: pd.DataFrame):
    """Friedman p across the three speeds on per-participant t0 (None if not testable)."""
    if units_df is None or not len(units_df):
        return None
    piv = units_df.pivot_table(index="participant", columns="condition", values="t0_ms").dropna(axis=0)
    if piv.shape[0] < 3 or piv.shape[1] < 3:
        return None
    try:
        p = friedmanchisquare(*[piv[c].values for c in sorted(piv.columns)])[1]
        return None if not np.isfinite(p) else float(p)
    except Exception:
        return None


def _p_label(p):
    if p is None:
        return "Friedman p = n.s."
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "(n.s.)"
    return f"Friedman p = {p:.3f} {star}"


def _ndt_panel(ax, units_df, effector, floor_ms, group=None):
    """Dots + group-mean marker with SD bar, on a zoomed axis (never a bar from zero)."""
    rng = np.random.default_rng(0)
    all_vals = []
    for c in range(len(SPEEDS)):
        vals = units_df[units_df["condition"] == c]["t0_ms"].values.astype(float)
        if not vals.size:
            continue
        all_vals.append(vals)
        m, sd = vals.mean(), (vals.std(ddof=1) if vals.size > 1 else 0.0)
        ax.scatter(c + rng.uniform(-0.16, 0.16, len(vals)), vals, s=24, color=FILL[c],
                   alpha=0.55, edgecolor="#555", linewidth=0.4, zorder=3)
        ax.errorbar(c, m, yerr=sd, fmt="o", ms=12, color=FILL[c], mec="#222", mew=1.4,
                    ecolor="#222", capsize=6, lw=2.0, zorder=5)
        ax.text(c + 0.23, m, f"{m:.0f} ms", ha="left", va="center", fontsize=10, fontweight="bold")

    flat = np.concatenate(all_vals) if all_vals else np.array([floor_ms])
    # Anchor to the published windows (HRT 118-205, SRT 55-150) so the panel reads the
    # same as the repository figure, and only widen if this dataset falls outside them.
    # A window that shrink-wraps the data would magnify sub-millisecond differences --
    # the same distortion that made the original truncated bars misleading.
    pref_lo, pref_hi = (floor_ms - 12.0, floor_ms + 75.0) if effector == "hand" \
        else (floor_ms - 15.0, floor_ms + 80.0)
    dlo, dhi = float(flat.min()), float(flat.max())
    pad = max((dhi - dlo) * 0.18, 3.0)
    ylo, yhi = min(pref_lo, dlo - pad), max(pref_hi, dhi + pad)
    ax.axhline(floor_ms, color="#777", ls=":", lw=1.3, zorder=1)
    ax.text(2.46, floor_ms + (yhi - ylo) * 0.015, f"Physiol. min ({floor_ms:.0f} ms)", ha="right",
            va="bottom", fontsize=8, style="italic", color="#999")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{int(SPEEDS[c])} deg/s" for c in range(len(SPEEDS))])
    ax.set_xlim(-0.5, 2.8)
    ax.set_ylabel("$t_0$ (ms)")
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    p = _friedman_p(units_df)
    ax.set_title(f"{EFF_LABEL[effector]} Non-Decision Time\n{_p_label(p)}", fontsize=11.5, fontweight="bold")


def ndt_dots(units_df: pd.DataFrame, group: pd.DataFrame, effector: str, floor_ms: float):
    """Single-effector non-decision-time panel (kept signature; upgraded styling)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    _ndt_panel(ax, units_df, effector, floor_ms, group)
    fig.tight_layout()
    return fig


def ndt_by_speed(res_all: dict):
    """Combined HRT/SRT non-decision-time figure (one panel per fitted effector)."""
    panels = []
    for eff in EFFECTORS:
        r = res_all.get(eff)
        if not r:
            continue
        src = r["units"] if isinstance(r.get("units"), pd.DataFrame) and len(r["units"]) \
            else r.get("preview", {}).get("cell")
        if src is None or not len(src):
            continue
        floor = r.get("preview", {}).get("floor_ms")
        if floor is None and isinstance(r.get("group"), pd.DataFrame) and "t0_floor_ms" in r["group"]:
            floor = float(r["group"]["t0_floor_ms"].iloc[0])
        if floor is None:
            floor = PHYSIO_FLOOR[eff] * 1000.0
        panels.append((eff, src, floor))

    if not panels:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off"); ax.text(0.5, 0.5, "No fitted cells to plot.", ha="center")
        return fig

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 6), squeeze=False)
    n = 0
    for ax, (eff, src, floor) in zip(axes[0], panels):
        _ndt_panel(ax, src, eff, floor)
        n = max(n, src["participant"].nunique() if "participant" in src else 0)
    fig.suptitle(f"Non-Decision Time ($t_0$) by Target Speed\nGroup mean +/- 1 SD"
                 + (f"  (n = {n} participants)" if n else ""),
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Why saccadic t0 floors  (port of why_saccadic_t0_floors.py)
# --------------------------------------------------------------------------- #
def why_floors(df: pd.DataFrame):
    """Shape-implied t0 (mean - 3*SD/skew) vs the physiological floor, per effector."""
    effs = [e for e in EFFECTORS if (df["effector"] == e).any()]
    if not effs:
        fig, ax = plt.subplots(figsize=(5, 3)); ax.axis("off")
        ax.text(0.5, 0.5, "No trials to plot.", ha="center"); return fig

    fig, axes = plt.subplots(1, len(effs), figsize=(6.75 * len(effs), 5.6), squeeze=False)
    for ax, eff in zip(axes[0], effs):
        color = FILL[0] if eff == "hand" else FILL[2]
        floor = PHYSIO_FLOOR[eff] * 1000.0
        r = _rt_ms(df, eff)
        r = r[np.isfinite(r)]
        if r.size < 10:
            ax.axis("off"); ax.set_title(EFF_LONG[eff]); continue
        m, sd, sk = r.mean(), r.std(), skew(r)
        implied = m - 3 * sd / sk if sk != 0 else np.nan

        xs = np.linspace(r.min(), np.percentile(r, 99), 400)
        kde = gaussian_kde(r)(xs)
        ax.fill_between(xs, kde, color=color, alpha=0.45, zorder=2)
        ax.plot(xs, kde, color=tuple(c * 0.55 for c in color), lw=1.8, zorder=3)

        ymax = kde.max() * 1.18
        ax.axvline(floor, color="#777", ls=":", lw=1.6, zorder=4)
        ax.text(floor, ymax * 0.97, f"physiological\nfloor {floor:.0f} ms", ha="center", va="top",
                fontsize=8.5, style="italic", color="#777")
        if np.isfinite(implied):
            ax.axvline(implied, color="#b00", ls="--", lw=2.0, zorder=5)
            ax.text(implied, ymax * 0.62, f"shape-implied\n$t_0$ = {implied:.0f} ms", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="#b00")
            if implied < floor:
                ax.annotate("", xy=(implied, ymax * 0.30), xytext=(floor, ymax * 0.30),
                            arrowprops=dict(arrowstyle="->", color="#b00", lw=1.6))
                ax.text((implied + floor) / 2, ymax * 0.34, "wants to floor", ha="center",
                        fontsize=8.5, color="#b00", fontweight="bold")

        ax.set_title(EFF_LONG[eff], fontsize=12, fontweight="bold")
        ax.set_xlabel("reaction time (ms)"); ax.set_ylabel("density")
        ax.set_ylim(0, ymax); ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        cv = sd / m if m else np.nan
        txt = f"skewness = {sk:.2f}\nskew/CV ratio = {sk / cv:.1f}\n(a pure Wald = 3.0)"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))

    fig.suptitle("Why saccadic non-decision time floors: it is the distribution SHAPE, not the data\n"
                 "implied $t_0$ = mean RT \u2212 3\u00b7SD / skewness   (forced by the Wald's skew\u2013spread geometry)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# LATER reciprobit  (port of LATER_analysis.py)
# --------------------------------------------------------------------------- #
def _recipro(rt_ms):
    rate = np.sort(1000.0 / rt_ms)
    n = len(rate)
    p = (np.arange(1, n + 1) - 0.5) / n
    return rate, norm.ppf(p)


def _fit_line(rate, z, lo=0.10, hi=0.90):
    n = len(rate)
    a, b = int(lo * n), int(hi * n)
    if b - a < 2:
        return 0.0, 0.0, 0.0
    sl, ic, r, _, _ = linregress(rate[a:b], z[a:b])
    return float(sl), float(ic), float(r ** 2)


_XT_LAT = [500, 300, 200, 150, 120, 100]
_XT_POS = [1000.0 / t for t in _XT_LAT]


def _reciprobit_panel(ax, df_eye, pid, title, two_line=False):
    sub = df_eye[df_eye["participant"] == pid]
    for c in range(len(SPEEDS)):
        rt = _rt_ms(sub, "eye", c)
        rt = rt[(rt >= 80) & (rt <= 600)]
        if rt.size < 20:
            continue
        rate, z = _recipro(rt)
        ax.scatter(rate, z, s=14, color=FILL[c], alpha=0.55, edgecolor="none", zorder=3, label=_spd(c))
        sl, ic, _ = _fit_line(rate, z)
        xs = np.linspace(rate.min(), np.percentile(rate, 90), 50)
        ax.plot(xs, ic + sl * xs, color=_line(c), lw=1.8, zorder=4)
    if two_line:
        rt = _rt_ms(sub, "eye"); rt = rt[(rt >= 80) & (rt <= 600)]
        if rt.size:
            rate, z = _recipro(rt)
            n = len(rate); fast, zf = rate[int(0.80 * n):], z[int(0.80 * n):]
            if len(fast) > 5:
                sl, ic, _, _, _ = linregress(fast, zf)
                xs = np.linspace(np.percentile(rate, 75), rate.max(), 30)
                ax.plot(xs, ic + sl * xs, color="#b00", lw=1.8, ls="--", zorder=5)
                ax.text(rate.max(), ic + sl * rate.max(), " express\n early line", color="#b00",
                        fontsize=8, fontweight="bold", va="center")
    ax.set_xticks(_XT_POS); ax.set_xticklabels(_XT_LAT)
    ax.set_xlabel("saccadic latency (ms, reciprocal scale)")
    ax.set_ylabel("cumulative probability (probit)")
    yt = [norm.ppf(p) for p in [.05, .25, .5, .75, .95]]
    ax.set_yticks(yt); ax.set_yticklabels(["5%", "25%", "50%", "75%", "95%"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.text(0.03, 0.96, "straight line =\nLATER (reciprocal-normal)", transform=ax.transAxes,
            va="top", fontsize=8, style="italic", color="#666")


def reciprobit(later_result: dict, df_eye: pd.DataFrame):
    """Reciprobit for a regular and an express-dominant participant + latency by speed."""
    pp = later_result.get("per_participant")
    per_cell = later_result.get("per_cell")
    if pp is None or not len(pp) or per_cell is None or not len(per_cell):
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        ax.axis("off"); ax.text(0.5, 0.5, "Not enough saccadic data for LATER.", ha="center")
        return fig

    reg_pool = pp[pp["express_frac"] < 0.05].sort_values("reciprobit_r2", ascending=False)
    regular = (reg_pool.iloc[0] if len(reg_pool)
               else pp.sort_values("express_frac").iloc[0])["participant"]
    express = pp.sort_values("express_frac", ascending=False).iloc[0]["participant"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    _reciprobit_panel(ax[0], df_eye, regular, f"Regular saccades ({regular})\nlatencies fall on a straight line")
    if express != regular:
        _reciprobit_panel(ax[1], df_eye, express, f"Express-dominant ({express})\ntwo populations: express + regular", two_line=True)
    else:
        _reciprobit_panel(ax[1], df_eye, express, f"Second participant ({express})", two_line=True)

    rng = np.random.default_rng(0)
    for c in range(len(SPEEDS)):
        vals = per_cell[per_cell["condition"] == c]["median_lat_ms"].values.astype(float)
        if not vals.size:
            continue
        m, sd = vals.mean(), (vals.std(ddof=1) if vals.size > 1 else 0.0)
        ax[2].scatter(c + rng.uniform(-0.15, 0.15, len(vals)), vals, s=24, color=FILL[c],
                      alpha=0.55, edgecolor="#555", linewidth=0.4, zorder=3)
        ax[2].errorbar(c, m, yerr=sd, fmt="o", ms=12, color=FILL[c], mec="#222", mew=1.4,
                       ecolor="#222", capsize=6, lw=2.0, zorder=5)
        ax[2].text(c + 0.22, m, f"{m:.0f} ms", ha="left", va="center", fontsize=10, fontweight="bold")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([_spd(c) for c in range(len(SPEEDS))])
    ax[2].set_xlim(-0.5, 2.8); ax[2].set_ylabel("LATER median latency (ms)")
    ax[2].set_title("Saccadic latency by speed (LATER)\nno non-decision parameter \u2014 nothing to floor",
                    fontsize=11, fontweight="bold")
    ax[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax[2].grid(True, axis="y", ls="--", alpha=0.3); ax[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"LATER model for saccades \u2014 the saccade-native alternative to the shifted-Wald "
                 f"(median $R^2$ = {later_result.get('median_r2', float('nan')):.2f})",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Vincentile suite  (port of vincentile_figures.py)
# --------------------------------------------------------------------------- #
_NBINS = 20


def _vincentiles(rts, nbins=_NBINS):
    rts = np.sort(rts)
    n = len(rts)
    if n < nbins:
        return None
    idx = np.linspace(0, n, nbins + 1).astype(int)
    return np.array([rts[idx[i]:idx[i + 1]].mean() for i in range(nbins)])


def _vincentile_shift(df):
    """Per speed: (n_participants x nbins) array of hand-minus-eye vincentile differences.

    The tidy table carries no trial id, so hand and eye cannot be paired within a
    trial; instead each effector is vincentized per participant and the vincentiles
    are differenced (the standard vincentile shift function). Same quantity by bin.
    """
    out = {}
    for c in range(len(SPEEDS)):
        rows = []
        sub = df[df["condition"] == c]
        for pid, g in sub.groupby("participant"):
            hv = _vincentiles(g[g["effector"] == "hand"]["rt"].values * 1000.0)
            ev = _vincentiles(g[g["effector"] == "eye"]["rt"].values * 1000.0)
            if hv is not None and ev is not None:
                rows.append(hv - ev)
        out[c] = np.array(rows)
    return out


def vincentile_suite(df: pd.DataFrame):
    """The four repository RT-distribution figures (list of (label, Figure))."""
    has_hand = (df["effector"] == "hand").any()
    has_eye = (df["effector"] == "eye").any()
    nP = df["participant"].nunique()
    figs = []

    def pooled(eff, c):
        return _rt_ms(df, eff, c)

    # ---- fig1: KDE overlay (SRT + HRT) ----
    if has_hand and has_eye:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        xs = np.linspace(100, 600, 400)
        for c in range(len(SPEEDS)):
            sr, hr = pooled("eye", c), pooled("hand", c)
            if sr.size > 5:
                ks = gaussian_kde(sr)
                ax[c].fill_between(xs, ks(xs), color=SRT_C, alpha=0.25); ax[c].plot(xs, ks(xs), color=SRT_C, lw=1.8)
                ax[c].axvline(np.median(sr), color=SRT_C, ls="--", lw=1)
                ax[c].text(np.median(sr), ax[c].get_ylim()[1] * 0.92, f"median {np.median(sr):.0f} ms",
                           color=SRT_C, fontsize=8, ha="right", rotation=90, va="top")
            if hr.size > 5:
                kh = gaussian_kde(hr)
                ax[c].fill_between(xs, kh(xs), color=HRT_C, alpha=0.20); ax[c].plot(xs, kh(xs), color=HRT_C, lw=1.8)
                ax[c].axvline(np.median(hr), color=HRT_C, ls="--", lw=1)
                ax[c].text(np.median(hr), ax[c].get_ylim()[1] * 0.92, f"median {np.median(hr):.0f} ms",
                           color=HRT_C, fontsize=8, ha="right", rotation=90, va="top")
            ax[c].set_title(_spd(c), fontsize=11, fontweight="bold"); ax[c].set_xlabel("RT (ms)")
            ax[c].spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("Density")
        from matplotlib.lines import Line2D
        ax[2].legend(handles=[Line2D([0], [0], color=SRT_C, lw=2, label="SRT (gaze)"),
                              Line2D([0], [0], color=HRT_C, lw=2, label="HRT (hand)")], fontsize=9, loc="upper right")
        fig.suptitle(f"RT Distributions (SRT & HRT Overlay)   n={nP} participants", fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        figs.append(("RT distributions — KDE overlay", fig))

    # ---- fig2: histograms (SRT top, HRT bottom) ----
    if has_hand and has_eye:
        fig, ax = plt.subplots(2, 3, figsize=(13, 7))
        xs_s = np.linspace(100, 600, 300); xs_h = np.linspace(200, 800, 300)
        for c in range(len(SPEEDS)):
            sr, hr = pooled("eye", c), pooled("hand", c)
            if sr.size > 5:
                ax[0, c].hist(sr, bins=40, density=True, color=SRT_C, alpha=0.35, edgecolor="white", linewidth=0.3)
                ax[0, c].plot(xs_s, gaussian_kde(sr)(xs_s), color=SRT_C, lw=1.8)
                ax[0, c].axvline(np.median(sr), color=SRT_C, ls="--", lw=1)
                ax[0, c].text(np.median(sr) + 6, ax[0, c].get_ylim()[1] * 0.82, f"median\n{np.median(sr):.0f} ms",
                              color=SRT_C, fontsize=8)
            ax[0, c].set_title(_spd(c), fontsize=11, fontweight="bold")
            if hr.size > 5:
                ax[1, c].hist(hr, bins=40, density=True, color=HRT_C, alpha=0.35, edgecolor="white", linewidth=0.3)
                ax[1, c].plot(xs_h, gaussian_kde(hr)(xs_h), color=HRT_C, lw=1.8)
                ax[1, c].axvline(np.median(hr), color=HRT_C, ls="--", lw=1)
                ax[1, c].text(np.median(hr) + 8, ax[1, c].get_ylim()[1] * 0.82, f"median\n{np.median(hr):.0f} ms",
                              color=HRT_C, fontsize=8)
            ax[1, c].set_xlabel("RT (ms)")
            for a in (ax[0, c], ax[1, c]):
                a.spines[["top", "right"]].set_visible(False)
        ax[0, 0].set_ylabel("Density\n(SRT — gaze RT)"); ax[1, 0].set_ylabel("Density\n(HRT — hand RT)")
        fig.suptitle(f"RT Histograms — SRT (top) & HRT (bottom)   n={nP} participants", fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        figs.append(("RT histograms — SRT & HRT", fig))

    # ---- fig3 + fig4: HRT - SRT vincentile shift ----
    if has_hand and has_eye:
        diffs = _vincentile_shift(df)
        x = np.arange(1, _NBINS + 1)
        if any(len(diffs[c]) for c in diffs):
            fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
            for c in range(len(SPEEDS)):
                if not len(diffs[c]):
                    continue
                m = diffs[c].mean(0); sd = diffs[c].std(0, ddof=1) if diffs[c].shape[0] > 1 else np.zeros_like(m)
                ax[c].set_facecolor(SPEED_FILL_HEX[int(SPEEDS[c])])
                ax[c].axhline(0, color="#555", ls="--", lw=0.9)
                ax[c].errorbar(x, m, yerr=sd, fmt="-o", color="#555", ecolor="#888", ms=4, lw=1.8, capsize=2)
                ax[c].set_title(_spd(c), fontsize=11, fontweight="bold"); ax[c].set_xlabel("Vincentile bin")
                ax[c].set_xticks(x); ax[c].set_xticklabels(x, fontsize=7)
                ax[c].spines[["top", "right"]].set_visible(False)
            ax[0].set_ylabel("HRT − SRT (ms)")
            fig.suptitle("HRT − SRT Vincentile by Speed", fontsize=13, fontweight="bold")
            fig.text(0.5, -0.02, f"Group mean ± 1 SD  |  n={nP} participants", ha="center", fontsize=9, color="#666")
            fig.tight_layout()
            figs.append(("HRT − SRT vincentile by speed", fig))

            fig, ax = plt.subplots(figsize=(8.5, 6))
            for c in range(len(SPEEDS)):
                if not len(diffs[c]):
                    continue
                m = diffs[c].mean(0)
                ax.plot(x, m, "-o", color=SPEED_LINE_HEX[int(SPEEDS[c])], lw=2, ms=4, label=_spd(c))
                ax.fill_between(x, 0, m, color=SPEED_LINE_HEX[int(SPEEDS[c])], alpha=0.12)
            ax.axhline(0, color="#555", ls="--", lw=0.9)
            ax.set_xlabel("Vincentile bin"); ax.set_ylabel("HRT − SRT (ms)")
            ax.set_xticks([1, 5, 10, 15, 20]); ax.legend(fontsize=10, loc="upper left")
            ax.spines[["top", "right"]].set_visible(False); ax.grid(True, ls="--", alpha=0.3)
            fig.suptitle("HRT − SRT Vincentile (All Speeds)", fontsize=13, fontweight="bold")
            fig.text(0.5, -0.02, f"Group mean  |  n={nP} participants", ha="center", fontsize=9, color="#666")
            fig.tight_layout()
            figs.append(("HRT − SRT vincentile (all speeds)", fig))

    # ---- single-effector fallback: per-effector vincentiles so something still shows ----
    if not figs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for eff in [e for e in EFFECTORS if (df["effector"] == e).any()]:
            for c in range(len(SPEEDS)):
                v = _vincentiles(pooled(eff, c))
                if v is not None:
                    q = (np.arange(1, _NBINS + 1) - 0.5) / _NBINS
                    ax.plot(q, v, "-o", ms=3, color=_line(c) if eff == "hand" else FILL[c],
                            label=f"{eff} · {_spd(c)}")
        ax.set_xlabel("within-participant quantile"); ax.set_ylabel("RT (ms)")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.set_title("Vincentiles (model-free)", fontsize=11)
        fig.tight_layout()
        figs.append(("Vincentiles (model-free)", fig))

    return figs


def vincentile_plot(vinc_df: pd.DataFrame, effector: str = ""):
    """Model-free vincentiles: quantile-averaged RT vs quantile (kept for the CLI)."""
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


# --------------------------------------------------------------------------- #
# Advanced-analysis diagnostics (kept)
# --------------------------------------------------------------------------- #
def fixed_t0_plot(sens_df: pd.DataFrame, effector: str = ""):
    """
    Fixed-t0 sensitivity, reproducing SRT_fixed_t0_analysis.py.

    Left: mean drift by speed, one line per assumed t0 -- the pattern is the same
    whichever value is fixed. Right: mean KS per assumed t0 -- fit quality is
    effectively identical, which is why the data cannot pick the value and fixing
    it costs nothing.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    t0_vals = sorted(sens_df["t0_fixed_ms"].unique()) if len(sens_df) else []
    styles = {0: ("--", "o"), 1: ("-", "s"), 2: (":", "^")}
    for i, t0 in enumerate(t0_vals):
        g = sens_df[sens_df["t0_fixed_ms"] == t0].sort_values("condition")
        ls, mk = styles.get(i, ("-", "o"))
        ax[0].plot(g["condition"], g["v"], ls, marker=mk, lw=2, ms=7,
                   label=f"$t_0$ = {int(t0)} ms")
    ax[0].set_xticks(range(len(SPEEDS)))
    ax[0].set_xticklabels([_spd(c) for c in range(len(SPEEDS))])
    ax[0].set_ylabel("mean drift rate $v$")
    ax[0].set_title("A.  Drift by speed is stable across the assumed $t_0$\n"
                    "(the conclusions do not depend on the fixed value)",
                    fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=9.5, title="fixed non-decision time")
    ax[0].grid(True, ls="--", alpha=0.3)

    ks_col = "mean_ks" if "mean_ks" in sens_df.columns else "median_ks"
    means = [sens_df[sens_df["t0_fixed_ms"] == t0][ks_col].mean() for t0 in t0_vals]
    ax[1].bar([f"{int(t0)} ms" for t0 in t0_vals], means,
              color=["#9ecae1", "#4292c6", "#08519c"][:len(t0_vals)],
              edgecolor="#333", width=0.6)
    for i, mn in enumerate(means):
        if np.isfinite(mn):
            ax[1].text(i, mn + 0.002, f"{mn:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax[1].axhline(0.10, color="#E84855", ls=":", lw=1.3)
    ax[1].text(len(t0_vals) - 0.6, 0.103, "acceptable < 0.10", ha="right",
               color="#E84855", fontsize=8.5)
    ax[1].set_ylabel("mean KS (fit quality)")
    ax[1].set_ylim(0, max([m for m in means if np.isfinite(m)] + [0.12]) * 1.15)
    ax[1].set_title("B.  Fit quality is identical across $t_0$\n"
                    "(data cannot distinguish the values — hence fixing it)",
                    fontsize=11, fontweight="bold")
    ax[1].grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("Saccadic RT with fixed non-decision time — the floor-piling artifact "
                 "is removed and conclusions are robust", fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


def identifiability_plot(sweep_df: pd.DataFrame, effector: str = ""):
    """
    Saccadic t0 identifiability, reproducing SRT_identifiability_check.py.

    Left: fitted t0 against the imposed floor, one line per cell, red where t0
    tracks the floor (slope > 0.7, so the floor is setting it) and green where it
    stays put. Right: the distribution of those slopes.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    t0_cols = sorted([c for c in sweep_df.columns if c.startswith("t0_at_")],
                     key=lambda c: int(c.split("_")[-1]))
    floors = [int(c.split("_")[-1]) for c in t0_cols]
    n_track = 0
    for _, r in sweep_df.iterrows():
        col = "#C0392B" if r["tracks_floor"] else "#27AE60"
        n_track += bool(r["tracks_floor"])
        ax[0].plot(floors, [r[c] for c in t0_cols], "-o", color=col, alpha=0.55, ms=3, lw=1)
    if floors:
        ax[0].plot([floors[0], floors[-1]], [floors[0], floors[-1]], "k--", lw=1.5,
                   label="$t_0$ = floor (unidentified)")
        ax[0].legend(fontsize=9)
    ax[0].set_xlabel("imposed non-decision floor (ms)")
    ax[0].set_ylabel("fitted $t_0$ (ms)")
    ax[0].set_title("SRT $t_0$ vs imposed floor\nred = tracks floor (unidentified); "
                    "green = stable (identified)", fontsize=11, fontweight="bold")
    ax[0].grid(True, ls="--", alpha=0.3)

    slopes = sweep_df["slope"].values if "slope" in sweep_df else np.array([])
    if slopes.size:
        ax[1].hist(slopes, bins=np.linspace(0, 1.05, 12), color="#7f8c8d", edgecolor="white")
        ax[1].axvline(0.7, color="#C0392B", ls=":", lw=1.5)
        ax[1].text(0.71, ax[1].get_ylim()[1] * 0.9, "tracks floor →", color="#C0392B", fontsize=9)
    ax[1].set_xlabel("slope of $t_0$ vs floor  (1 = perfectly tracks floor)")
    ax[1].set_ylabel("number of cells")
    ax[1].set_title(f"{n_track}/{len(sweep_df)} SRT single cells are floor-determined",
                    fontsize=11, fontweight="bold")
    ax[1].grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("Saccadic non-decision time is largely NOT identifiable — "
                 "it sits at the imposed floor", fontsize=12.5, fontweight="bold", y=1.0)
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


def group_ci_plot(group: pd.DataFrame, effector: str = "", param: str = "t0_ms"):
    """Group-level parameter per speed with 94% credible intervals (per-speed model)."""
    labels = {"t0_ms": "non-decision time (ms)", "v": "drift v", "a": "boundary a"}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for _, r in group.iterrows():
        c = int(np.argmin([abs(r["speed"] - s) for s in SPEEDS]))
        lo, hi = r.get(f"{param}_lo", r[param]), r.get(f"{param}_hi", r[param])
        ax.errorbar(c, r[param], yerr=[[r[param] - lo], [hi - r[param]]], fmt="o", ms=9,
                    color=_line(c), capsize=6, lw=2, zorder=3)
    if param == "t0_ms" and "t0_floor_ms" in group.columns:
        ax.axhline(group["t0_floor_ms"].iloc[0], color="0.4", ls="--", lw=1.0)
        ax.text(2.0, group["t0_floor_ms"].iloc[0] + 1, "floor", fontsize=8, color="0.4")
    ax.set_xticks(range(len(SPEEDS)))
    ax.set_xticklabels([_spd(c) for c in range(len(SPEEDS))])
    ax.set_ylabel(labels.get(param, param), fontsize=9)
    ax.set_title(f"{effector.capitalize()} {labels.get(param, param)} by speed "
                 f"(per-speed model, 94% CI)", fontsize=10)
    fig.tight_layout()
    return fig
