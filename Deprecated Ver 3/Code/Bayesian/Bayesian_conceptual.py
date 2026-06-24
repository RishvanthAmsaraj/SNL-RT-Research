"""
DDM_conceptual.py  --  conceptual single-boundary diffusion schematics (match original suite)
==============================================================================================
Produces six annotated DDM schematics -- one per measure x speed -- matching the original look:
a noisy evidence-accumulation path rising from the starting point (z) at the baseline to the
response threshold (a), the predicted RT distribution sitting ABOVE the threshold, and the
non-decision (t0) / decision-time decomposition along a time axis.

  ddm_hrt_0_degs.pdf  ddm_hrt_75_degs.pdf  ddm_hrt_150_degs.pdf
  ddm_srt_0_degs.pdf  ddm_srt_75_degs.pdf  ddm_srt_150_degs.pdf   (+ .png each)

Group-mean parameters are read from the DDM fits; for SRT mixture cells the regular-saccade
component (t0r, vr, ar) is used so the schematic reflects the ordinary, non-express process.
The RT distribution drawn is the shifted-Wald density at those group-mean parameters. NOTE:
the saccadic t0 shown is the per-cell fit value and is at the physiological floor / not
separately identifiable (see SRT_identifiability_check.py and Bayesian_SRT_ndt.py); these
schematics illustrate the *process*, not a t0 measurement.

Reads DDM_hrt_fits.csv, DDM_srt_fits.csv, pooled_data.csv.  Run: python DDM_conceptual.py
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run DDM_fit.py first.")
    return p
SPEEDS = [0, 75, 150]
COL = {0: "#4a7c59", 75: "#9e5a5a", 150: "#496aa3"}
XMAX = {"hrt": 600, "srt": 500}
DIST_LABEL = {"hrt": "HRT Distribution\n(Interception)", "srt": "SRT Distribution\n(Interception)"}

def wald_pdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return (a / np.sqrt(2 * np.pi * t**3)) * np.exp(-(a - v * t)**2 / (2 * t))

def group_params(df, measure):
    out = {}
    for s in SPEEDS:
        d = df[df.spd == s]
        if measure == "hrt":
            v, a, t0 = d.v, d.a, d.t0
        else:
            sing = d[d.model == "single"]; mix = d[d.model == "mixture"]
            v = pd.concat([sing.v, mix.vr]) if "vr" in d.columns else sing.v
            a = pd.concat([sing.a, mix.ar]) if "ar" in d.columns else sing.a
            t0 = pd.concat([sing.t0, mix.t0r]) if "t0r" in d.columns else sing.t0
        out[s] = dict(v=v.mean(), a=a.mean(), t0=t0.mean(), v_sd=v.std(), a_sd=a.std(), t0_sd=t0.std())
    return out

def median_rt(dfi, measure, s):
    z = dfi[dfi.Speed_deg_per_s == s]
    if measure == "hrt":
        x = z["HandRT_ms"].values.astype(float); x = x[(~np.isnan(x)) & (x >= 150) & (x <= 800)]
    else:
        x = z["GazeSRT_ms"].values.astype(float); x = x[(~np.isnan(x)) & (x >= 80) & (x <= 600)]
    return float(np.median(x))

def diffusion_path(t0_ms, t_hit_ms, y0, y1, color, ax, seed):
    rng = np.random.default_rng(seed)
    for k in range(4):
        n = 160
        t_end = t_hit_ms * (0.85 + 0.18 * rng.random())
        tt = np.linspace(t0_ms, t_end, n)
        ramp = np.linspace(y0, y1, n)
        noise = np.cumsum(rng.standard_normal(n)) * 0.030
        noise -= np.linspace(noise[0], noise[-1], n)
        path = np.clip(ramp + noise, y0 - 0.10, y1)
        if k == 0:
            ax.plot(tt, path, color=color, lw=1.7, alpha=0.95, zorder=5, solid_capstyle="round")
        else:
            ax.plot(tt, path, color=color, lw=0.8, alpha=0.28, zorder=3)

def draw(measure, s, p, med, color):
    v, a, t0 = p["v"], p["a"], p["t0"]; xmax = XMAX[measure]
    fig, ax = plt.subplots(figsize=(11, 6))
    y_base, y_thr = 0.34, 0.72
    ax.add_patch(plt.Rectangle((0, 0.17), t0, 0.66, facecolor="#ededed", edgecolor="none", zorder=0))
    ax.plot([0, xmax], [y_thr, y_thr], color=color, lw=2.6, zorder=4)
    ax.text(xmax, y_thr + 0.012, f"Response threshold ($a$ = {a:.2f})", ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=color)
    ax.plot([t0, xmax], [y_base, y_base], color="#9a9a9a", ls=(0, (5, 4)), lw=1.0, zorder=1)
    ax.text(xmax, y_base - 0.03, "Baseline", ha="right", va="top", fontsize=9, style="italic", color="#9a9a9a")
    tt = np.linspace(t0 + 1, xmax, 400)
    dens = wald_pdf((tt - t0) / 1000.0, v, a)
    if dens.max() > 0: dens = dens / dens.max()
    band_lo, band_h = y_thr + 0.015, 0.30
    ax.fill_between(tt, band_lo, band_lo + dens * band_h, color=color, alpha=0.22, zorder=2)
    ax.plot(tt, band_lo + dens * band_h, color=color, lw=1.6, zorder=3)
    ax.text(xmax * 0.66, band_lo + band_h * 0.55, DIST_LABEL[measure], ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=color)
    diffusion_path(t0, med, y_base, y_thr, color, ax, seed=42 + s + (0 if measure == "hrt" else 7))
    ax.plot(t0, y_base, "o", color="#1a1a1a", ms=8, zorder=6)
    ax.text(t0 - xmax * 0.012, y_base + 0.085, "Starting\npoint ($z$)", ha="right", va="center", fontsize=9.5)
    ax.annotate("", xy=(t0 + (med - t0) * 0.42, 0.555), xytext=(t0 + 4, y_base + 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#111"), zorder=7)
    ax.text(t0 + (med - t0) * 0.46, 0.55, f"Drift rate ($v$ = {v:.2f})", fontsize=10, fontweight="bold", va="center")
    xa = xmax * 0.052
    ax.annotate("", xy=(xa, y_thr), xytext=(xa, y_base), arrowprops=dict(arrowstyle="<->", lw=1.6, color="#1a1a1a"))
    ax.text(xa - xmax * 0.012, (y_base + y_thr) / 2, "Threshold\nheight ($a$)", ha="right", va="center", fontsize=9.5)
    ax.annotate("", xy=(xmax * 1.02, 0.17), xytext=(0, 0.17), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333"))
    for tk in range(0, xmax + 1, 100):
        ax.plot([tk, tk], [0.17, 0.155], color="#333", lw=1)
        ax.text(tk, 0.125, f"{tk}", ha="center", va="top", fontsize=8.5, color="#333")
    ax.text(xmax / 2, 0.06, "Time (ms)", ha="center", va="top", fontsize=10.5)
    ax.plot([t0, t0], [0.17, 0.145], color="#333", lw=1.4)
    ax.text(t0, 0.125, f"{t0:.0f} ms\n($t_0$)", ha="center", va="top", fontsize=8.5, fontweight="bold")
    yb = 0.015
    ax.annotate("", xy=(t0, yb), xytext=(0, yb), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#555"))
    ax.text(t0 / 2, yb - 0.055, f"Non-decision time ($t_0$) = {t0:.0f} ms", ha="center", va="top", fontsize=9, style="italic", color="#555")
    ax.annotate("", xy=(xmax, yb), xytext=(t0, yb), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#999"))
    ax.text((t0 + xmax) / 2, yb - 0.055, "Decision time", ha="center", va="top", fontsize=9, style="italic", color="#999")
    fig.suptitle(f"Bayesian (Single Boundary, {measure.upper()}) -- {s} deg/s", fontsize=12.5, fontweight="bold", y=0.99)
    ax.set_title(f"Group mean:  $v$ = {v:.2f} +/- {p['v_sd']:.2f},   $a$ = {a:.2f} +/- {p['a_sd']:.2f},   "
                 f"$t_0$ = {t0:.0f} +/- {p['t0_sd']:.0f} ms", fontsize=11, pad=10)
    ax.set_xlim(-xmax * 0.02, xmax * 1.06); ax.set_ylim(-0.10, 1.05); ax.axis("off")
    fig.tight_layout()
    base = os.path.join(HERE, f"bayes_{measure}_{s}_degs")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

dh = pd.read_csv(_need("Bayesian_hrt_fits.csv")); ds = pd.read_csv(_need("Bayesian_srt_fits.csv"))
dfi = pd.read_csv(_need("pooled_data.csv")); dfi = dfi[dfi.BlockType == "I"]
ph = group_params(dh, "hrt"); ps = group_params(ds, "srt")
for s in SPEEDS:
    draw("hrt", s, ph[s], median_rt(dfi, "hrt", s), COL[s])
    draw("srt", s, ps[s], median_rt(dfi, "srt", s), COL[s])
print("saved 6 conceptual Bayesian schematics to", HERE)
print("HRT t0:", {s: round(ph[s]['t0']) for s in SPEEDS}, "| SRT t0:", {s: round(ps[s]['t0']) for s in SPEEDS})
