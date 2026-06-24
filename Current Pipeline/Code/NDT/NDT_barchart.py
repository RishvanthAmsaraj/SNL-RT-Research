"""
NDT_barchart.py  --  Non-decision time (t0) by target speed  (matches the original figure)
==========================================================================================
Two panels (HRT, SRT): group mean +/- 1 SD bars per target speed, with EVERY participant shown
as a jittered dot so the distribution -- and any floor-piling -- is directly visible. This is
the original layout, kept deliberately because the dots are how we check whether saccadic t0 is
piling at its lower bound again.

Reading guide:
  * HRT t0 is well-identified -> interpret normally.
  * SRT t0 is at / near the physiological floor and is NOT separately identifiable per cell
    (~20/33 cells track the floor; see SRT_identifiability_check.py). The dots clustering near
    the floor here are exactly that artifact; the trustworthy saccadic t0 estimate (per
    participant, with individual differences) is Bayesian_srt_ndt.pdf. Read this SRT panel as a
    diagnostic, not a measurement.

SRT per-cell t0 uses the single-cell t0, or the regular-component t0r for express/regular
mixture cells. Floors: HRT 130 ms (reach-prep, Haith 2016), SRT 70 ms (afferent+efferent saccadic conduction).
Reads DDM_hrt_fits.csv, DDM_srt_fits.csv. Output: NDT_barchart.pdf / .png.
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import matplotlib.font_manager as fm
import matplotlib.ticker
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run DDM_fit.py first.")
    return p
SPEEDS = [0, 75, 150]
SC = {0: (0.45, 0.68, 0.40), 75: (0.85, 0.55, 0.55), 150: (0.50, 0.62, 0.82)}

def srt_t0_table(ds):
    """one saccadic t0 per participant per speed: t0 (single) or t0r (mixture regular comp)."""
    rows = []
    for _, r in ds.iterrows():
        t0 = r["t0r"] if (r.get("model") == "mixture" and "t0r" in ds.columns and pd.notna(r.get("t0r"))) else r.get("t0")
        if pd.notna(t0): rows.append(dict(pid=r["pid"], spd=int(r["spd"]), t0=float(t0)))
    return pd.DataFrame(rows)

def friedman_p(tbl):
    piv = tbl.pivot_table(index="pid", columns="spd", values="t0").dropna(axis=0)
    if piv.shape[0] < 3 or piv.shape[1] < 3: return None, piv.shape[0]
    try:
        p = friedmanchisquare(*[piv[s].values for s in SPEEDS])[1]
        return (None if not np.isfinite(p) else p), piv.shape[0]
    except Exception:
        return None, piv.shape[0]

def p_label(p):
    if p is None: return "Friedman p = n.s."
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "(n.s.)"
    return f"Friedman p = {p:.3f} {star}"

def panel(ax, tbl, title, floor, ylo, yhi, n):
    rng = np.random.default_rng(0)
    for i, s in enumerate(SPEEDS):
        vals = tbl[tbl.spd == s]["t0"].values
        m, sd = vals.mean(), vals.std(ddof=1)
        # participant dots (jittered) so the spread -- and any floor-piling -- is visible
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=24, color=SC[s],
                   alpha=0.55, edgecolor="#555", linewidth=0.4, zorder=3)
        # group mean as a marker + SD bar (NOT a bar from zero: under a zoomed axis a
        # truncated bar would visually exaggerate the differences; a point+CI does not)
        ax.errorbar(i, m, yerr=sd, fmt="o", ms=12, color=SC[s], mec="#222", mew=1.4,
                    ecolor="#222", capsize=6, lw=2.0, zorder=5)
        ax.text(i + 0.23, m, f"{m:.0f} ms", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.axhline(floor, color="#777", ls=":", lw=1.3, zorder=1)
    ax.text(2.46, floor + (yhi - ylo) * 0.015, f"Physiol. min ({floor:.0f} ms)", ha="right", va="bottom",
            fontsize=8, style="italic", color="#999")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{s} deg/s" for s in SPEEDS]); ax.set_xlim(-0.5, 2.8)
    ax.set_ylabel("$t_0$ (ms)"); ax.set_ylim(ylo, yhi); ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, axis="y", ls="--", alpha=0.3)

dh = pd.read_csv(_need("DDM_hrt_fits.csv")); ds = pd.read_csv(_need("DDM_srt_fits.csv"))
hrt = dh[["pid", "spd", "t0"]].copy(); srt = srt_t0_table(ds)
ph, nph = friedman_p(hrt); psr, npsr = friedman_p(srt)
n = dh["pid"].nunique()

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
panel(ax[0], hrt, f"HRT Non-Decision Time\n{p_label(ph)}", 130, 118, 205, nph)
panel(ax[1], srt, f"SRT Non-Decision Time\n{p_label(psr)}", 70, 55, 150, npsr)
fig.suptitle(f"Non-Decision Time ($t_0$) by Target Speed\nGroup mean +/- 1 SD  (n = {n} participants)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "NDT_barchart.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "NDT_barchart.png"), dpi=140, bbox_inches="tight", facecolor="white")
print(f"saved NDT_barchart.pdf/.png (n={n}); HRT Friedman p={ph}, SRT Friedman p={psr}")
print("SRT t0 by speed:", {s: round(srt[srt.spd==s].t0.mean()) for s in SPEEDS})
