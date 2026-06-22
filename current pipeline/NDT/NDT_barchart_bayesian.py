"""
NDT_barchart_bayesian.py  --  Non-decision time (t0), HIERARCHICAL BAYESIAN estimates
======================================================================================
Replaces the DDM/MLE NDT bar chart with the Bayesian (Method B) estimates, which are the
ones to report: the hierarchical model floors no cells and returns full credible intervals.

Because the two effectors differ in what is identifiable, the panels differ HONESTLY:

  LEFT  (HRT)  -- non-decision time IS identified per participant x speed, so it is shown
                 by target speed (group mean +/- 1 SD, every participant as a dot).
                 The Bayesian fit floors 0/48 cells (the DDM floored 3 at 150 deg/s), and
                 the speed effect is strong: Friedman p = 0.002 on the Bayesian t0.

  RIGHT (SRT)  -- saccadic non-decision time is NOT identifiable per cell (fast saccades
                 cannot separate non-decision from decision time). The Bayesian model
                 therefore estimates ONE t0 per participant, SHARED across target speed,
                 with wide CIs where the data are uninformative. There is no per-speed SRT
                 t0 to plot; the honest result is the per-participant forest below.

Reads Bayesian_hrt_ndt.csv, Bayesian_srt_ndt.csv (next to this script).
Output: NDT_barchart_bayesian.pdf / .png
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare
import matplotlib.font_manager as fm

_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run the Bayesian fits first.")
    return p

SPEEDS = [0, 75, 150]
SC = {0: (0.45, 0.68, 0.40), 75: (0.85, 0.55, 0.55), 150: (0.50, 0.62, 0.82)}  # match original suite
HRT_FLOOR = 100
TIGHT_C, LOOSE_C = "#2c7fb8", "#d95f0e"   # well-constrained vs regularized (match Bayesian_srt_ndt.py)

bh = pd.read_csv(_need("Bayesian_hrt_ndt.csv"))
bs = pd.read_csv(_need("Bayesian_srt_ndt.csv"))

# ---- HRT Friedman on the Bayesian t0 ----
piv = bh.pivot_table(index="pid", columns="spd", values="t0_ms")
pH = friedmanchisquare(piv[0], piv[75], piv[150])[1]
def p_label(p):
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "(n.s.)"
    return f"Friedman p = {p:.3f} {star}"

fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.15]})

# ================= Panel A: HRT non-decision time by speed (Bayesian) =================
rng = np.random.default_rng(0)
for i, s in enumerate(SPEEDS):
    vals = bh[bh.spd == s]["t0_ms"].values
    m, sd = vals.mean(), vals.std(ddof=1)
    ax[0].bar(i, m, 0.62, color=SC[s], edgecolor="#444", linewidth=0.9, zorder=2)
    ax[0].errorbar(i, m, yerr=sd, fmt="none", ecolor="#222", capsize=5, lw=1.6, zorder=4)
    ax[0].scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=20, color="#555",
                  alpha=0.55, edgecolor="none", zorder=5)
    ax[0].text(i, m + sd + 5, f"{m:.0f} \u00b1 {sd:.0f} ms", ha="center", fontsize=9.5, fontweight="bold")
ax[0].axhline(HRT_FLOOR, color="#777", ls=":", lw=1.3, zorder=1)
ax[0].text(2.46, HRT_FLOOR + 3, "Physiol. min\n(100 ms)", ha="right", va="bottom",
           fontsize=8, style="italic", color="#999")
ax[0].text(0.02, 0.025, "0 / 48 cells floored", transform=ax[0].transAxes, fontsize=8.5,
           style="italic", color="#2c7fb8")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f"{s} deg/s" for s in SPEEDS])
ax[0].set_ylabel("$t_0$ (ms)"); ax[0].set_ylim(0, 215)
ax[0].set_title(f"HRT Non-Decision Time  (n = {piv.shape[0]})\n{p_label(pH)}", fontsize=11.5, fontweight="bold")
ax[0].spines[["top", "right"]].set_visible(False); ax[0].grid(True, axis="y", ls="--", alpha=0.3)

# ================= Panel B: SRT non-decision time per participant (Bayesian) =================
nn = bs.sort_values("t0_ms").reset_index(drop=True)
pop_mean = nn.t0_ms.mean()
for i, r in nn.iterrows():
    tight = r.ci_width_ms < 35
    c = TIGHT_C if tight else LOOSE_C
    ax[1].plot([r.t0_lo95, r.t0_hi95], [i, i], color=c, lw=2.2, alpha=0.85, zorder=3)
    ax[1].plot(r.t0_ms, i, "o", color=c, ms=6, zorder=4)
ax[1].axvline(pop_mean, color="#444", ls="--", lw=1.3, zorder=2)
ax[1].set_yticks(range(len(nn))); ax[1].set_yticklabels(nn.pid, fontsize=8.5)
ax[1].set_xlabel("saccadic $t_0$ (ms, posterior mean \u00b1 95% CI)")
ax[1].set_xlim(20, 140)
ax[1].set_title(f"SRT Non-Decision Time  (n = {len(nn)}; per participant)\nestimated, speed-invariant \u2014 not floored",
                fontsize=11.5, fontweight="bold")
ax[1].legend(handles=[
    Line2D([0], [0], color=TIGHT_C, lw=2.2, marker="o", ms=5, label="well-constrained (95% CI < 35 ms)"),
    Line2D([0], [0], color=LOOSE_C, lw=2.2, marker="o", ms=5, label="regularized (95% CI \u2265 35 ms)"),
    Line2D([0], [0], color="#444", ls="--", lw=1.3, label=f"across-participant mean ({pop_mean:.0f} ms)")],
    fontsize=8.3, loc="lower right", framealpha=0.95)
ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(True, axis="x", ls="--", alpha=0.3)

fig.suptitle("Non-Decision Time ($t_0$) \u2014 Hierarchical Bayesian estimates",
             fontsize=13.5, fontweight="bold", y=1.005)
fig.text(0.5, -0.015,
         "HRT: hand non-decision time decreases with target speed (no cells floored).   "
         "SRT: saccadic non-decision time is per-participant (shared across speed) with honest uncertainty.   "
         "2 participants (CMT003, CMT004) are express/regular-mixture at all speeds and appear in the mixture decomposition.",
         ha="center", fontsize=7.8, color="#666")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "NDT_barchart_bayesian.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "NDT_barchart_bayesian.png"), dpi=150, bbox_inches="tight", facecolor="white")
print("saved NDT_barchart_bayesian.pdf/.png")
print(f"HRT Bayesian t0 by speed: {{{', '.join(f'{s}:{bh[bh.spd==s].t0_ms.mean():.0f}' for s in SPEEDS)}}}  Friedman p={pH:.4f}")
print(f"SRT per-participant t0: {nn.t0_ms.min()}-{nn.t0_ms.max()} ms, mean {pop_mean:.0f} ms, n={len(nn)}")
