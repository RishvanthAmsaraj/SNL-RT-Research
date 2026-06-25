"""
NDT_barchart_bayesian.py  --  NDT bar charts (Bayesian fits)

Two panels: HRT (t0 is identified per participant x speed, group means with dots)
and SRT (per-participant forest plot; t0 not identifiable per cell). Bayesian
model floors 0/48 HRT cells.

Reads Bayesian_hrt_ndt.csv, Bayesian_srt_ndt.csv. Output: NDT_barchart_bayesian.pdf/.png

Run: python NDT_barchart_bayesian.py  (needs Bayesian fit outputs first)
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.ticker
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
HRT_FLOOR = 130
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
    ax[0].scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=24, color=SC[s],
                  alpha=0.55, edgecolor="#555", linewidth=0.4, zorder=3)
    ax[0].errorbar(i, m, yerr=sd, fmt="o", ms=12, color=SC[s], mec="#222", mew=1.4,
                   ecolor="#222", capsize=6, lw=2.0, zorder=5)
    ax[0].text(i + 0.23, m, f"{m:.0f} ms", ha="left", va="center", fontsize=10, fontweight="bold")
ax[0].axhline(HRT_FLOOR, color="#777", ls=":", lw=1.3, zorder=1)
ax[0].text(2.46, HRT_FLOOR + 1.3, f"Physiol. min ({HRT_FLOOR:.0f} ms)", ha="right", va="bottom",
           fontsize=8, style="italic", color="#999")
ax[0].text(0.02, 0.03, "0 / 48 cells floored", transform=ax[0].transAxes, fontsize=8.5,
           style="italic", color="#2c7fb8")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f"{s} deg/s" for s in SPEEDS]); ax[0].set_xlim(-0.5, 2.8)
ax[0].set_ylabel("$t_0$ (ms)"); ax[0].set_ylim(118, 205)
ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
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
ax[1].axvline(70, color="#777", ls=":", lw=1.3, zorder=1)
ax[1].text(70.5, 0.5, "70 ms physiological floor", rotation=90, fontsize=8, style="italic", color="#999", va="bottom")
ax[1].set_xlim(40, 150)
ax[1].set_title(f"SRT Non-Decision Time  (n = {len(nn)})\nnot identifiable \u2014 pinned at the 70 ms floor",
                fontsize=11.5, fontweight="bold")
ax[1].text(0.97, 0.04, "data favour <70 ms; pinned at floor\n(not estimable above it; Bompas 2024)",
           transform=ax[1].transAxes, ha="right", fontsize=7.6, style="italic", color="#b00")
ax[1].legend(handles=[
    Line2D([0], [0], color=TIGHT_C, lw=2.2, marker="o", ms=5, label="well-constrained (95% CI < 35 ms)"),
    Line2D([0], [0], color=LOOSE_C, lw=2.2, marker="o", ms=5, label="regularized (95% CI \u2265 35 ms)"),
    Line2D([0], [0], color="#444", ls="--", lw=1.3, label=f"across-participant mean ({pop_mean:.0f} ms)")],
    fontsize=8.3, loc="lower right", framealpha=0.95)
ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(True, axis="x", ls="--", alpha=0.3)

fig.suptitle("Non-Decision Time ($t_0$) \u2014 Hierarchical Bayesian estimates",
             fontsize=13.5, fontweight="bold", y=1.005)
fig.text(0.5, -0.015,
         "HRT: hand non-decision time is identifiable and decreases with target speed (170\u2192158\u2192148 ms; 0/48 cells floored).   "
         "SRT: saccadic non-decision time is NOT identifiable \u2014 with the 70 ms physiological floor enforced, every participant pins at the floor "
         "(the data favour even lower values), so it is reported as fixed at 70 ms rather than estimated per participant.",
         ha="center", fontsize=7.8, color="#666")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "NDT_barchart_bayesian.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "NDT_barchart_bayesian.png"), dpi=150, bbox_inches="tight", facecolor="white")
print("saved NDT_barchart_bayesian.pdf/.png")
print(f"HRT Bayesian t0 by speed: {{{', '.join(f'{s}:{bh[bh.spd==s].t0_ms.mean():.0f}' for s in SPEEDS)}}}  Friedman p={pH:.4f}")
print(f"SRT per-participant t0: {nn.t0_ms.min()}-{nn.t0_ms.max()} ms, mean {pop_mean:.0f} ms, n={len(nn)}")
