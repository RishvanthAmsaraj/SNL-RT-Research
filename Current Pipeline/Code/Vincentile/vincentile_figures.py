"""
vincentile_figures.py  --  RT-distribution & Vincentile figures

Produces four distributional figures: KDE overlay, histograms, Vincentile
differences (HRT-SRT) per speed and combined. Standard Vincentizing (Ratcliff 1979):
20 equal-count bins per participant x speed, averaged across participants.

Reads pooled_data.csv only. Outputs: vincentile_results_fig{1-4}_*.pdf/.png

Run: python vincentile_figures.py
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "pooled_data.csv")
if not os.path.exists(DATA): sys.exit("ERROR: pooled_data.csv must sit next to this script.")
SPEEDS = [0, 75, 150]
SPEED_FILL = {0: "#cfe8cf", 75: "#f3d4d4", 150: "#d3ddef"}     # light per-speed backgrounds
SPEED_LINE = {0: "#4a7c59", 75: "#9e4a4a", 150: "#496aa3"}     # saturated per-speed lines
SRT_C, HRT_C = "#2c7fb8", "#d9772b"                            # SRT cool, HRT warm
NBINS = 20

def load():
    d = pd.read_csv(DATA); d = d[d["BlockType"] == "I"]
    out = {}; pairs = {}
    for s in SPEEDS:
        z = d[d["Speed_deg_per_s"] == s]
        out[s] = {}; pairs[s] = {}
        for pid, g in z.groupby("Participant"):
            h = g["HandRT_ms"].values.astype(float); r = g["GazeSRT_ms"].values.astype(float)
            # marginal distributions (for KDE / histograms)
            hh = h[(~np.isnan(h)) & (h >= 150) & (h <= 800)]
            rr = r[(~np.isnan(r)) & (r >= 80) & (r <= 600)]
            out[s][pid] = (rr, hh)
            # per-TRIAL paired differences (both valid in the same trial) for vincentiles
            ok = (~np.isnan(h)) & (~np.isnan(r)) & (h >= 150) & (h <= 800) & (r >= 80) & (r <= 600)
            pairs[s][pid] = (h[ok] - r[ok])
    return d, out, pairs

def vincentiles(rts, nbins=NBINS):
    rts = np.sort(rts); n = len(rts)
    if n < nbins: return None
    idx = np.linspace(0, n, nbins + 1).astype(int)
    return np.array([rts[idx[i]:idx[i+1]].mean() for i in range(nbins)])

def vincentile_diffs(pairs):
    """per speed: array (n_participants x nbins) of vincentized per-trial HRT-SRT differences."""
    out = {}
    for s in SPEEDS:
        rows = []
        for pid, dif in pairs[s].items():
            v = vincentiles(dif)
            if v is not None: rows.append(v)
        out[s] = np.array(rows)
    return out

def pooled(data, s, which):
    return np.concatenate([data[s][p][0 if which == "srt" else 1] for p in data[s]])

# ---------------------------------------------------------------- load
d_all, data, pairs = load()
nP = d_all["Participant"].nunique()
diffs = vincentile_diffs(pairs)

# ---------------------------------------------------------------- fig1: KDE overlay
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
xs = np.linspace(100, 600, 400)
for i, s in enumerate(SPEEDS):
    sr, hr = pooled(data, s, "srt"), pooled(data, s, "hrt")
    ks, kh = gaussian_kde(sr), gaussian_kde(hr)
    ax[i].fill_between(xs, ks(xs), color=SRT_C, alpha=0.25); ax[i].plot(xs, ks(xs), color=SRT_C, lw=1.8)
    ax[i].fill_between(xs, kh(xs), color=HRT_C, alpha=0.20); ax[i].plot(xs, kh(xs), color=HRT_C, lw=1.8)
    ms, mh = np.median(sr), np.median(hr)
    ax[i].axvline(ms, color=SRT_C, ls="--", lw=1); ax[i].axvline(mh, color=HRT_C, ls="--", lw=1)
    ax[i].text(ms, ax[i].get_ylim()[1]*0.92, f"median {ms:.0f} ms", color=SRT_C, fontsize=8, ha="right", rotation=90, va="top")
    ax[i].text(mh, ax[i].get_ylim()[1]*0.92, f"median {mh:.0f} ms", color=HRT_C, fontsize=8, ha="right", rotation=90, va="top")
    ax[i].set_title(f"{s} deg/s", fontsize=11, fontweight="bold"); ax[i].set_xlabel("RT (ms)")
    ax[i].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Density")
ax[2].legend(handles=[Line2D([0],[0],color=SRT_C,lw=2,label="SRT (gaze)"),
                      Line2D([0],[0],color=HRT_C,lw=2,label="HRT (hand)")], fontsize=9, loc="upper right")
fig.suptitle(f"RT Distributions (SRT & HRT Overlay)   n={nP} participants", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "vincentile_results_fig1_kde_overlay.pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig2: histograms (SRT top, HRT bottom)
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for i, s in enumerate(SPEEDS):
    sr, hr = pooled(data, s, "srt"), pooled(data, s, "hrt")
    xs_s = np.linspace(100, 600, 300); xs_h = np.linspace(200, 800, 300)
    ax[0, i].hist(sr, bins=40, density=True, color=SRT_C, alpha=0.35, edgecolor="white", linewidth=0.3)
    ax[0, i].plot(xs_s, gaussian_kde(sr)(xs_s), color=SRT_C, lw=1.8)
    ax[0, i].axvline(np.median(sr), color=SRT_C, ls="--", lw=1)
    ax[0, i].text(np.median(sr)+6, ax[0,i].get_ylim()[1]*0.82, f"median\n{np.median(sr):.0f} ms", color=SRT_C, fontsize=8)
    ax[0, i].set_title(f"{s} deg/s", fontsize=11, fontweight="bold")
    ax[1, i].hist(hr, bins=40, density=True, color=HRT_C, alpha=0.35, edgecolor="white", linewidth=0.3)
    ax[1, i].plot(xs_h, gaussian_kde(hr)(xs_h), color=HRT_C, lw=1.8)
    ax[1, i].axvline(np.median(hr), color=HRT_C, ls="--", lw=1)
    ax[1, i].text(np.median(hr)+8, ax[1,i].get_ylim()[1]*0.82, f"median\n{np.median(hr):.0f} ms", color=HRT_C, fontsize=8)
    ax[1, i].set_xlabel("RT (ms)")
    for a in (ax[0,i], ax[1,i]): a.spines[["top","right"]].set_visible(False)
ax[0, 0].set_ylabel("Density\n(SRT — gaze RT)"); ax[1, 0].set_ylabel("Density\n(HRT — hand RT)")
fig.suptitle(f"RT Histograms — SRT (top) & HRT (bottom)   n={nP} participants", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "vincentile_results_fig2_histograms.pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig3: HRT-SRT vincentile by speed (3 panels)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
x = np.arange(1, NBINS + 1)
for i, s in enumerate(SPEEDS):
    m = diffs[s].mean(0); sd = diffs[s].std(0, ddof=1)
    ax[i].set_facecolor(SPEED_FILL[s])
    ax[i].axhline(0, color="#555", ls="--", lw=0.9)
    ax[i].errorbar(x, m, yerr=sd, fmt="-o", color="#555", ecolor="#888", ms=4, lw=1.8, capsize=2)
    ax[i].set_title(f"{s} deg/s", fontsize=11, fontweight="bold"); ax[i].set_xlabel("Vincentile bin")
    ax[i].set_xticks(x); ax[i].set_xticklabels(x, fontsize=7)
    ax[i].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("HRT − SRT (ms)")
fig.suptitle("HRT − SRT Vincentile by Speed", fontsize=13, fontweight="bold")
fig.text(0.5, -0.02, f"Group mean ± 1 SD  |  n={nP} participants", ha="center", fontsize=9, color="#666")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "vincentile_results_fig3_vincentile_by_speed.pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- fig4: combined vincentile (all speeds)
fig, ax = plt.subplots(figsize=(8.5, 6))
for s in SPEEDS:
    m = diffs[s].mean(0)
    ax.plot(x, m, "-o", color=SPEED_LINE[s], lw=2, ms=4, label=f"{s} deg/s")
    ax.fill_between(x, 0, m, color=SPEED_LINE[s], alpha=0.12)
ax.axhline(0, color="#555", ls="--", lw=0.9)
ax.set_xlabel("Vincentile bin"); ax.set_ylabel("HRT − SRT (ms)")
ax.set_xticks([1, 5, 10, 15, 20]); ax.legend(fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False); ax.grid(True, ls="--", alpha=0.3)
fig.suptitle("HRT − SRT Vincentile (All Speeds)", fontsize=13, fontweight="bold")
fig.text(0.5, -0.02, f"Group mean  |  n={nP} participants", ha="center", fontsize=9, color="#666")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "vincentile_results_fig4_combined_vincentile.pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig)

print(f"saved 4 vincentile figures (n={nP}); HRT-SRT bin-20 diff by speed:",
      {s: round(diffs[s].mean(0)[-1]) for s in SPEEDS})
