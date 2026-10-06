"""P1_vs_P2_NDT_bayesian_overlay.py -- Bayesian (Method B) non-decision time, both paradigms on ONE axis per effector.
Hand: Bayesian_hrt_fits.csv. Saccade: Bayesian_srt_fits.csv single-component cells (one t0 per participant x speed).
Small dots = participants, big markers = group mean, bars = +/-1 SD. Shared speeds (75, 150) are offset so both are visible."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import friedmanchisquare
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DIRS = {"P1": os.path.join(REPO, "Current Pipeline", "Code", "Bayesian"), "P2": os.path.join(REPO, "Paradigm 2 Pipeline", "Code", "Bayesian")}
FIG = os.path.join(REPO, "Paradigm 2 Pipeline", "Figures", "Comparison")
SP = {"P1": [0, 75, 150], "P2": [75, 100, 125, 150]}; COL = {"P1": "#1b7f79", "P2": "#d95f02"}; MK = {"P1": "s", "P2": "o"}
NAME = {"P1": "Paradigm 1 (CMT)", "P2": "Paradigm 2 (CIR)"}; OFF = {"P1": -5, "P2": 5}
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4))
for j, (eff, f, floor) in enumerate([("Hand", "Bayesian_hrt_fits.csv", 130), ("Saccade", "Bayesian_srt_fits.csv", 70)]):
    a = ax[j]; sub = []
    for P in ["P1", "P2"]:
        d = pd.read_csv(os.path.join(DIRS[P], f)); d = d[d.model == "single"] if "model" in d else d
        for s in SP[P]:
            v = d[d.spd == s].t0.values; x = s + OFF[P]
            a.scatter(x + np.random.default_rng(int(s) + (0 if P == "P1" else 9)).uniform(-2.2, 2.2, len(v)), v, s=14, color=COL[P], alpha=0.35, zorder=2)
            a.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt=MK[P], color=COL[P], mec="k", ms=11, capsize=5, lw=1.8, zorder=4)
            a.text(x, v.mean() + v.std(ddof=1) + 0.8, f"{v.mean():.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COL[P])   # label above the bar
        w = d.pivot_table(index="pid", columns="spd", values="t0")[SP[P]].dropna()
        nf = int(((d.t0 - floor) < 2).sum())
        sub.append(f"{P}: Friedman p = {friedmanchisquare(*[w[s] for s in SP[P]]).pvalue:.3f}, {nf}/{len(d)} cells at floor")
        a.plot([], [], MK[P], color=COL[P], mec="k", ms=9, label=NAME[P])
    a.axhline(floor, color="#C0392B", ls=":", lw=1.4, label=f"{floor} ms floor"); a.axvspan(62, 163, color="#999", alpha=0.06, lw=0)
    a.set_xticks([0, 75, 100, 125, 150]); a.set_xticklabels(["0\n(stationary)", "75", "100", "125", "150"]); a.set_xlim(-22, 168)
    a.set_xlabel("target speed (deg/s)"); a.set_ylabel(f"{eff.lower()} $t_0$ (ms)")
    extra = "\n(values held near the floor — differences between speeds are not a finding)" if eff == "Saccade" else ""
    a.set_title(f"{'AB'[j]}.  {eff} non-decision time (Bayesian)\n" + "  |  ".join(sub) + extra, fontsize=10.5, fontweight="bold")
    a.legend(fontsize=9, loc="upper right"); a.spines[["top", "right"]].set_visible(False); a.grid(True, axis="y", ls="--", alpha=0.3)
fig.suptitle("Bayesian non-decision time ($t_0$), Paradigm 1 vs Paradigm 2 on the same axes\n"
             "big markers = group mean ± 1 SD; small dots = participants; grey band = moving speeds", fontsize=12.5, fontweight="bold", y=1.03)
fig.tight_layout()
for ext, dpi in [("png", 130), ("pdf", 300)]: fig.savefig(os.path.join(FIG, f"P1_vs_P2_NDT_bayesian_overlay.{ext}"), dpi=dpi, bbox_inches="tight", facecolor="white")
print("saved P1_vs_P2_NDT_bayesian_overlay")
