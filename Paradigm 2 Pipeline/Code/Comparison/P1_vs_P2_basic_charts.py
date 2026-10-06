"""
P1_vs_P2_basic_charts.py -- the basic Paradigm 1 charts redrawn side by side with Paradigm 2, same style and same scale.
  P1_vs_P2_NDT_bayesian.png : hand and saccadic t0 by speed, Method B (hierarchical Bayesian) -- like NDT_barchart_bayesian
  P1_vs_P2_NDT_methodA.png  : the same chart for Method A (MLE)                                  -- like NDT_barchart
  P1_vs_P2_eye_hand_lag.png : HRT - SRT on the same trial, by vincentile and by speed               -- like the vincentile figures
Saccade panels use one t0 per participant x speed (Bayesian: single-component cells; Method A: regular component for
two-component cells, as NDT_barchart does), so hand and saccade can be drawn the same way.
Run: python P1_vs_P2_basic_charts.py   (inside the repo; reads both paradigms' committed tables)
"""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import friedmanchisquare
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P1C, P2C = os.path.join(REPO, "Current Pipeline", "Code"), os.path.join(REPO, "Paradigm 2 Pipeline", "Code")
FIG = os.path.join(REPO, "Paradigm 2 Pipeline", "Figures", "Comparison"); os.makedirs(FIG, exist_ok=True)
SP = {"P1": [0, 75, 150], "P2": [75, 100, 125, 150]}; NAME = {"P1": "Paradigm 1 (CMT)", "P2": "Paradigm 2 (CIR)"}
SC = {0: (0.45, 0.68, 0.40), 75: (0.85, 0.55, 0.55), 100: (0.88, 0.68, 0.36), 125: (0.66, 0.55, 0.80), 150: (0.50, 0.62, 0.82)}
LINE = {"P1": "#1b7f79", "P2": "#d95f02"}
tab = lambda P, sub, f: pd.read_csv(os.path.join(P1C if P == "P1" else P2C, sub, f))

def tables(method):
    out = {}
    for P in ["P1", "P2"]:
        if method == "B":
            h = tab(P, "Bayesian", "Bayesian_hrt_fits.csv")[["pid", "spd", "t0"]]
            s = tab(P, "Bayesian", "Bayesian_srt_fits.csv"); nmix = int((s.model == "mixture").sum()); s = s[s.model == "single"][["pid", "spd", "t0"]]
        else:
            h = tab(P, "DDM", "DDM_hrt_fits.csv")[["pid", "spd", "t0"]]
            s = tab(P, "DDM", "DDM_srt_fits.csv"); nmix = int((s.model == "mixture").sum())
            s = s.assign(t0=np.where(s.model == "mixture", s.t0r, s.t0))[["pid", "spd", "t0"]]
        out[P] = (h.dropna(), s.dropna(), nmix)
    return out

def fried(d, speeds):
    w = d.pivot_table(index="pid", columns="spd", values="t0")[speeds].dropna()
    return friedmanchisquare(*[w[s] for s in speeds]).pvalue, len(w)

def panel(ax, d, speeds, floor, ylim, title, method):
    for i, s in enumerate(speeds):
        v = d[d.spd == s].t0.values
        ax.scatter(i + 0.24 + np.random.default_rng(7 + i).uniform(-0.12, 0.12, len(v)), v, s=18, color=SC[s], alpha=0.6, zorder=2)
        m, sd = v.mean(), v.std(ddof=1)
        ax.errorbar(i, m, yerr=sd, fmt="o", color="k", mfc=SC[s], mec="k", ms=12, capsize=6, lw=1.6, zorder=4)
        ax.text(i - 0.17, m, f"{m:.0f} ms", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.axhline(floor, color="#777", ls=":", lw=1.3)
    ax.text(len(speeds) - 0.45, floor + (ylim[1] - ylim[0]) * 0.012, f"floor {floor} ms", ha="right", va="bottom", fontsize=8.5, style="italic", color="#888")
    nf = int(((d.t0 - floor) < 2).sum()) if method == "B" else int((d.t0 <= floor + 0.5).sum())
    p, n = fried(d, speeds)
    ax.set_xticks(range(len(speeds))); ax.set_xticklabels([f"{s} deg/s" + ("\n(stationary)" if s == 0 else "") for s in speeds])
    ax.set_xlim(-0.75, len(speeds) - 0.3); ax.set_ylim(*ylim); ax.set_ylabel("$t_0$ (ms)")
    ax.set_title(f"{title}\n{nf}/{len(d)} cells at the floor  ·  Friedman p = {p:.3f} (n = {n})", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, axis="y", ls="--", alpha=0.3)
    return nf, p, n

def ndt_figure(method, fname, label):
    T = tables(method); stats = {}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10.5), gridspec_kw={"width_ratios": [3, 4]})
    for row, (k, floor, eff) in enumerate([(0, 130, "Hand"), (1, 70, "Saccade")]):
        allv = pd.concat([T[P][k].t0 for P in ["P1", "P2"]]); ylim = (min(floor - 12, allv.min() - 6), allv.max() + 8)
        for col, P in enumerate(["P1", "P2"]):
            stats[(P, eff)] = panel(ax[row, col], T[P][k], SP[P], floor, ylim, f"{NAME[P]} — {eff} $t_0$", method)
            stats[(P, eff, "means")] = T[P][k].groupby("spd").t0.mean().round(1).to_dict()
            if eff == "Saccade":   # most saccade values sit on the floor, so differences between speeds are not a finding
                ax[row, col].set_title(ax[row, col].get_title() + "\n(values held near the 70 ms floor — not a speed effect)", fontsize=11, fontweight="bold")
    note = ("Saccade panels: single-component cells only (two-component cells excluded: "
            f"P1 {T['P1'][2]}, P2 {T['P2'][2]})." if method == "B" else
            "Saccade panels: two-component cells use their regular component's t0, as in NDT_barchart.")
    fig.suptitle(f"Non-decision time ($t_0$) by target speed — {label}: Paradigm 1 vs Paradigm 2\n"
                 "same chart, same y-scale per row; small dots = participants, big dot = group mean, bars = ±1 SD",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.01, note + "  Colours: 75 and 150 deg/s are the same colour in both paradigms.", ha="center", fontsize=9, color="#555")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname + ".png"), dpi=130, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, fname + ".pdf"), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return stats

def lag_figure():
    raw = {"P1": pd.read_csv(os.path.join(REPO, "Working Iterations", "SNL RT Research", "pooled_data.csv")),
           "P2": pd.read_csv(os.path.join(P2C, "pooled_data_P2.csv"), usecols=["Participant", "BlockType", "Speed_deg_per_s", "HandRT_ms", "GazeSRT_ms"])}
    V, M = {}, {}
    for P, d in raw.items():
        d = d[d.BlockType == ("I" if P == "P1" else "P2")]
        d = d[d.HandRT_ms.between(150, 800) & d.GazeSRT_ms.between(80, 600)]
        for (pid, s), z in d.groupby(["Participant", "Speed_deg_per_s"]):
            x = np.sort((z.HandRT_ms - z.GazeSRT_ms).values)
            V.setdefault((P, s), []).append([b.mean() for b in np.array_split(x, 20)]); M.setdefault((P, s), []).append(x.mean())
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for j, s in enumerate([75, 150]):
        for P in ["P1", "P2"]:
            a = np.array(V[(P, s)]); m, sd = a.mean(0), a.std(0, ddof=1); x = np.arange(1, 21)
            ax[j].fill_between(x, m - sd, m + sd, color=LINE[P], alpha=0.15); ax[j].plot(x, m, "-o", ms=4, color=LINE[P], lw=2, label=NAME[P])
        ax[j].axhline(0, color="#777", lw=1, ls="--"); ax[j].set_xticks([1, 5, 10, 15, 20])
        ax[j].set_xlabel("vincentile bin (1 = trials where the hand led the eye most, 20 = lagged most)"); ax[j].set_ylabel("HRT − SRT on the same trial (ms)")
        ax[j].set_title(f"{'AB'[j]}.  Eye-to-hand lag at {s} deg/s", fontsize=11, fontweight="bold"); ax[j].legend(fontsize=9, loc="upper left")
        ax[j].spines[["top", "right"]].set_visible(False); ax[j].grid(True, ls="--", alpha=0.3)
    out = {}
    for P in ["P1", "P2"]:
        sp = SP[P]; m = [np.mean(M[(P, s)]) for s in sp]; sd = [np.std(M[(P, s)], ddof=1) for s in sp]; out[P] = dict(zip(sp, np.round(m, 1)))
        ax[2].errorbar(sp, m, yerr=sd, fmt="-o", color=LINE[P], ms=8, capsize=5, lw=2, label=NAME[P])
    ax[2].set_xticks([0, 75, 100, 125, 150]); ax[2].set_xlabel("target speed (deg/s)"); ax[2].set_ylabel("mean HRT − SRT (ms), ±1 SD across participants")
    ax[2].set_title("C.  Average eye-to-hand lag by speed", fontsize=11, fontweight="bold"); ax[2].legend(fontsize=9)
    ax[2].spines[["top", "right"]].set_visible(False); ax[2].grid(True, ls="--", alpha=0.3)
    fig.suptitle("How much later the hand starts than the eyes — Paradigm 1 vs Paradigm 2 (trials with both a valid hand and eye RT)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "P1_vs_P2_eye_hand_lag.png"), dpi=130, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, "P1_vs_P2_eye_hand_lag.pdf"), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return out

if __name__ == "__main__":
    for meth, f, lab in [("B", "P1_vs_P2_NDT_bayesian", "hierarchical Bayesian (Method B)"), ("A", "P1_vs_P2_NDT_methodA", "MLE (Method A)")]:
        st = ndt_figure(meth, f, lab); print(meth, {k: v for k, v in st.items()})
    print("lag means", lag_figure())
