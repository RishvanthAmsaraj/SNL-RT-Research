"""
P1_vs_P2_comparison.py  --  side-by-side comparison of Paradigm 1 (CMT; 0/75/150 deg/s) and Paradigm 2
(CIR; 75/100/125/150 deg/s), computed from each paradigm's committed result tables and raw trials.

Same definitions for both paradigms throughout: interception trials only (P1 BlockType "I", P2 "P2"),
hand 150-800 ms, saccade 80-600 ms, Method B (Bayesian) hand t0 and decision time a/v from
Bayesian_hrt_fits.csv, participant-level saccadic t0 from Bayesian_srt_ndt.csv, per-cell bounds from the
Method A tables. Group intervals: participant bootstrap, 3000 resamples, seed 0 (as in the pipeline).

Figures (Figures/Comparison/):
  P1_vs_P2_hand.pdf/.png        hand t0, decision time, raw RT by speed; the t0 / decision-time trade-off
  P1_vs_P2_saccade.pdf/.png     saccadic t0 per participant against both bounds; raw SRT; per-cell bounds;
                                floor-sweep slopes (same code run on both paradigms)
  P1_vs_P2_distributions.pdf/.png   pooled HRT / SRT distributions at the shared speeds (75, 150)
Table: P1_vs_P2_summary.csv (every number the figures and documents quote)

Run from anywhere inside the repo:  python P1_vs_P2_comparison.py
Paths default to the repo layout; override with --repo <repo root>.
"""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.stats import wilcoxon, gaussian_kde
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family": _fam, "font.size": 10.5, "pdf.fonttype": 42, "ps.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P1C = os.path.join(REPO, "Current Pipeline", "Code"); P2C = os.path.join(REPO, "Paradigm 2 Pipeline", "Code")
P1_DATA = os.path.join(REPO, "Working Iterations", "SNL RT Research", "pooled_data.csv")
P2_DATA = os.path.join(P2C, "pooled_data_P2.csv")
FIG = os.path.join(REPO, "Paradigm 2 Pipeline", "Figures", "Comparison"); os.makedirs(FIG, exist_ok=True)
SP = {"P1": [0, 75, 150], "P2": [75, 100, 125, 150]}
COL = {"P1": "#1b7f79", "P2": "#d95f02"}; MK = {"P1": "s", "P2": "o"}
LAB = {"P1": "Paradigm 1 (CMT)", "P2": "Paradigm 2 (CIR)"}
N_BOOT, SEED = 3000, 0

def need(p):
    if not os.path.exists(p): sys.exit(f"ERROR: missing {p}")
    return p

def boot_ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; r = np.random.default_rng(SEED)
    b = np.array([r.choice(x, len(x)).mean() for _ in range(N_BOOT)]); return np.percentile(b, [2.5, 97.5])

def load():
    raw = {"P1": pd.read_csv(need(P1_DATA)), "P2": pd.read_csv(need(P2_DATA), usecols=["Participant", "BlockType", "Speed_deg_per_s", "HandRT_ms", "GazeSRT_ms"])}
    raw["P1"] = raw["P1"][raw["P1"].BlockType == "I"]; raw["P2"] = raw["P2"][raw["P2"].BlockType == "P2"]
    tab = {"P1": P1C, "P2": P2C}; out = {}
    for P, d in raw.items():
        h = d[d.HandRT_ms.between(150, 800)]; s = d[d.GazeSRT_ms.between(80, 600)]
        cell = pd.DataFrame({"hrt_med": h.groupby(["Participant", "Speed_deg_per_s"]).HandRT_ms.median(),
                             "hrt_p10": h.groupby(["Participant", "Speed_deg_per_s"]).HandRT_ms.quantile(0.10),
                             "hrt_min": h.groupby(["Participant", "Speed_deg_per_s"]).HandRT_ms.min(),
                             "srt_med": s.groupby(["Participant", "Speed_deg_per_s"]).GazeSRT_ms.median(),
                             "srt_min": s.groupby(["Participant", "Speed_deg_per_s"]).GazeSRT_ms.min()}).reset_index()
        cell = cell.rename(columns={"Participant": "pid", "Speed_deg_per_s": "spd"})
        bh = pd.read_csv(need(os.path.join(tab[P], "Bayesian", "Bayesian_hrt_fits.csv")))[["pid", "spd", "v", "a", "t0", "t0_lo95", "t0_hi95"]]
        bh["dt"] = 1000 * bh.a / bh.v                       # mean decision time of the Wald component (ms)
        cell = cell.merge(bh, on=["pid", "spd"], how="left")
        out[P] = dict(raw=d, cell=cell, h=h, s=s,
                      ndt=pd.read_csv(need(os.path.join(tab[P], "Bayesian", "Bayesian_srt_ndt.csv"))),
                      dh=pd.read_csv(need(os.path.join(tab[P], "DDM", "DDM_hrt_fits.csv"))),
                      ds=pd.read_csv(need(os.path.join(tab[P], "DDM", "DDM_srt_fits.csv"))))
    return out

def contrast(cell, col, a, b):
    w = cell.pivot_table(index="pid", columns="spd", values=col); x = (w[b] - w[a]).dropna()
    lo, hi = boot_ci(x.values)
    return dict(mean=x.mean(), ci_lo=lo, ci_hi=hi, n_neg=int((x < 0).sum()), n=len(x), wilcoxon_p=wilcoxon(x).pvalue, values=x)

def group_line(ax, P, cell, col, label=True):
    sp = SP[P]; w = cell.pivot_table(index="pid", columns="spd", values=col)[sp]
    for _, r in w.iterrows(): ax.plot(sp, r.values, color=COL[P], lw=0.6, alpha=0.18)
    m = w.mean(); ci = np.array([boot_ci(w[s].values) for s in sp])
    ax.fill_between(sp, ci[:, 0], ci[:, 1], color=COL[P], alpha=0.16, lw=0)
    ax.plot(sp, m.values, "-", marker=MK[P], color=COL[P], lw=2.2, ms=7, label=LAB[P] if label else None)
    return m

def style(ax, title, ylab):
    ax.axvspan(75, 150, color="#999", alpha=0.07, lw=0)
    ax.set_xticks([0, 75, 100, 125, 150]); ax.set_xlim(-12, 162); ax.set_xlabel("target speed (deg/s)")
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, axis="y", ls="--", alpha=0.3)

def main():
    D = load(); rows = []
    def rec(measure, P, contrast_name, r):
        rows.append(dict(measure=measure, paradigm=P, contrast=contrast_name, mean=round(r["mean"], 2), ci_lo=round(r["ci_lo"], 2),
                         ci_hi=round(r["ci_hi"], 2), n_negative=r["n_neg"], n=r["n"], wilcoxon_p=round(r["wilcoxon_p"], 4)))
    for P in ["P1", "P2"]:
        c = D[P]["cell"]
        for col in ["t0", "dt", "hrt_med", "hrt_p10", "srt_med", "v", "a"]:
            for s, v in c.groupby("spd")[col].mean().items():
                rows.append(dict(measure=col, paradigm=P, contrast=f"group mean at {s}", mean=round(v, 2)))
    steps = [("P1", 0, 75), ("P1", 75, 150), ("P2", 75, 150)]
    C = {}
    for P, a, b in steps:
        for col in ["t0", "dt", "hrt_med", "hrt_p10", "srt_med"]:
            C[(P, a, b, col)] = contrast(D[P]["cell"], col, a, b); rec(col, P, f"{b} - {a}", C[(P, a, b, col)])
    for P in ["P1", "P2"]:
        x, y = C[(P, 75, 150, "t0")]["values"], C[(P, 75, 150, "dt")]["values"]; xy = pd.concat([x, y], axis=1).dropna()
        rows.append(dict(measure="corr(delta t0, delta decision time)", paradigm=P, contrast="150 - 75", mean=round(np.corrcoef(xy.iloc[:, 0], xy.iloc[:, 1])[0, 1], 3), n=len(xy)))

    # ------------------------------------------------ Figure 1: hand
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.6))
    for P in ["P1", "P2"]:
        group_line(ax[0, 0], P, D[P]["cell"], "t0"); group_line(ax[0, 1], P, D[P]["cell"], "dt")
        group_line(ax[1, 0], P, D[P]["cell"], "hrt_med"); group_line(ax[1, 1], P, D[P]["cell"], "hrt_p10")
    style(ax[0, 0], "A.  Hand non-decision time $t_0$ (Method B)", "$t_0$ (ms)"); ax[0, 0].legend(fontsize=9, loc="upper right")
    style(ax[0, 1], "B.  Hand decision time $a/v$ (Method B)", "mean decision time (ms)")
    style(ax[1, 0], "D.  Raw median HRT (model-free)", "HRT (ms)")
    style(ax[1, 1], "E.  Raw 10th-percentile HRT (leading edge, model-free)", "HRT (ms)")
    a3 = ax[0, 2]
    for P in ["P1", "P2"]:
        x, y = C[(P, 75, 150, "t0")]["values"], C[(P, 75, 150, "dt")]["values"]; xy = pd.concat([x, y], axis=1).dropna()
        r = np.corrcoef(xy.iloc[:, 0], xy.iloc[:, 1])[0, 1]
        a3.scatter(xy.iloc[:, 0], xy.iloc[:, 1], color=COL[P], marker=MK[P], s=38, alpha=0.85, label=f"{LAB[P]}: r = {r:.2f}")
    lim = 40; a3.plot([-lim, lim], [lim, -lim], color="#777", ls="--", lw=1, label="full compensation (RT unchanged)")
    a3.axhline(0, color="#bbb", lw=0.8); a3.axvline(0, color="#bbb", lw=0.8)
    a3.set_xlabel("change in $t_0$, 75 → 150 deg/s (ms)"); a3.set_ylabel("change in decision time, 75 → 150 (ms)")
    a3.set_title("C.  $t_0$ and decision-time changes cancel each other\n(one point per participant; r ≈ −0.75 arises from estimation noise alone)", fontsize=10.5, fontweight="bold")
    a3.legend(fontsize=8.6, loc="lower left"); a3.spines[["top", "right"]].set_visible(False); a3.grid(True, ls="--", alpha=0.3)
    a4 = ax[1, 2]; labels = ["P1  0 → 75", "P1  75 → 150", "P2  75 → 150"]
    meas = [("t0", "$t_0$", "#4c4c4c"), ("dt", "decision time", "#8c6bb1"), ("hrt_med", "median HRT", "#2c7fb8"), ("hrt_p10", "10th-pct HRT", "#41ab5d")]
    for k, (col, nm, cc) in enumerate(meas):
        for j, (P, a, b) in enumerate(steps):
            r = C[(P, a, b, col)]; y = j * 5 + k
            a4.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", color=cc, ms=6, capsize=3, lw=1.6,
                        label=nm if j == 0 else None)
            star = "**" if r["wilcoxon_p"] < 0.01 else ("*" if r["wilcoxon_p"] < 0.05 else "")
            a4.text(r["ci_hi"] + 0.8, y, f"p={r['wilcoxon_p']:.3f}{star}", va="center", fontsize=7.6, color=cc)
    a4.axvline(0, color="#777", lw=1); a4.set_yticks([j * 5 + 1.5 for j in range(3)]); a4.set_yticklabels(labels)
    a4.invert_yaxis(); a4.set_xlabel("per-participant change (ms), mean ± 95% bootstrap CI")
    a4.set_title("F.  Which steps are real?  (Wilcoxon signed-rank p)", fontsize=10.5, fontweight="bold")
    a4.legend(fontsize=8.4, loc="lower right"); a4.spines[["top", "right"]].set_visible(False); a4.grid(True, axis="x", ls="--", alpha=0.3)
    fig.suptitle("Hand: Paradigm 1 vs Paradigm 2 — the robust Paradigm 1 step is stationary → moving; between moving speeds neither paradigm shows a reliable change\n"
                 "(shaded band = speeds shared by both paradigms; thin lines = participants; bands = 95% bootstrap CI of the group mean)",
                 fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "P1_vs_P2_hand")

    # ------------------------------------------------ Figure 2: saccade + identifiability
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.5))
    a0 = ax[0, 0]; y = 0; yt, yl = [], []
    for P in ["P1", "P2"]:
        n = D[P]["ndt"].sort_values("t0_ms")
        for r in n.itertuples():
            ceil = r.min_srt_ms - 1; at_fl = r.t0_lo95 <= 71; at_ce = r.t0_hi95 >= ceil - 1
            a0.plot([r.t0_lo95, r.t0_hi95], [y, y], color=COL[P], lw=2, alpha=0.85); a0.plot(r.t0_ms, y, MK[P], color=COL[P], ms=5)
            a0.plot(ceil, y, "|", color="#888", ms=9, mew=1.5); yt.append(y); yl.append(f"{r.pid}"); y += 1
        y += 1.5
    a0.axvline(70, color="#C0392B", ls=":", lw=1.4); a0.set_yticks(yt); a0.set_yticklabels(yl, fontsize=6.8); a0.invert_yaxis()
    a0.set_xlabel("saccadic $t_0$ (ms, posterior mean ± 95% CI; grey tick = fastest saccade − 1 ms)")
    nb = {P: (int((D[P]["ndt"].t0_lo95 <= 71).sum()), int((D[P]["ndt"].t0_hi95 >= D[P]["ndt"].min_srt_ms - 2).sum()), len(D[P]["ndt"])) for P in ["P1", "P2"]}
    a0.set_title(f"A.  Participant-level saccadic $t_0$ vs its two bounds\nP1: {nb['P1'][0]}/{nb['P1'][2]} at floor, {nb['P1'][1]} at ceiling   |   "
                 f"P2: {nb['P2'][0]}/{nb['P2'][2]} at floor, {nb['P2'][1]} at ceiling", fontsize=10.5, fontweight="bold")
    a0.legend(handles=[Line2D([0], [0], color=COL[P], marker=MK[P], lw=2, label=LAB[P]) for P in ["P1", "P2"]] +
              [Line2D([0], [0], color="#C0392B", ls=":", label="70 ms floor")], fontsize=8.5, loc="lower right")
    a0.spines[["top", "right"]].set_visible(False); a0.grid(True, axis="x", ls="--", alpha=0.3)
    for P in ["P1", "P2"]: group_line(ax[0, 1], P, D[P]["cell"], "srt_med")
    style(ax[0, 1], "B.  Raw median SRT (model-free)", "SRT (ms)"); ax[0, 1].legend(fontsize=9, loc="lower right")
    a2 = ax[1, 0]; bars = []
    # Bayesian (Method B) per-cell 95% intervals vs the two bounds: the floor and the cell's fastest RT (the model's ceiling)
    for eff, f, fl, lab in [("hand", "Bayesian_hrt_fits.csv", 130, "hrt_min"), ("saccade", "Bayesian_srt_fits.csv", 70, "srt_min")]:
        for P in ["P1", "P2"]:
            t = pd.read_csv(need(os.path.join(P1C if P == "P1" else P2C, "Bayesian", f))); t = t[t.model == "single"] if "model" in t else t
            t = t.merge(D[P]["cell"][["pid", "spd", lab]], on=["pid", "spd"])
            rf = (t.t0_lo95 <= fl + 1).values; rc = (t.t0_hi95 >= t[lab] - 2).values
            cnt = (int((rf & ~rc).sum()), int((rc & ~rf).sum()), int((rf & rc).sum()), int((~rf & ~rc).sum()), len(t))
            bars.append((f"{P} {eff}",) + cnt)
            rows.append(dict(measure=f"Bayesian {eff} cells: 95% interval reaches floor only / fastest RT only / both / neither", paradigm=P,
                             contrast="per cell", mean=np.nan, n=len(t), note=" / ".join(str(c) for c in cnt[:4])))
    cols = ["#C0392B", "#E67E22", "#8E44AD", "#27AE60"]
    for i, (nm, f_, c_, b_, n_, N) in enumerate(bars):
        left = 0
        for val, cc in zip([f_, c_, b_, n_], cols):
            a2.barh(i, 100 * val / N, left=left, color=cc, alpha=0.82); left += 100 * val / N
        a2.text(101, i, f"{f_}/{c_}/{b_}/{n_} of {N}", va="center", fontsize=8.5)
    a2.set_yticks(range(len(bars))); a2.set_yticklabels([b[0] for b in bars]); a2.invert_yaxis(); a2.set_xlim(0, 128)
    a2.set_xlabel("% of cells (Bayesian fits; saccade = single-component cells)")
    a2.set_title("C.  Does each cell's 95% interval for $t_0$ run into a bound?\nred = floor, orange = the cell's fastest RT, purple = both, green = neither",
                 fontsize=10.5, fontweight="bold")
    a2.spines[["top", "right"]].set_visible(False)
    a3 = ax[1, 1]; sw = {}
    for P, path in [("P1", os.path.join(HERE, "P1_floor_sweep_rerun.csv")), ("P2", os.path.join(P2C, "Supplementary", "HRT_floor_control.csv"))]:
        if not os.path.exists(path): continue
        t = pd.read_csv(path); t = t[(t.eligible == True) & (t.ceiling_bound.fillna(False).astype(bool) == False)]
        for k, eff in enumerate(["HRT", "SRT"]):
            v = t[t.effector == eff].slope.values; xpos = k * 3 + (0 if P == "P1" else 1)
            a3.scatter(xpos + np.random.default_rng(1).uniform(-0.18, 0.18, len(v)), v, color=COL[P], marker=MK[P], s=18, alpha=0.6)
            k = int((v > 0.7).sum()); sw[(P, eff)] = (k, len(v))   # slopes form two clusters (near 0 and near 1), so report the share, not the median
            a3.text(xpos, 1.11, f"follow the floor:\n{k}/{len(v)} ({100 * k / len(v):.0f}%)", ha="center", fontsize=8.5, fontweight="bold", color=COL[P])
            rows.append(dict(measure=f"floor sweep ({eff}, eligible, not ceiling-bound): share of fits following the floor (slope > 0.7)", paradigm=P,
                             contrast="share", mean=round(100 * k / len(v), 1), n=len(v), n_negative=np.nan, note=f"{k}/{len(v)}; median slope {np.median(v):.3f}"))
    a3.axhline(0.7, color="#555", ls=":", lw=1.2); a3.set_xticks([0.5, 3.5]); a3.set_xticklabels(["hand", "saccade"]); a3.set_ylim(-0.15, 1.3)
    a3.set_ylabel("slope of fitted $t_0$ on imposed floor (1 = set by the floor)")
    a3.set_title("D.  Floor sweep, same code on both paradigms: share of fits whose $t_0$ follows the floor\n(dots = fits; above the dotted line = follows the floor; fits stuck at their fastest RT excluded)", fontsize=10.5, fontweight="bold")
    a3.legend(handles=[Line2D([0], [0], color=COL[P], marker=MK[P], ls="", label=LAB[P]) for P in ["P1", "P2"]], fontsize=8.5, loc="center right")
    a3.spines[["top", "right"]].set_visible(False); a3.grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("Saccades and identifiability: Paradigm 1 vs Paradigm 2 — hand $t_0$ identified in both; saccadic $t_0$ held by a bound in both\n"
                 "(P1 by the floor; P2 by the floor or the participant's fastest saccade)", fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "P1_vs_P2_saccade")

    # ------------------------------------------------ Figure 3: distributions at the shared speeds
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.4), sharex="row")
    for i, (col, lo, hi, nm) in enumerate([("HandRT_ms", 150, 800, "HRT"), ("GazeSRT_ms", 80, 600, "SRT")]):
        xs = np.linspace(lo, 450 if nm == "HRT" else 400, 400)
        for j, s in enumerate([75, 150]):
            a = ax[i, j]
            for P in ["P1", "P2"]:
                r = D[P]["raw"]; x = r[(r.Speed_deg_per_s == s) & r[col].between(lo, hi)][col].values
                a.fill_between(xs, gaussian_kde(x, bw_method=0.15)(xs), color=COL[P], alpha=0.22)
                a.plot(xs, gaussian_kde(x, bw_method=0.15)(xs), color=COL[P], lw=2, label=f"{LAB[P]} (n={len(x):,}, median {np.median(x):.0f} ms)")
                a.axvline(np.median(x), color=COL[P], ls="--", lw=1.2)
            a.set_title(f"{'ABCD'[i * 2 + j]}.  {nm} at {s} deg/s", fontsize=10.5, fontweight="bold"); a.set_xlabel(f"{nm} (ms)")
            a.set_ylabel("density"); a.legend(fontsize=8.5); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Raw RT distributions at the two speeds both paradigms share (pooled trials)", fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "P1_vs_P2_distributions")

    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, "P1_vs_P2_summary.csv"), index=False)
    print(out[out.contrast.astype(str).str.contains("-")].to_string(index=False)); print(f"saved P1_vs_P2_summary.csv and 3 figures to {FIG}")

def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)

if __name__ == "__main__":
    main()
