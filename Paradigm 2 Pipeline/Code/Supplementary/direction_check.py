"""
direction_check.py  --  Paradigm 2 (supplementary): Left vs Right targets

The lab sees a launch-angle / variability asymmetry between Left and Right targets. The RT
model can say whether that asymmetry also reaches the time to START moving and, if so,
whether it sits in the decision stage (v, a) or the non-decision stage (t0). Every production
fit pools both directions within a participant x speed cell (as in Paradigm 1), so this
also checks that pooling is harmless.

A) model-free: median HRT and SRT per participant x speed x direction; Right - Left, Wilcoxon.
B) Method A single Wald per participant x speed x direction (DDM_fit.fit_single, hand floor
   130 ms, 5% contamination -- the production fitter, imported unchanged); Right - Left for
   t0, v, a, Wilcoxon (16 participants per speed; "all" = participant mean over speeds).
C) aiming (descriptive): share of trials with the hand ahead of the target (signed error > 0)
   and mean signed error per speed x direction, at both measurement points in the file
   (SignedError_deg, SignedError_deg_HRT50); plus the within-cell Spearman correlation of
   HandRT with SignedError_deg -- the continuous stand-in for Paradigm 1's lead-vs-lag RT
   comparison, which cannot be run here because 91-99% of trials lead.
Outputs: direction_check_cells.csv, direction_check_summary.csv
Run    : python direction_check.py
"""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from DDM_fit import fit_single, HRT_FLOOR, P_CONTAM
SPEEDS = [75, 100, 125, 150]

def wx(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.mean(x)), int((x > 0).sum()), len(x), float(wilcoxon(x).pvalue) if len(x) > 5 and np.any(x != 0) else np.nan)

def main():
    d = pd.read_csv(os.path.join(HERE, "pooled_data_P2.csv")); d = d[d.BlockType == "P2"]
    cells = []
    for (pid, spd, dr), z in d.groupby(["Participant", "Speed_deg_per_s", "Direction"]):
        h = z.HandRT_ms.values.astype(float); hk = (~np.isnan(h)) & (h >= 150) & (h <= 800)
        g = z.GazeSRT_ms.values.astype(float); gk = (~np.isnan(g)) & (g >= 80) & (g <= 600)
        x, _, ks = fit_single(h[hk] / 1000.0, HRT_FLOOR, P_CONTAM)
        se = z.SignedError_deg.values.astype(float); ok = hk & np.isfinite(se)
        se50 = z.SignedError_deg_HRT50.values.astype(float)
        rho = spearmanr(h[ok], se[ok]).correlation if ok.sum() > 10 else np.nan
        cells.append(dict(pid=pid, spd=int(spd), direction=dr, n_hand=int(hk.sum()), n_eye=int(gk.sum()),
                          hrt_median=float(np.median(h[hk])), srt_median=float(np.median(g[gk])),
                          t0=x[2] * 1000, v=x[0], a=x[1], ks=ks,
                          lead_share=float(np.nanmean(se[np.isfinite(se)] > 0)), mean_signed_error=float(np.nanmean(se)),
                          lead_share_HRT50=float(np.nanmean(se50[np.isfinite(se50)] > 0)), mean_signed_error_HRT50=float(np.nanmean(se50)),
                          rho_hrt_vs_signed_error=rho))
    c = pd.DataFrame(cells); c.round(4).to_csv(os.path.join(HERE, "direction_check_cells.csv"), index=False)
    rows = []
    for m in ["hrt_median", "srt_median", "t0", "v", "a", "mean_signed_error", "mean_signed_error_HRT50"]:
        w = c.pivot_table(index=["pid", "spd"], columns="direction", values=m)
        diff = (w["Right"] - w["Left"]).unstack("spd")
        for spd in SPEEDS + ["all"]:
            x = diff.mean(axis=1) if spd == "all" else diff[spd]
            mean, npos, n, p = wx(x)
            rows.append(dict(measure=m, speed=spd, right_minus_left_mean=round(mean, 3), n_right_greater=npos, n=n, wilcoxon_p=round(p, 4)))
    for m in ["lead_share", "lead_share_HRT50"]:
        for spd in SPEEDS:
            for dr in ["Left", "Right"]:
                v = c[(c.spd == spd) & (c.direction == dr)][m]
                rows.append(dict(measure=f"{m}_{dr}", speed=spd, right_minus_left_mean=round(v.mean(), 3), n=len(v)))
    r = c.groupby(["pid", "spd"]).rho_hrt_vs_signed_error.mean().unstack("spd")
    for spd in SPEEDS:
        mean, npos, n, p = wx(r[spd])
        rows.append(dict(measure="spearman_rho_HRT_vs_signed_error", speed=spd, right_minus_left_mean=round(mean, 3),
                         n_right_greater=npos, n=n, wilcoxon_p=round(p, 4)))
    s = pd.DataFrame(rows); s.to_csv(os.path.join(HERE, "direction_check_summary.csv"), index=False)
    print(s.to_string(index=False)); print("saved direction_check_cells.csv, direction_check_summary.csv")
    print("NOTE: rows 'lead_share_*' hold the plain mean share in the right_minus_left_mean column; rho rows hold mean rho (n_right_greater = n positive).")

if __name__ == "__main__":
    main()
