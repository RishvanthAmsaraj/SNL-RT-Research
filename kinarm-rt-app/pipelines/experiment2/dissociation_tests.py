"""
dissociation_tests.py  --  Paradigm 2 (supplementary): speed-effect battery on non-decision time

Paradigm 1 supported "hand t0 changes with target speed" with a Friedman test (pipeline),
plus a participant-resampling bootstrap and a within-participant permutation test
(kinarm-rt-app, kinarm_rt/stats_tests.py). The three functions below are exact ports of
those app functions (same statistics, same resampling scheme, same seeds); only the
condition labels are speeds instead of condition indices.

One addition, because Paradigm 2 has four ORDERED speeds: a per-participant linear slope
of t0 on speed (ms per 25 deg/s), tested against zero with a two-sided Wilcoxon
signed-rank test. Friedman stays the primary (pre-specified, as in Paradigm 1) test.

Inputs : Bayesian_hrt_fits.csv (primary), DDM_hrt_fits.csv, DDM_srt_fits.csv
Output : dissociation_tests_P2.csv
Run    : python dissociation_tests.py
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
SPEEDS = [75, 100, 125, 150]
N_RESAMPLE = 3000          # the app's dissociation_report() default
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run the fits first.")
    return p

# ------------------------------------------------ exact ports of kinarm_rt/stats_tests.py
def _wide(t0_table):
    w = t0_table.pivot_table(index="participant", columns="condition", values="t0_ms", aggfunc="mean")
    return w.dropna()

def friedman_t0_by_speed(t0_table):
    w = _wide(t0_table)
    stat, p = friedmanchisquare(*[w[c].values for c in w.columns])
    return {"n": int(w.shape[0]), "statistic": float(stat), "p_value": float(p)}

def bootstrap_speed_effect(t0_table, lo_cond, hi_cond, n_boot=N_RESAMPLE, seed=0):
    w = _wide(t0_table)
    diff = (w[hi_cond] - w[lo_cond]).values
    rng = np.random.default_rng(seed); n = len(diff)
    boot = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "mean_diff_ms": float(diff.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}

def permutation_test(t0_table, n_perm=N_RESAMPLE, seed=0):
    w = _wide(t0_table); mat = w.values
    obs, _ = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    rng = np.random.default_rng(seed); count = 0
    for _ in range(n_perm):
        pm = np.apply_along_axis(rng.permutation, 1, mat)
        stat, _ = friedmanchisquare(*[pm[:, j] for j in range(pm.shape[1])])
        if stat >= obs: count += 1
    return {"n": int(w.shape[0]), "observed_statistic": float(obs), "p_value": float((count + 1) / (n_perm + 1))}

# ------------------------------------------------ addition: per-participant trend
def slope_test(t0_table, n_boot=N_RESAMPLE, seed=0):
    w = _wide(t0_table)
    x = (np.array(w.columns, float) - 75.0) / 25.0                     # 0,1,2,3 -> ms per 25 deg/s step
    slopes = np.array([np.polyfit(x, w.loc[p].values, 1)[0] for p in w.index])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(slopes, size=len(slopes), replace=True).mean() for _ in range(n_boot)])
    return {"n": len(slopes), "mean_slope_ms_per_25": float(slopes.mean()),
            "ci95_lo": float(np.percentile(boot, 2.5)), "ci95_hi": float(np.percentile(boot, 97.5)),
            "n_negative": int((slopes < 0).sum()), "wilcoxon_p": float(wilcoxon(slopes).pvalue)}

def srt_t0_table(ds):   # same rule as NDT_barchart.py: t0 (single) or t0r (regular component of a mixture cell)
    rows = []
    for _, r in ds.iterrows():
        t0 = r["t0r"] if (r.get("model") == "mixture" and pd.notna(r.get("t0r"))) else r.get("t0")
        if pd.notna(t0): rows.append(dict(pid=r["pid"], spd=int(r["spd"]), t0=float(t0)))
    return pd.DataFrame(rows)

def main():
    sets = {"hand_MethodB_Bayesian": pd.read_csv(_need("Bayesian_hrt_fits.csv"))[["pid", "spd", "t0"]],
            "hand_MethodA_MLE": pd.read_csv(_need("DDM_hrt_fits.csv"))[["pid", "spd", "t0"]],
            "saccade_MethodA_MLE": srt_t0_table(pd.read_csv(_need("DDM_srt_fits.csv")))}
    rows = []
    for name, t in sets.items():
        tt = t.rename(columns={"pid": "participant", "spd": "condition", "t0": "t0_ms"})
        means = tt.groupby("condition").t0_ms.mean()
        f = friedman_t0_by_speed(tt); b = bootstrap_speed_effect(tt, 75, 150)
        p = permutation_test(tt); s = slope_test(tt)
        rows.append(dict(set=name, n=f["n"], **{f"mean_t0_{c}": round(means[c], 1) for c in SPEEDS},
                         friedman_chi2=round(f["statistic"], 3), friedman_p=round(f["p_value"], 4),
                         perm_p=round(p["p_value"], 4),
                         boot_150_minus_75_ms=round(b["mean_diff_ms"], 1), boot_ci_lo=round(b["ci95_lo"], 1),
                         boot_ci_hi=round(b["ci95_hi"], 1), boot_excludes_zero=b["excludes_zero"],
                         slope_ms_per_25=round(s["mean_slope_ms_per_25"], 2), slope_ci_lo=round(s["ci95_lo"], 2),
                         slope_ci_hi=round(s["ci95_hi"], 2), slope_n_negative=s["n_negative"],
                         slope_wilcoxon_p=round(s["wilcoxon_p"], 4)))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, "dissociation_tests_P2.csv"), index=False)
    print(out.T.to_string()); print("saved dissociation_tests_P2.csv")

if __name__ == "__main__":
    main()
