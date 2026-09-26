"""
P1_parity_check.py  --  validation: does THIS environment reproduce the published Paradigm 1 fits?

Procedure (see Documents/VALIDATION_P2.md):
  1. copy the ORIGINAL Paradigm 1 scripts DDM_fit.py and Bayesian_HRT_fit.py, unmodified, plus
     the Paradigm 1 pooled_data.csv, into an empty folder and run them there;
  2. python P1_parity_check.py <published Current Pipeline/Code folder> <rerun folder>
Compares the rerun against the committed DDM_hrt_fits.csv, DDM_srt_fits.csv (Method A:
deterministic, should match to the stored rounding) and Bayesian_hrt_fits.csv (Method B:
MCMC, should match to Monte Carlo error).
Output: P1_parity_results.csv
"""
import os, sys, numpy as np, pandas as pd
pub, new = sys.argv[1], sys.argv[2]
def rd(folder, sub, f):
    for p in (os.path.join(folder, sub, f), os.path.join(folder, f)):
        if os.path.exists(p): return pd.read_csv(p)
    sys.exit(f"missing {f} in {folder}")
rows = []
for f, sub, cols in [("DDM_hrt_fits.csv", "DDM", ["v", "a", "t0", "ks"]),
                     ("DDM_srt_fits.csv", "DDM", ["v", "a", "t0", "ks", "pi", "express_mode", "reg_mode"]),
                     ("Bayesian_hrt_fits.csv", "Bayesian", ["v", "a", "t0", "t0_lo95", "t0_hi95"])]:
    a, b = rd(pub, sub, f), rd(new, sub, f)
    m = a.merge(b, on=["pid", "spd"], suffixes=("_pub", "_new"))
    row = dict(table=f, cells_published=len(a), cells_rerun=len(b), cells_matched=len(m))
    if "model_pub" in m: row["model_choice_agree"] = int((m.model_pub == m.model_new).sum())
    for c in cols:
        d = (m[f"{c}_new"] - m[f"{c}_pub"]).abs()
        row[f"max_abs_diff_{c}"] = round(float(d.max()), 4) if d.notna().any() else np.nan
        if c == "t0": row["mean_abs_diff_t0"] = round(float(d.mean()), 3); row["corr_t0"] = round(float(m[f"t0_pub"].corr(m[f"t0_new"])), 4)
    if f.startswith("Bayesian"):
        g = m.groupby("spd")[["t0_pub", "t0_new"]].mean().round(1)
        row["group_t0_pub"] = "/".join(str(x) for x in g.t0_pub); row["group_t0_new"] = "/".join(str(x) for x in g.t0_new)
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "P1_parity_results.csv"), index=False)
print(out.T.to_string())
