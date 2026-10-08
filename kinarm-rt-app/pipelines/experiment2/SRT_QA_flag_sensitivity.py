"""
SRT_QA_flag_sensitivity.py  --  Paradigm 2 (supplementary): does the extraction's QA flag matter?

The v0_1_36 extraction marks a whole trial EXCLUDED when HandRT <= 100 ms, which sets
IncludeInEyeAnalysis = 0 for 48 saccades that are inside the 80-600 ms window. The
Paradigm 1 rule (RT window only) keeps them -- Paradigm 1 kept 33 such saccades -- so the
production fits keep them too. This script reruns the exact DDM_fit.py saccade procedure
(single Wald; mixture only where single KS > 0.10 and the mixture passes the rule) on every
affected cell under BOTH rules and reports the differences.

Inputs : pooled_data_P2.csv, DDM_fit.py (functions imported unchanged)
Output : SRT_QA_flag_sensitivity.csv
Run    : python SRT_QA_flag_sensitivity.py
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from DDM_fit import fit_single, fit_mixture, SRT_FLOOR, P_CONTAM

def select(rts):
    """Verbatim decision logic of DDM_fit.py main() for one saccade cell."""
    xs, _, ks_s = fit_single(rts, SRT_FLOOR, P_CONTAM)
    if ks_s > 0.10:
        xm, _, ks_m = fit_mixture(rts, SRT_FLOOR, P_CONTAM); pi = xm[0]
        em = (xm[3] + xm[2] / xm[1]) * 1000; rm = (xm[6] + xm[5] / xm[4]) * 1000
        if (ks_m < 0.10) and (0.10 <= pi <= 0.90) and ((rm - em) >= 30):
            return dict(model="mixture", ks=ks_m, t0=np.nan, t0r=xm[6] * 1000, express_mode=em, reg_mode=rm, pi=pi)
    return dict(model="single", ks=ks_s, t0=xs[2] * 1000, v=xs[0], a=xs[1])

def main():
    d = pd.read_csv(os.path.join(HERE, "pooled_data_P2.csv")); d = d[d.BlockType == "P2"]
    win = d.GazeSRT_ms.between(80, 600)
    flagged = d[win & (d.IncludeInEyeAnalysis == 0)]
    cells = sorted(set(zip(flagged.Participant, flagged.Speed_deg_per_s)))
    print(f"{len(flagged)} in-window saccades carry IncludeInEyeAnalysis==0, in {len(cells)} cells")
    rows = []
    for pid, spd in cells:
        z = d[(d.Participant == pid) & (d.Speed_deg_per_s == spd) & win]
        a = z.GazeSRT_ms.values / 1000.0
        b = z[z.IncludeInEyeAnalysis == 1].GazeSRT_ms.values / 1000.0
        ra, rb = select(a), select(b)
        rows.append(dict(pid=pid, spd=int(spd), n_P1rule=len(a), n_QArule=len(b),
                         model_P1rule=ra["model"], model_QArule=rb["model"],
                         ks_P1rule=round(ra["ks"], 4), ks_QArule=round(rb["ks"], 4),
                         t0_P1rule=round(ra.get("t0", np.nan), 1), t0_QArule=round(rb.get("t0", np.nan), 1),
                         reg_mode_P1rule=round(ra.get("reg_mode", np.nan), 1), reg_mode_QArule=round(rb.get("reg_mode", np.nan), 1)))
    out = pd.DataFrame(rows); out["t0_diff_ms"] = (out.t0_QArule - out.t0_P1rule).round(1)
    out.to_csv(os.path.join(HERE, "SRT_QA_flag_sensitivity.csv"), index=False)
    print(out.to_string(index=False))
    same = (out.model_P1rule == out.model_QArule).sum()
    print(f"\nmodel choice unchanged in {same}/{len(out)} cells; |t0 change| (single cells): "
          f"max {out.t0_diff_ms.abs().max():.1f} ms, mean {out.t0_diff_ms.abs().mean():.1f} ms")
    print("saved SRT_QA_flag_sensitivity.csv")

if __name__ == "__main__":
    main()
