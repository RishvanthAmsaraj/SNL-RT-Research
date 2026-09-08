#!/usr/bin/env python
"""
v2_ks_calibration.py -- calibrated goodness-of-fit p-values for every cell.

    python v2_ks_calibration.py [--B 200]

WHY
---
DDM_fit.py compares the KS statistic to a fixed 0.10. That threshold assumes the
parameters are KNOWN. Yours are estimated from the same data, so the null
distribution is much tighter (the Lilliefors problem) and 0.10 is not a 5% test.

WHAT IT DOES
------------
Per cell: refits, then simulates B datasets from the fitted (v, a, t0), refits each,
and locates the observed KS in that null. Gives a real p-value plus the cell-specific
5% critical value.

WHAT TO CHECK
-------------
1. crit95 per cell. Expect ~0.09 at n~110 and ~0.08 at n~160, i.e. WELL BELOW 0.10.
   If your crit95 values come out near 0.13, something is wrong with the refit.
2. Cells with ks_p < 0.05 that are recorded as model="single" in your CSV. Those are
   formally rejected fits currently reported without qualification. From the published
   DDM_srt_fits.csv the expected offenders are CMT0012@0, CMT009@0, CMT0012@75,
   CMT009@150, and marginally CMT0014@0 and CMT0011@150.
3. The overall SRT picture. If median ks_p is comfortably above 0.05, the "SRT fits
   remain imperfect" framing is too harsh and should be softened in the writeup.
4. HRT should be clean across the board. If any HRT cell rejects, look at it directly.

WHAT TO COMPARE AGAINST
-----------------------
Your own ks / ks_single columns. The statistic should reproduce exactly (same
formula); it is the INTERPRETATION that changes. If ks_new differs from ks_pub by
more than ~0.005 on a non-boundary cell, stop -- that is a v1 parity issue, not a
calibration issue.

OUTPUT
------
    v2_ks_calibration.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

try:
    from wald_fast import bootstrap_ks, HRT_FLOOR, SRT_FLOOR
except ImportError:
    sys.exit("ERROR: wald_fast.py must sit in this folder.")

HERE = os.path.dirname(os.path.abspath(__file__))


def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        sys.exit(f"ERROR: {f} not found next to this script.")
    return p


def load_cell(dfi, pid, spd, col, lo, hi):
    sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
    x = sub[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200, help="bootstrap replicates per cell")
    args = ap.parse_args()

    if args.B < 100:
        print(f"WARNING: B={args.B} means the smallest achievable p-value is "
              f"{1/(args.B+1):.3f}.\n         Use --B 200 or more for the run you report.\n")

    dfi = pd.read_csv(_need("pooled_data.csv"))
    dfi = dfi[dfi["BlockType"] == "I"]

    rows = []
    for tag, fits_file, col, lo, hi, floor in [
            ("HRT", "DDM_hrt_fits.csv", "HandRT_ms", 150, 800, HRT_FLOOR),
            ("SRT", "DDM_srt_fits.csv", "GazeSRT_ms", 80, 600, SRT_FLOOR)]:
        pub = pd.read_csv(_need(fits_file))
        for i, (_, r) in enumerate(pub.iterrows()):
            rts = load_cell(dfi, r.pid, int(r.spd), col, lo, hi)
            if len(rts) < 15:
                continue
            b = bootstrap_ks(rts, floor, B=args.B, seed=int(abs(hash((r.pid, int(r.spd)))) % 10**6))
            ks_pub = float(r.ks_single) if "ks_single" in pub.columns and pd.notna(r.get("ks_single")) \
                else float(r.ks)
            rows.append(dict(
                effector=tag, pid=r.pid, spd=int(r.spd), n=len(rts),
                model_pub=r.get("model", "single"),
                ks_pub=ks_pub, ks_new=round(b["ks"], 4),
                ks_p=round(b["ks_p"], 4), crit95=round(b["crit95"], 4),
                rejected=b["rejected"],
                passes_old_rule=bool(b["ks"] < 0.10),
                at_bound=b["at_bound"]))
            print(f"  {tag} {r.pid}@{int(r.spd):<4d} ks={b['ks']:.4f} "
                  f"p={b['ks_p']:.3f} crit95={b['crit95']:.4f}"
                  f"{'  <-- REJECTED' if b['rejected'] else ''}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "v2_ks_calibration.csv"), index=False)

    print("\n" + "=" * 74)
    print("V2  CALIBRATED GOODNESS OF FIT")
    print("=" * 74)
    for tag in ("HRT", "SRT"):
        s = out[out.effector == tag]
        if not len(s):
            continue
        print(f"\n{tag}  ({len(s)} cells)")
        print(f"    median KS                     : {s.ks_new.median():.4f}")
        print(f"    median calibrated 5% crit val : {s.crit95.median():.4f}   "
              f"(the old rule used 0.10)")
        print(f"    median p-value                : {s.ks_p.median():.3f}")
        print(f"    rejected at p < 0.05          : {s.rejected.sum()}/{len(s)}")
        print(f"    would pass the old ks < 0.10  : {s.passes_old_rule.sum()}/{len(s)}")

        bad = s[s.rejected & (s.model_pub == "single")]
        if len(bad):
            print(f"\n    REJECTED but reported as model='single':")
            for _, w in bad.iterrows():
                print(f"        {w.pid}@{int(w.spd):<4d} n={w.n:<4d} "
                      f"ks={w.ks_new:.4f}  p={w.ks_p:.3f}  crit95={w.crit95:.4f}")

        dks = (s.ks_new - s.ks_pub).abs()
        print(f"\n    max |ks_new - ks_pub| = {dks.max():.4f}  "
              f"({'OK' if dks.max() < 0.005 else 'CHECK v1 PARITY FIRST'})")

    print("\n" + "-" * 74)
    print("Wrote v2_ks_calibration.csv")
    print("Add ks_p and crit95 as columns to your fit tables; drop the fixed 0.10 line")
    print("from DDM_figures.py panel A and B in favour of the per-cell critical value.")


if __name__ == "__main__":
    main()
