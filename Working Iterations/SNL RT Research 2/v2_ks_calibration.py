#!/usr/bin/env python
"""
v2_ks_calibration.py (v2) -- calibrated goodness-of-fit p-values, memory-safe.

    python v2_ks_calibration.py --B 200

CHANGES FROM v1
---------------
The v1 script held every cell's work in memory and wrote once at the end, so a
B=200 run over 96 cells could exhaust memory and lose everything. This version:

  * writes each cell's row to CSV as it completes  -> RESUMABLE
  * skips cells already present in the output      -> restart after any interruption
  * uses the direct-ECDF KS (about 6x faster than scipy.stats.kstest with a callable)

Re-running the same command after an interruption picks up where it stopped. Delete
v2_ks_calibration.csv, or pass --restart, to start over.

WHY B MATTERS -- read before interpreting any p-value
------------------------------------------------------
The smallest achievable p-value is 1/(B+1), and p is QUANTISED to that grid.

    B = 50   -> p in {0.020, 0.039, 0.059, 0.078, ...}, minimum 0.020
    B = 200  -> p in {0.005, 0.010, 0.015, ...},        minimum 0.005

At B=50 a reported median p of 0.049 is not an achievable value -- it is the average
of two adjacent grid points (0.039 and 0.059) straddling the threshold. You cannot
tell which side of 0.05 the median sits on. Use B >= 200 for anything you report.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    from wald_fast import bootstrap_ks, HRT_FLOOR, SRT_FLOOR
except ImportError:
    sys.exit("ERROR: wald_fast.py (v2) must sit in this folder.")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "v2_ks_calibration.csv")


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
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--restart", action="store_true", help="delete existing output first")
    args = ap.parse_args()

    if args.restart and os.path.exists(OUT):
        os.remove(OUT)
        print("removed existing output; starting over\n")

    if args.B < 200:
        print(f"WARNING: B={args.B}. Smallest achievable p = {1/(args.B+1):.3f}, and p is")
        print(f"         quantised to multiples of 1/{args.B+1}. Fine for a look, NOT for")
        print(f"         anything you report. Use --B 200 or more.\n")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.effector, prev.pid, prev.spd))
        print(f"resuming: {len(done)} cells already complete\n")

    dfi = pd.read_csv(_need("pooled_data.csv"))
    dfi = dfi[dfi["BlockType"] == "I"]
    header = not os.path.exists(OUT)

    for tag, fits_file, col, lo, hi, floor in [
            ("HRT", "DDM_hrt_fits.csv", "HandRT_ms", 150, 800, HRT_FLOOR),
            ("SRT", "DDM_srt_fits.csv", "GazeSRT_ms", 80, 600, SRT_FLOOR)]:
        pub = pd.read_csv(_need(fits_file))
        for _, r in pub.iterrows():
            key = (tag, r.pid, int(r.spd))
            if key in done:
                continue
            rts = load_cell(dfi, r.pid, int(r.spd), col, lo, hi)
            if len(rts) < 15:
                continue
            b = bootstrap_ks(rts, floor, B=args.B, seed=int(abs(hash(key)) % 10 ** 6))
            ks_pub = (float(r.ks_single) if "ks_single" in pub.columns
                      and pd.notna(r.get("ks_single")) else float(r.ks))
            row = dict(effector=tag, pid=r.pid, spd=int(r.spd), n=len(rts),
                       model_pub=r.get("model", "single"),
                       ks_pub=ks_pub, ks_new=round(b["ks"], 4),
                       ks_p=round(b["ks_p"], 4), p_floor=round(b["p_floor"], 4),
                       crit95=round(b["crit95"], 4), crit90=round(b["crit90"], 4),
                       rejected=b["rejected"], B=b["B"],
                       passes_old_rule=bool(b["ks"] < 0.10),
                       at_bound=b["at_bound"], v_at_cap=b["v_at_cap"],
                       t0_at_floor=b["t0_at_floor"])
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=header, index=False)
            header = False
            print(f"  {tag} {r.pid}@{int(r.spd):<4d} ks={b['ks']:.4f} "
                  f"p={b['ks_p']:.3f} crit95={b['crit95']:.4f}"
                  f"{'  <-- REJECTED' if b['rejected'] else ''}", flush=True)

    out = pd.read_csv(OUT)
    print("\n" + "=" * 74)
    print(f"V2  CALIBRATED GOODNESS OF FIT   (B = {args.B})")
    print("=" * 74)
    for tag in ("HRT", "SRT"):
        s = out[out.effector == tag]
        if not len(s):
            continue
        n_at_floor = int((s.ks_p <= s.p_floor + 1e-9).sum())
        print(f"\n{tag}  ({len(s)} cells)")
        print(f"    median KS                     : {s.ks_new.median():.4f}")
        print(f"    median calibrated 5% crit val : {s.crit95.median():.4f}   "
              f"(old rule used 0.10)")
        print(f"    median p-value                : {s.ks_p.median():.3f}")
        print(f"    rejected at p < 0.05          : {s.rejected.sum()}/{len(s)}")
        print(f"    pinned at the p resolution    : {n_at_floor}  "
              f"(p = {s.p_floor.iloc[0]:.3f}; raise B to resolve)")
        print(f"    would pass the old ks < 0.10  : {s.passes_old_rule.sum()}/{len(s)}")

        disagree = s[s.rejected & s.passes_old_rule]
        if len(disagree):
            print(f"\n    PASS the old rule but FAIL calibration ({len(disagree)}):")
            for _, w in disagree.iterrows():
                print(f"        {w.pid}@{int(w.spd):<4d} ks={w.ks_new:.4f} "
                      f"p={w.ks_p:.3f} crit95={w.crit95:.4f}")

        bad = s[s.rejected & (s.model_pub == "single")]
        if len(bad):
            rec = bad.pid.value_counts()
            rec = rec[rec > 1]
            if len(rec):
                print(f"\n    participants rejected in >1 cell: "
                      + ", ".join(f"{p} ({c})" for p, c in rec.items()))
                print("      -> recurring participants indicate structural heterogeneity,")
                print("         not cell-level noise. Cross-check their express_frac in")
                print("         LATER_fits.csv before calling these fit failures.")

        capped = s[s.rejected & s.v_at_cap]
        if len(capped):
            print(f"\n    rejected AND drift pinned at the cap: {len(capped)}")
            print("      -> the rejection may be the CAP, not the model. Refit these")
            print("         with a wider v bound before concluding the Wald fails.")

        dks = (s.ks_new - s.ks_pub).abs()
        print(f"\n    max |ks_new - ks_pub| = {dks.max():.4f}  "
              f"({'OK' if dks.max() < 0.005 else 'CHECK v1 PARITY FIRST'})")

    print("\n" + "-" * 74)
    print(f"Wrote {os.path.basename(OUT)}  ({len(out)} cells)")


if __name__ == "__main__":
    main()
