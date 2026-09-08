#!/usr/bin/env python
"""
run_all.py -- run every validation check in order and write a combined summary.

    python run_all.py              # B=200 bootstrap (report-quality)
    python run_all.py --quick      # B=50, for a first look only

Requires in this folder:
    wald_fast.py
    pooled_data.csv
    DDM_hrt_fits.csv, DDM_srt_fits.csv     (from DDM_fit.py)
    Bayesian_hrt_fits.csv, Bayesian_srt_fits.csv   (optional, enables the A-vs-B check)

Nothing here modifies your repo. Every script only READS your CSVs and writes new
v*_*.csv files alongside them.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("v1_parity_check.py", [],
     "Does the profile fitter reproduce your published DE fits?"),
    ("v2_ks_calibration.py", ["--B", "{B}"],
     "Which cells actually fail goodness of fit, once the KS test is calibrated?"),
    ("v3_mode_and_control.py", [],
     "Is express_mode the mean? And does the hand floor-sweep control hold?"),
    ("v4_intervals_and_AvB.py", [],
     "Confidence intervals for Method A, and a fair Method A vs B comparison."),
]

REQUIRED = ["wald_fast.py", "pooled_data.csv", "DDM_hrt_fits.csv", "DDM_srt_fits.csv"]
OPTIONAL = ["Bayesian_hrt_fits.csv", "Bayesian_srt_fits.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B=50 instead of 200")
    args = ap.parse_args()
    B = "50" if args.quick else "200"

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        sys.exit("ERROR: missing required files in this folder:\n  " + "\n  ".join(missing))
    absent = [f for f in OPTIONAL if not os.path.exists(os.path.join(HERE, f))]
    if absent:
        print("NOTE: Bayesian CSVs not found, so the Method A vs B comparison in v4")
        print("      will be skipped:  " + ", ".join(absent) + "\n")
    if args.quick:
        print("QUICK MODE: B=50. Fine for a first look; re-run without --quick before")
        print("            you act on any v2 result.\n")

    results = []
    for script, extra, desc in STEPS:
        args_ = [a.replace("{B}", B) for a in extra]
        print("\n" + "#" * 74)
        print(f"# {script}   --   {desc}")
        print("#" * 74)
        t = time.perf_counter()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args_,
                           cwd=HERE)
        dt = time.perf_counter() - t
        results.append((script, r.returncode, dt))
        if r.returncode != 0:
            print(f"\n!! {script} exited with code {r.returncode}. Stopping.")
            break

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for script, code, dt in results:
        print(f"  {'OK ' if code == 0 else 'ERR'}  {script:32s} {dt:7.1f}s")
    print("\nOutputs written:")
    for f in ["v1_parity_results.csv", "v2_ks_calibration.csv", "v3_mode_vs_mean.csv",
              "v3_floor_control.csv", "v4_profile_intervals.csv"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            print(f"    {f}")
    print("\nNext: work through VALIDATION_CHECKLIST.md Phase 2 onwards.")


if __name__ == "__main__":
    main()
