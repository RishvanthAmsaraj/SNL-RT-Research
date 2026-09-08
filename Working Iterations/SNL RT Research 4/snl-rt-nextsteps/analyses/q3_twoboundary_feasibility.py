#!/usr/bin/env python
"""
q3_twoboundary_feasibility.py -- run the power analysis on YOUR numbers.

    python q3_twoboundary_feasibility.py --error-rate 0.06 --n 110
    python q3_twoboundary_feasibility.py --error-rate 0.06 --n 110 --reps 40

q1 tells you whether a directional outcome exists and what the error rate is. This
takes that number and answers the professor's question directly: at that error rate
and that many trials, can a two-boundary DDM actually be identified?

WHY THIS IS THE RIGHT FRAMING
-----------------------------
"Drift diffusion models work better when there are right and wrong cases" is exactly
right, and the reason is structural: with an unbiased start,

    P(error) = 1 / (1 + exp(v*a))

so the error rate IS a statement about v*a. All the extra information a two-boundary
model has over a single-boundary one lives in the error RT distribution. Few errors,
little extra information, and the additional parameters (notably the starting point
w) go unidentified while still costing you degrees of freedom.

WHAT TO DO WITH A MARGINAL RESULT
---------------------------------
Per-cell infeasible does not mean project-infeasible. With P participants x n trials
per speed you have P*n*p_err error trials at the GROUP level. A hierarchical
two-boundary model with participant random effects can be identified from the pooled
error distribution even when no individual cell can. That is the version worth
attempting, and it is the same partial-pooling argument that already justifies your
hierarchical Wald.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

from ddm2 import power_analysis, p_upper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-rate", type=float, default=None,
                    help="observed initial-direction error rate (from q1)")
    ap.add_argument("--n", type=int, default=110, help="trials per cell")
    ap.add_argument("--participants", type=int, default=16)
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()

    print("=" * 76)
    print("Q3  TWO-BOUNDARY FEASIBILITY")
    print("=" * 76)

    rates = [0.02, 0.05, 0.10, 0.20, 0.30]
    if args.error_rate is not None and args.error_rate not in rates:
        rates = sorted(set(rates + [round(args.error_rate, 4)]))

    print(f"\n  Simulating recovery at n={args.n} trials/cell, {args.reps} reps each.")
    print("  (v and a chosen to produce each target error rate; t0=160ms, w=0.5)\n")
    rows = power_analysis(rates, args.n, n_reps=args.reps, verbose=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "q3_twoboundary_power.csv"), index=False)

    print("\n  Reading the columns:")
    print("    RMSE v    - error in the drift rate. Compare to typical v ~ 1-3.")
    print("    MAE t0    - error in non-decision time, in ms. Your effect sizes are")
    print("                ~20 ms between speeds, so anything near that is fatal.")
    print("    MAE w     - error in the starting point. w is the parameter that ONLY")
    print("                the error trials can identify, so it degrades first.")
    print("    converged - fits that did not fail outright.")

    if args.error_rate is not None:
        pe = args.error_rate
        per_cell = pe * args.n
        pooled = pe * args.n * args.participants
        print("\n" + "=" * 76)
        print(f"YOUR SITUATION: error rate {pe:.4f}, {args.n} trials/cell, "
              f"{args.participants} participants")
        print("=" * 76)
        print(f"  error trials per cell            : {per_cell:.1f}")
        print(f"  error trials per speed, pooled   : {pooled:.0f}")
        print(f"  implied v*a                      : {np.log((1-pe)/pe):.2f}")
        print()
        if per_cell < 5:
            print("  >>> PER-CELL: not feasible. Expect fit failures and an")
            print("      unidentified starting point.")
            print(f"  >>> HIERARCHICAL: {pooled:.0f} pooled error trials per speed is")
            print("      enough to attempt a group-level model. Fit participants as")
            print("      random effects with a shared w, and check whether the w")
            print("      posterior is actually updated away from its prior -- if it")
            print("      is not, the errors are not informing the model and you should")
            print("      report the single-boundary result.")
        elif per_cell < 12:
            print("  >>> PER-CELL: marginal. Fits will run but v and w carry real error.")
            print("  >>> Go hierarchical; report per-cell fits only as a sensitivity check.")
        else:
            print("  >>> PER-CELL: feasible. Fit it, and compare to the single-boundary")
            print("      Wald using the same machinery as q2 (Vuong, CV log-lik,")
            print("      bootstrap KS). Note the models are NOT nested, so a likelihood")
            print("      ratio test is not appropriate.")

        print("\n  WHAT A TWO-BOUNDARY FIT WOULD BUY YOU")
        print("    - a non-decision term that is not forced against a floor, because")
        print("      the error RT distribution constrains t0 from a second direction")
        print("    - comparability between hand and eye in one model class")
        print("    - a principled account of interception as a directional decision")
        print("  WHAT IT COSTS")
        print("    - two extra parameters per cell (w, and the second boundary)")
        print("    - a strong assumption that interception IS a discrete left/right")
        print("      choice, which for a predictable trajectory it may not be")
    else:
        print("\n  Run q1_direction_audit.py first, then pass --error-rate.")

    print(f"\nWrote q3_twoboundary_power.csv")


if __name__ == "__main__":
    main()
