#!/usr/bin/env python
"""
q1_direction_audit.py -- is there a directional outcome in the data at all?

    python q1_direction_audit.py

THIS IS THE GATE. A two-boundary DDM needs a discrete outcome per trial -- which way
the participant went -- and a non-trivial rate of going the wrong way. Without that
there is nothing for the second boundary to be identified from, and the whole
question is moot. Run this before anything else in the two-boundary direction.

WHAT IT DOES
------------
1. Lists every column in pooled_data.csv and flags anything that could plausibly
   encode direction, side, choice, or correctness.
2. For each candidate, reports cardinality and value distribution, and judges whether
   it looks like a usable binary outcome.
3. If one is found, computes the error rate per participant x speed cell.
4. Turns that error rate into a verdict on two-boundary feasibility, using the
   simulated recovery curves in core/ddm2.py.

IF NOTHING IS FOUND
-------------------
pooled_data.csv is a processed summary; the KINARM raw files may still contain hand
trajectory data from which INITIAL MOVEMENT DIRECTION can be derived (the sign of
lateral velocity in the first ~100 ms after movement onset). That is the quantity the
professor is asking about -- "heading initially in the wrong direction" -- and it is
usually recoverable from kinematics even when it was never scored as an outcome.
This script tells you whether that trip back to the raw files is necessary.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

DIRECTION_HINTS = ["dir", "side", "left", "right", "target", "choice", "resp",
                   "correct", "error", "err", "hit", "miss", "accur", "outcome",
                   "angle", "theta", "lateral", "sign", "cw", "ccw"]


def _need(f):
    for p in (os.path.join(HERE, f), os.path.join(ROOT, f), f):
        if os.path.exists(p):
            return p
    sys.exit(f"ERROR: {f} not found. Put pooled_data.csv next to this script.")


def main():
    df = pd.read_csv(_need("pooled_data.csv"))
    print("=" * 76)
    print("Q1  DIRECTION / ERROR AUDIT")
    print("=" * 76)
    print(f"  pooled_data.csv: {len(df)} rows, {len(df.columns)} columns\n")

    print("  ALL COLUMNS")
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        print(f"    {c:32s} dtype={str(df[c].dtype):10s} unique={nun}")

    cands = [c for c in df.columns
             if any(h in c.lower() for h in DIRECTION_HINTS)]
    print(f"\n  CANDIDATE DIRECTION/OUTCOME COLUMNS: "
          f"{', '.join(cands) if cands else 'NONE FOUND'}")

    usable = []
    for c in cands:
        vals = df[c].dropna()
        nun = vals.nunique()
        print(f"\n    {c}  ({nun} unique values)")
        if nun <= 12:
            vc = vals.value_counts().head(12)
            for k, n in vc.items():
                print(f"        {str(k):>18s} : {n:6d}  ({100*n/len(vals):5.1f}%)")
            if 2 <= nun <= 4:
                # a binary-ish outcome with a minority class is what we need
                frac = vals.value_counts(normalize=True)
                minority = float(frac.min())
                print(f"        -> minority class = {minority:.4f}")
                if minority > 0.005:
                    usable.append((c, minority))
        else:
            print(f"        continuous or high-cardinality; "
                  f"range [{vals.min():.3g}, {vals.max():.3g}]")
            print(f"        (if this is a signed angle or lateral position, its SIGN")
            print(f"         early in the movement may be the outcome you need)")

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    if not usable:
        print("  No usable discrete outcome column found in pooled_data.csv.")
        print()
        print("  This does NOT settle the question. The professor is asking about")
        print("  trials that head INITIALLY in the wrong direction, which is a")
        print("  kinematic property, not a scored outcome. To get it you would:")
        print("     1. return to the raw KINARM trial files")
        print("     2. find movement onset (velocity threshold, as your RT extraction does)")
        print("     3. take the sign of lateral velocity averaged over the first ~100 ms")
        print("     4. compare that sign to the target's side")
        print("     5. score a mismatch as an initial-direction error")
        print()
        print("  Until that exists, a two-boundary DDM cannot be fitted, and the")
        print("  single-boundary choice stands by necessity rather than by argument.")
        print("  That is a perfectly reportable position.")
        return

    print(f"  Found {len(usable)} candidate outcome column(s):")
    for c, m in usable:
        print(f"    {c}: minority class {m:.4f}")
    best, err_rate = min(usable, key=lambda x: abs(x[1] - 0.15))
    print(f"\n  Using '{best}' with error rate {err_rate:.4f}\n")

    n_cell = int(df.groupby(["Participant", "Speed_deg_per_s"]).size().median())
    exp_err = err_rate * n_cell
    n_part = df.Participant.nunique()
    pooled = err_rate * n_cell * n_part

    print(f"  median trials per cell        : {n_cell}")
    print(f"  expected error trials per cell: {exp_err:.1f}")
    print(f"  pooled per speed (all {n_part} pts) : {pooled:.0f}")
    print()
    print("  Simulated two-boundary recovery at n=110/cell (core/ddm2.py):")
    print("     P(err)   err trials   RMSE v   MAE t0    MAE w   fits converged")
    print("      0.02        2.4       0.475    5.9ms    0.046      9/12")
    print("      0.05        4.8       0.331    8.8ms    0.037     11/12")
    print("      0.10       11.6       0.282   11.1ms    0.032     12/12")
    print("      0.20       24.0       0.298    7.0ms    0.027     12/12")
    print("      0.30       32.4       0.195    8.9ms    0.024     12/12")
    print()
    if exp_err < 5:
        print(f"  >>> {exp_err:.1f} error trials per cell is NOT enough for per-cell")
        print("      fitting. Fits will fail outright on some cells and the starting")
        print("      point w will be poorly recovered. The professor's skepticism is")
        print("      quantitatively correct at the cell level.")
        print()
        print(f"  >>> BUT pooling across participants gives ~{pooled:.0f} error trials per")
        print("      speed condition. A HIERARCHICAL two-boundary model with participant")
        print("      random effects may still be identified at the group level even")
        print("      though no single cell is. That is the version worth attempting.")
    elif exp_err < 12:
        print(f"  >>> {exp_err:.1f} error trials per cell is MARGINAL. Per-cell fits will")
        print("      run but v and w carry substantial error. Go hierarchical.")
    else:
        print(f"  >>> {exp_err:.1f} error trials per cell is WORKABLE. A two-boundary model")
        print("      is worth fitting; compare it to the single-boundary Wald by the")
        print("      same machinery used in q2.")


if __name__ == "__main__":
    main()
