#!/usr/bin/env python
"""
run_all.py -- run the next-steps analyses in order.

    python run_all.py                      # full run
    python run_all.py --quick              # B=50, fewer power reps

SETUP
-----
Copy pooled_data.csv into analyses/ (or this folder). Nothing else is required.
Only numpy, scipy, pandas, matplotlib. No PyMC.

ORDER MATTERS
-------------
  q1  direction audit          -> is a two-boundary model even possible?
  q2  LATER vs Wald            -> the professor's directive
  q3  two-boundary feasibility -> uses q1's error rate
  q4  figures                  -> uses q2's output

q1 is a gate. If it finds no directional outcome, q3 has nothing to work with and
the two-boundary question is answered by absence of data, not by modelling.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "analyses")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--error-rate", type=float, default=None,
                    help="pass through to q3; take it from q1's output")
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--participants", type=int, default=16)
    args = ap.parse_args()

    B = "50" if args.quick else "200"
    reps = "8" if args.quick else "25"

    data = None
    for c in (os.path.join(AN, "pooled_data.csv"), os.path.join(HERE, "pooled_data.csv")):
        if os.path.exists(c):
            data = c
            break
    if data is None:
        sys.exit("ERROR: pooled_data.csv not found. Put it in analyses/.")
    print(f"using {data}\n")

    steps = [
        ("q1_direction_audit.py", [],
         "Is there a directional outcome? Gate for the two-boundary question."),
        ("q2_later_vs_wald.py", ["--B", B],
         "LATER vs Wald on both effectors, on a fair footing."),
    ]
    if args.error_rate is not None:
        steps.append(("q3_twoboundary_feasibility.py",
                      ["--error-rate", str(args.error_rate), "--n", str(args.n),
                       "--participants", str(args.participants), "--reps", reps],
                      "Can a two-boundary DDM be identified at your error rate?"))
    steps.append(("q4_figures.py", [], "Diagnostics for the presentation."))

    results = []
    for script, extra, desc in steps:
        print("\n" + "#" * 76)
        print(f"# {script}   --   {desc}")
        print("#" * 76)
        t = time.perf_counter()
        r = subprocess.run([sys.executable, os.path.join(AN, script)] + extra, cwd=AN)
        results.append((script, r.returncode, time.perf_counter() - t))
        if r.returncode != 0:
            print(f"\n!! {script} exited {r.returncode}. Stopping.")
            break

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    for s, code, dt in results:
        print(f"  {'OK ' if code == 0 else 'ERR'}  {s:36s} {dt:7.1f}s")
    if args.error_rate is None:
        print("\n  q3 was skipped. Read q1's error rate, then re-run with")
        print("  --error-rate <value> to get the two-boundary feasibility verdict.")
    print("\n  Outputs are in analyses/. Read PROFESSOR_MEMO.md next.")


if __name__ == "__main__":
    main()
