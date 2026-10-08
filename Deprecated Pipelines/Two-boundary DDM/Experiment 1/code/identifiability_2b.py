"""
identifiability_2b.py  --  where does the two-boundary DDM stop being identifiable?

Same job as SRT_identifiability_check.py, one model up. Simulates from known
(v, a, w, t0), refits by maximum likelihood, and reports recovery as a function of
trials per cell and minority-response share. The point is to have a quantitative
answer to "can we fit a two-boundary model to these cells?" before spending hours
in NUTS -- and to be able to show the professor exactly where the wall is rather
than asserting it.

The single-boundary shifted Wald cannot express a start point at all, so w is the
parameter the two-boundary model is really being asked to buy. Watch its recovery
column: when minority responses run out, w reverts to the prior/starting value and
a trades off against v.

Runtime note: the cells with very FEW minority responses are the slow ones, because
the likelihood surface is nearly flat in a and w and differential evolution runs to
its iteration cap every time instead of converging. That slowness is itself the
result -- it is the optimiser reporting that the parameters are not identified.
Budget roughly 1-2 s per fit per core; the sweep is reps x 18 fits.

Run: python identifiability_2b.py                 (default sweep)
     python identifiability_2b.py --reps 40       (tighter, slower)
     python identifiability_2b.py --jobs 8        (cores for the optimiser)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from wfpt import numpy_wfpt_logpdf_vec, sample_ddm, numpy_p_upper

V_MAX, A_MAX = 20.0, 5.0        # a is a SEPARATION here, so the cap is above the
                                # single-boundary A_MAX = 2.5 in DDM_fit.py


def fit_mle(rt, choice, floor, seed=42, maxiter=150, workers=1):
    """Per-cell MLE of (v, a, w, t0) by differential evolution -- Method A, two-boundary."""
    hi_t0 = max(np.percentile(rt, 3) - 0.002, floor + 1e-3)

    def nll(p):
        v, a, w, t0 = p
        tau = rt - t0
        if np.any(tau <= 0):
            return 1e10
        lp = numpy_wfpt_logpdf_vec(tau, v, a, w, choice)
        if not np.all(np.isfinite(lp)):
            return 1e10
        return -float(np.sum(lp))

    bounds = [(-V_MAX, V_MAX), (0.05, A_MAX), (0.02, 0.98), (floor, hi_t0)]
    best = None
    for s in (seed, seed + 5):
        r = differential_evolution(nll, bounds, seed=s, maxiter=maxiter, tol=1e-8,
                                   popsize=15, polish=True, workers=workers,
                                   updating="deferred" if workers != 1 else "immediate")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--maxiter", type=int, default=150,
                    help="differential-evolution iteration cap")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="cores for differential evolution (-1 = all, 1 = serial)")
    ap.add_argument("--floor", type=float, default=0.130)
    ap.add_argument("--out", default="identifiability_2b.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    A_TRUE, W_TRUE, T0_TRUE = 1.5, 0.58, 0.165

    # drifts chosen to span minority shares from ~50% down to ~0%
    drifts = [0.0, 1.0, 2.0, 3.0, 4.5, 6.5]
    sizes = [50, 120, 300]

    rows = []
    print(f"{'n':>5}{'true v':>9}{'minority %':>12}{'v err':>9}{'a err':>9}"
          f"{'w err':>9}{'t0 err ms':>11}")
    print("-" * 64)
    for n in sizes:
        for v_true in drifts:
            errs = []
            shares = []
            for r in range(args.reps):
                rt, ch = sample_ddm(n, v_true, A_TRUE, W_TRUE, T0_TRUE, rng=rng)
                ok = np.isfinite(rt)
                rt, ch = rt[ok], ch[ok]
                if len(rt) < 15:
                    continue
                n_up = ch.sum()
                shares.append(min(n_up, len(ch) - n_up) / len(ch))
                est = fit_mle(rt, ch.astype(float), args.floor, seed=100 + r,
                              maxiter=args.maxiter, workers=args.jobs)
                errs.append([est[0] - v_true, est[1] - A_TRUE, est[2] - W_TRUE,
                             (est[3] - T0_TRUE) * 1000])
            if not errs:
                continue
            e = np.array(errs)
            row = dict(n=n, v_true=v_true,
                       minority_share=float(np.mean(shares)),
                       p_upper_analytic=numpy_p_upper(v_true, A_TRUE, W_TRUE),
                       v_bias=float(e[:, 0].mean()), v_rmse=float(np.sqrt((e[:, 0] ** 2).mean())),
                       a_bias=float(e[:, 1].mean()), a_rmse=float(np.sqrt((e[:, 1] ** 2).mean())),
                       w_bias=float(e[:, 2].mean()), w_rmse=float(np.sqrt((e[:, 2] ** 2).mean())),
                       t0_bias_ms=float(e[:, 3].mean()),
                       t0_rmse_ms=float(np.sqrt((e[:, 3] ** 2).mean())),
                       reps=len(errs))
            rows.append(row)
            print(f"{n:>5}{v_true:>9.1f}{100 * row['minority_share']:>11.1f}%"
                  f"{row['v_rmse']:>9.2f}{row['a_rmse']:>9.2f}{row['w_rmse']:>9.3f}"
                  f"{row['t0_rmse_ms']:>11.1f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 64)
    good = out[out.w_rmse < 0.08]
    if len(good):
        print(f"start point w recovers to within 0.08 when the minority share is at least "
              f"{100 * good.minority_share.min():.1f}%")
    bad = out[out.w_rmse >= 0.15]
    if len(bad):
        print(f"w is effectively unrecoverable (RMSE >= 0.15) below "
              f"{100 * bad.minority_share.max():.1f}% minority responses")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
