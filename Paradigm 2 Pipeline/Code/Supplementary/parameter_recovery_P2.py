"""
parameter_recovery_P2.py  --  Paradigm 2 (supplementary): can t0 be recovered at these trial counts?

Simulate cells from known shifted-Wald parameters at the Paradigm 2 cell size (n = 158, the
median), apply the production RT window, refit with the production fitter (DDM_fit.fit_single:
differential evolution, 5% contamination, the same floors), and compare fitted t0 to the truth.

  HAND : truth = per-speed mean of the Method B estimates (Bayesian_hrt_fits.csv).
  EYE-A: truth t0 = 90 ms (ABOVE the 70 ms floor), with saccade-shaped (v, a) = per-speed mean of
         the fixed-t0 fits (SRT_fixedt0_fits.csv). Tests whether a saccadic t0 above the floor
         would be recovered -- i.e. whether flooring in the real data is a power/fitter limit.
  EYE-B: truth t0 = 44 ms (the pooled shape-implied saccadic value from why_saccadic_t0_floors,
         BELOW the floor). Shows what the fitter returns when the truth is below the floor.
Outputs: parameter_recovery_P2.csv (every replicate), parameter_recovery_P2_summary.csv
Run    : python parameter_recovery_P2.py   (after Bayesian_HRT_fit.py and SRT_fixed_t0_analysis.py)
"""
import os, sys, numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from DDM_fit import fit_single, HRT_FLOOR, SRT_FLOOR, P_CONTAM
SPEEDS = [75, 100, 125, 150]; N_TRIALS, N_REP, SEED = 158, 40, 2026

def simulate(rng, v, a, t0, n, lo, hi):
    out = np.empty(0)
    while len(out) < n:
        x = t0 + stats.invgauss(mu=1.0 / (v * a), scale=a ** 2).rvs(size=2 * n, random_state=rng)
        out = np.concatenate([out, x[(x >= lo) & (x <= hi)]])
    return out[:n]

def main():
    rng = np.random.default_rng(SEED)
    bh = pd.read_csv(os.path.join(HERE, "Bayesian_hrt_fits.csv")).groupby("spd")[["v", "a", "t0"]].mean()
    fx = pd.read_csv(os.path.join(HERE, "SRT_fixedt0_fits.csv")).groupby("spd")[["v", "a"]].mean()
    scen = []
    for s in SPEEDS:
        scen.append(("HAND", s, bh.loc[s, "v"], bh.loc[s, "a"], bh.loc[s, "t0"] / 1000, HRT_FLOOR, 0.150, 0.800))
        scen.append(("EYE-A_t0_90", s, fx.loc[s, "v"], fx.loc[s, "a"], 0.090, SRT_FLOOR, 0.080, 0.600))
        scen.append(("EYE-B_t0_44", s, fx.loc[s, "v"], fx.loc[s, "a"], 0.044, SRT_FLOOR, 0.080, 0.600))
    rows = []
    for name, s, v, a, t0, floor, lo, hi in scen:
        for rep in range(N_REP):
            x = simulate(rng, v, a, t0, N_TRIALS, lo, hi)
            est, _, ks = fit_single(x, floor, P_CONTAM)
            rows.append(dict(scenario=name, spd=s, rep=rep, true_v=v, true_a=a, true_t0_ms=t0 * 1000, floor_ms=floor * 1000,
                             fit_v=est[0], fit_a=est[1], fit_t0_ms=est[2] * 1000, ks=ks))
    r = pd.DataFrame(rows); r.round(4).to_csv(os.path.join(HERE, "parameter_recovery_P2.csv"), index=False)
    r["err"] = r.fit_t0_ms - r.true_t0_ms; r["at_floor"] = (r.fit_t0_ms - r.floor_ms) < 1.0
    sm = r.groupby(["scenario", "spd"]).agg(true_t0_ms=("true_t0_ms", "first"), mean_fit_t0_ms=("fit_t0_ms", "mean"),
                                           bias_ms=("err", "mean"), rmse_ms=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                           sd_ms=("fit_t0_ms", "std"), pct_at_floor=("at_floor", lambda b: 100 * b.mean())).round(1)
    sm.to_csv(os.path.join(HERE, "parameter_recovery_P2_summary.csv")); print(sm.to_string()); print("saved parameter_recovery_P2*.csv")

if __name__ == "__main__":
    main()
