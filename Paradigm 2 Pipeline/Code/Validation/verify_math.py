"""verify_math.py -- independent checks of the core math and tables (Paradigm 2): the Wald density against scipy,
Method A optima against an independent multi-start search, Bayesian table consistency, and a plug-in posterior-predictive
check. Optional: --raw <folder with the 16 CIR files> also recounts trials from the raw files.
Run: python verify_math.py [--raw <folder>]"""
import os, numpy as np, pandas as pd, glob, sys, warnings
from scipy import stats
from scipy.optimize import minimize
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DDM"))
from DDM_fit import wald_pdf, fit_single
import os
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..")
RAW = sys.argv[sys.argv.index("--raw") + 1] if "--raw" in sys.argv else None
# 1) Wald density used by Method A == scipy inverse Gaussian with mean a/v and shape a^2
t = np.linspace(0.005, 0.6, 400); worst = 0
for v, a in [(5, .6), (10.5, .9), (13, 1.2), (8, .8)]:
    worst = max(worst, np.max(np.abs(wald_pdf(t, v, a) - stats.invgauss(mu=1 / (v * a), scale=a ** 2).pdf(t)) / stats.invgauss(mu=1 / (v * a), scale=a ** 2).pdf(t).max()))
    m = stats.invgauss(mu=1 / (v * a), scale=a ** 2).mean(); assert abs(m - a / v) < 1e-12
print(f"[1] Method A Wald pdf vs scipy invgauss: max relative difference {worst:.2e}; mean = a/v confirmed")
# 2) raw-data counts recomputed straight from the 16 CIR files (independent of the pooled builder)
if RAW:
  raw = pd.concat([pd.read_csv(f, usecols=["ParticipantID", "BlockType", "HandRT_ms", "GazeSRT_ms"]) for f in sorted(glob.glob(os.path.join(RAW, "CIR*_TRIAL_Summary_v0_1_36.csv")))])
  print(f"[2] from raw files: trials {len(raw)}, hand kept {int(raw.HandRT_ms.between(150, 800).sum())}, saccade kept {int(raw.GazeSRT_ms.between(80, 600).sum())}, participants {raw.ParticipantID.nunique()}")
# 3) independent Method A check: NLL written from scipy, optimised with Nelder-Mead from many starts; pipeline optimum must be at least as good
d = pd.read_csv(os.path.join(C, "pooled_data_P2.csv"), usecols=["Participant", "Speed_deg_per_s", "HandRT_ms", "GazeSRT_ms"])
def nll_indep(p, x, floor, cap):
    v, a, t0 = p
    if not (0.1 <= v <= 20 and 0.05 <= a <= 2.5 and floor <= t0 <= cap) or np.any(x - t0 <= 0): return 1e10
    f = stats.invgauss(mu=1 / (v * a), scale=a ** 2).pdf(x - t0); return -np.sum(np.log(0.95 * f + 0.05 / (x.max() - x.min())))
rng = np.random.default_rng(3); worse, gaps = 0, []
for col, lo, hi, floor, tab in [("HandRT_ms", 150, 800, .130, "DDM_hrt_fits.csv"), ("GazeSRT_ms", 80, 600, .070, "DDM_srt_fits.csv")]:
    T = pd.read_csv(os.path.join(C, "DDM", tab)); T = T[T.model == "single"] if "model" in T else T
    for r in T.sample(10, random_state=1).itertuples():
        x = d[(d.Participant == r.pid) & (d.Speed_deg_per_s == r.spd)][col].values.astype(float); x = x[(x >= lo) & (x <= hi)] / 1000
        cap = max(np.percentile(x, 3) - .002, floor + 1e-3); best = None
        for _ in range(25):
            s0 = [rng.uniform(2, 18), rng.uniform(.3, 1.8), rng.uniform(floor, cap)]
            o = minimize(nll_indep, s0, args=(x, floor, cap), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
            best = o if best is None or o.fun < best.fun else best
        mine = nll_indep([r.v, r.a, r.t0 / 1000], x, floor, cap); gaps.append(mine - best.fun); worse += (mine - best.fun) > 0.05
print(f"[3] Method A optimum vs independent multi-start search (20 cells): pipeline NLL - best independent NLL: max {max(gaps):+.4f}, min {min(gaps):+.4f}; cells where the independent search found a clearly better optimum: {worse}")
# 4) internal consistency of the Bayesian tables
bh, bs, nd = [pd.read_csv(os.path.join(C, "Bayesian", f)) for f in ["Bayesian_hrt_fits.csv", "Bayesian_srt_fits.csv", "Bayesian_srt_ndt.csv"]]
mh = d[d.HandRT_ms.between(150, 800)].groupby(["Participant", "Speed_deg_per_s"]).HandRT_ms.min()
ms_ = d[d.GazeSRT_ms.between(80, 600)].groupby(["Participant", "Speed_deg_per_s"]).GazeSRT_ms.min()
okh = all((r.t0_lo95 <= r.t0 <= r.t0_hi95) and (r.t0_lo95 >= 129.5) and (r.t0_hi95 <= mh.loc[(r.pid, r.spd)] + 0.5) for r in bh.itertuples())
su = bs[bs.model == "single"]
oks = all((r.t0_lo95 <= r.t0 <= r.t0_hi95) and (r.t0_lo95 >= 69.5) and (r.t0_hi95 <= ms_.loc[(r.pid, r.spd)] + 0.5) for r in su.itertuples())
okn = all((r.t0_lo95 <= r.t0_ms <= r.t0_hi95) and r.t0_lo95 >= 69.5 and r.t0_hi95 <= r.min_srt_ms for r in nd.itertuples())
okm = bool(((bs[bs.model == "mixture"].express_mode < bs[bs.model == "mixture"].reg_mode)).all())
print(f"[4] Bayesian tables: hand intervals ordered & inside [floor, fastest RT]: {okh}; saccade single cells: {oks}; participant ndt: {okn}; mixture fast < slow: {okm}")
# 5) posterior predictive check (plug-in posterior means): simulated vs observed quantiles per cell
qs = [.1, .3, .5, .7, .9]; rng = np.random.default_rng(11)
for lab, tab, col, lo, hi in [("hand", bh, "HandRT_ms", 150, 800), ("saccade", su, "GazeSRT_ms", 80, 600)]:
    errs = []
    for r in tab.itertuples():
        x = d[(d.Participant == r.pid) & (d.Speed_deg_per_s == r.spd)][col].values.astype(float); x = x[(x >= lo) & (x <= hi)]
        y = r.t0 + 1000 * stats.invgauss(mu=1 / (r.v * r.a), scale=r.a ** 2).rvs(size=20000, random_state=rng); y = y[(y >= lo) & (y <= hi)]
        errs.append(np.quantile(y, qs) - np.quantile(x, qs))
    e = np.array(errs)
    print(f"[5] PPC {lab} ({len(e)} cells): mean |predicted - observed| at 10/30/50/70/90%: {np.round(np.abs(e).mean(0), 1)} ms; mean signed: {np.round(e.mean(0), 1)} ms")
