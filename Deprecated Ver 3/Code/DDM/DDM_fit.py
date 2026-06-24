"""
DDM_fit.py  --  Drift-Diffusion Model (frequentist / maximum-likelihood) fitting
=================================================================================
Method A of the KINARM interception RT analysis. Fits the single-boundary diffusion
model (shifted Wald, s = 1) by maximum likelihood to hand RT (HRT) and saccadic RT
(SRT), for every participant x target-speed cell, with literature-grounded bounds and
an automatic express/regular mixture for SRT where the data require it.

This single script reproduces both output tables:
    DDM_hrt_fits.csv   -- HRT, one row per cell
    DDM_srt_fits.csv   -- SRT, one row per cell (single OR express/regular mixture)

Run:  python DDM_fit.py        (resumable; re-running skips finished cells)

--------------------------------------------------------------------------------
MODEL.  Single-boundary diffusion = shifted Wald density (unit diffusion noise, s=1):
    f(t) = a / sqrt(2*pi*t^3) * exp( -(a - v*t)^2 / (2t) ),   t = RT - t0 > 0
  parameters: drift v, boundary a, non-decision time t0. A 5% uniform contamination
  term (Ratcliff & Tuerlinckx, 2002) adds robustness to stray trials.

BOUNDS (literature-grounded; s = 1 scaling convention, stated explicitly):
  a in [0.05, 2.5]   -- boundary separation is a policy/caution parameter, NOT a
                        physiological bound (Heitz & Schall 2012; Ratcliff & McKoon 2008).
                        Range follows the systematic DDM review of Tran et al. (2020): at
                        s=1 the empirical envelope is ~0.11-7.47 (practical mass 0.2-3), so
                        the 2.5 cap sits well inside it. Absolute a is scaling/model-
                        dependent (Ratcliff et al. 2016).
  v in [0.1, 20]     -- drift is a latent evidence-quality parameter, not a biological
                        speed limit (Ratcliff & McKoon 2008; Ratcliff et al. 2016). Cap 20
                        follows the Tran et al. (2020) s=1 envelope (|v| <~ 18.5). NB this
                        is a single-boundary Wald, whose v/a geometry differs from the
                        two-choice DDM that review pooled, so the envelope is a plausibility
                        bound, not an exact prior.
  t0 floor: HRT 130 ms -- reach-preparation floor (Haith et al. 2016: preparation ~130 ms);
                        a fitted hand non-decision time below this is hard to defend.
            SRT 70 ms  -- saccadic dead-time floor (Bompas et al. 2017; Ludwig et al. 2007).
  t0 ceiling: 3rd percentile of the cell's RTs - 2 ms.
  NB t0 is the model non-decision PARAMETER; in moving-target interception it can absorb
     decision-action overlap and urgency (Barany et al. 2020; Bompas et al. 2024; Weindel
     et al. 2020), so moving-target t0 should be interpreted cautiously, not as a pure
     sensorimotor delay.

SRT MIXTURE (express + regular saccades).  Saccadic distributions are often bimodal.
  Selection rule (fit-driven + structural validation; avoids BIC over-detection and
  dip-test under-detection):
    1. fit single Wald; only if it is inadequate (KS > 0.10) consider a mixture;
    2. adopt the 2-component mixture only if it (i) achieves KS < 0.10,
       (ii) gives two substantial components (0.10 <= pi <= 0.90), and
       (iii) recovers well-separated modes (>= 30 ms apart).
  Component number is judged on the CORE densities (no contamination), because the
  contamination term otherwise lets a single Wald absorb the express mode. The mixture
  is optimized robustly (Gaussian-mixture seeding + multi-start + differential-evolution
  backstop) so a genuine two-mode cell is never collapsed to one mode by a local optimum.
  Hartigan's dip-test p-value is reported per cell as corroboration.

CITATIONS: Tran et al. (2020) Front. Psychol. 11:608287 (v/a envelopes at s=1);
  Ratcliff et al. (2016) Trends Cogn. Sci. 20:260-281 (scaling & interpretation cautions);
  Ratcliff & McKoon (2008) Neural Comput. 20:873-922 (parameter definitions);
  Ratcliff & Tuerlinckx (2002) Psychon. Bull. Rev. 9:438-481 (contamination handling);
  Anders, Alario & Van Maanen (2016) Psychol. Methods 21:309-327 (shifted Wald);
  Haith et al. (2016) J. Neurosci. 36:3007-3015 (hand non-decision floor ~130 ms);
  Bompas et al. (2017) Cogn. Psychol. 94:26-52 & Ludwig et al. (2007) J. Neurophysiol.
  97:795-805 (saccadic dead time); Barany et al. (2020) J. Neurophysiol. (interception
  caution); Knox & Wolohan (2015) PLoS ONE 10:e0120437; Hartigan & Hartigan (1985)
  Ann. Statist. 13:70-84.
"""
import os, numpy as np, pandas as pd, warnings
from scipy import stats
from scipy.optimize import differential_evolution, minimize
warnings.filterwarnings("ignore")
try:
    from sklearn.mixture import GaussianMixture
    import diptest
    _HAVE_EXTRAS = True
except Exception:
    _HAVE_EXTRAS = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "pooled_data.csv")
SPEEDS = [0, 75, 150]
A_MAX, V_MAX = 2.5, 20.0          # a,v caps follow Tran et al. (2020) s=1 envelope (a<=7.47, |v|<=18.5)
HRT_FLOOR, SRT_FLOOR = 0.130, 0.070  # HRT floor = reach-prep ~130ms (Haith 2016); SRT floor = saccadic dead time (Bompas 2017; Ludwig 2007)
P_CONTAM = 0.05

# ---------------------------------------------------------------- densities
def wald_pdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return (a / np.sqrt(2*np.pi*t**3)) * np.exp(-(a - v*t)**2 / (2*t))
def wald_cdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return stats.invgauss.cdf(t, mu=1/(v*a), scale=a**2)

# ---------------------------------------------------------------- single Wald
def fit_single(rts, floor, contam):
    Tr = rts.max() - rts.min()
    def nll(p):
        v, a, t0 = p; adj = rts - t0
        if np.any(adj <= 0): return 1e10
        w = wald_pdf(adj, v, a)
        if np.any(w <= 0) or not np.all(np.isfinite(w)): return 1e10
        d = (1-contam)*w + (contam/Tr if contam > 0 else 0.0)
        if np.any(d <= 0): return 1e10
        return -np.sum(np.log(d))
    b = [(0.1, V_MAX), (0.05, A_MAX), (floor, max(np.percentile(rts,3)-0.002, floor+1e-3))]
    best = None
    for s in [42, 7]:
        r = differential_evolution(nll, b, seed=s, maxiter=400, tol=1e-9, popsize=12, polish=True)
        if best is None or r.fun < best.fun: best = r
    x = best.x; adj = rts - x[2]
    return x, best.fun, stats.kstest(adj, lambda z: wald_cdf(z, x[0], x[1])).statistic

# ---------------------------------------------------------------- 2-component mixture (robust)
def _mix_nll(p, rts, Tr, contam):
    pi, ve, ae, t0e, vr, ar, t0r = p
    ee = rts - t0e; rr = rts - t0r
    if np.any(ee <= 0) or np.any(rr <= 0): return 1e10
    if (t0e + ae/ve) >= (t0r + ar/vr): return 1e10            # express mean < regular mean
    core = pi*wald_pdf(ee, ve, ae) + (1-pi)*wald_pdf(rr, vr, ar)
    d = (1-contam)*core + (contam/Tr if contam > 0 else 0.0)
    if np.any(d <= 0) or not np.all(np.isfinite(d)): return 1e10
    return -np.sum(np.log(d))

def _moments_to_wald(mean, sd, floor):
    t0 = min(max(floor, mean - 2.5*sd), mean - 1e-3)
    mu = mean - t0; var = max(sd**2, 1e-6)
    lam = mu**3/var; a = np.sqrt(lam); v = a/mu
    return [np.clip(v, 0.1, V_MAX), np.clip(a, 0.05, A_MAX), np.clip(t0, floor, mean-1e-3)]

def fit_mixture(rts, floor, contam):
    Tr = rts.max() - rts.min(); mn = rts.min()
    cands = []
    if _HAVE_EXTRAS:
        try:
            gm = GaussianMixture(2, n_init=8, random_state=0).fit(rts.reshape(-1,1))
            o = np.argsort(gm.means_.ravel())
            m1, m2 = gm.means_.ravel()[o]; s1, s2 = np.sqrt(gm.covariances_.ravel()[o]); w1 = gm.weights_[o][0]
            cands.append([np.clip(w1,0.05,0.95)] + _moments_to_wald(m1,s1,floor) + _moments_to_wald(m2,s2,floor))
        except Exception: pass
    for pe, pr, piw in [(15,65,0.5),(25,70,0.4),(10,55,0.6),(30,75,0.5)]:
        me, mr = np.percentile(rts, pe), np.percentile(rts, pr)
        if mr - me < 0.005: continue
        cands.append([piw] + _moments_to_wald(me,(mr-me)/3,floor) + _moments_to_wald(mr,(rts.max()-mr)/3,floor))
    # t0r may be as high as min(rt) (both shifts only need to be < fastest RT)
    b = [(0.02,0.98),(0.1,V_MAX),(0.05,A_MAX),(floor, mn-1e-3),(0.1,V_MAX),(0.05,A_MAX),(floor, mn-1e-3)]
    best = None
    for c in cands:
        try:
            r = minimize(_mix_nll, c, args=(rts, Tr, contam), method='L-BFGS-B', bounds=b)
            if r.fun < 1e9 and (best is None or r.fun < best.fun): best = r
        except Exception: pass
    rde = differential_evolution(_mix_nll, b, args=(rts, Tr, contam), seed=42, maxiter=400, tol=1e-9, popsize=14, polish=True)
    if best is None or rde.fun < best.fun: best = rde
    x = best.x; pi = x[0]
    def mc(z): return pi*wald_cdf(z-x[3], x[1], x[2]) + (1-pi)*wald_cdf(z-x[6], x[4], x[5])
    return x, best.fun, stats.kstest(rts, mc).statistic

# ---------------------------------------------------------------- driver
def load_cell(dfi, pid, spd, col, lo, hi):
    sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
    x = sub[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0

def done(path):
    return set(zip(*[pd.read_csv(path)[c] for c in ["pid","spd"]])) if os.path.exists(path) else set()

def main():
    dfi = pd.read_csv(DATA); dfi = dfi[dfi["BlockType"] == "I"]
    pids = sorted(dfi["Participant"].unique())

    # ---- HRT: single Wald ----
    hpath = os.path.join(SCRIPT_DIR, "DDM_hrt_fits.csv"); hd = done(hpath); hh = not os.path.exists(hpath)
    for spd in SPEEDS:
        for pid in pids:
            if (pid, spd) in hd: continue
            rts = load_cell(dfi, pid, spd, "HandRT_ms", 150, 800)
            if len(rts) < 15: continue
            x, _, ks = fit_single(rts, HRT_FLOOR, P_CONTAM)
            pd.DataFrame([dict(pid=pid, spd=spd, n=len(rts), model="single",
                               v=round(x[0],3), a=round(x[1],4), t0=round(x[2]*1000), ks=round(ks,4))]
                         ).to_csv(hpath, mode="a", header=hh, index=False); hh = False
            print(f"HRT {pid} {spd}: t0={x[2]*1000:.0f}ms a={x[1]:.2f} ks={ks:.3f}", flush=True)

    # ---- SRT: single, or express/regular mixture where justified ----
    spath = os.path.join(SCRIPT_DIR, "DDM_srt_fits.csv"); sd = done(spath); sh = not os.path.exists(spath)
    for spd in SPEEDS:
        for pid in pids:
            if (pid, spd) in sd: continue
            rts = load_cell(dfi, pid, spd, "GazeSRT_ms", 80, 600)
            if len(rts) < 15: continue
            n = len(rts)
            dipp = diptest.diptest(rts*1000)[1] if _HAVE_EXTRAS else np.nan
            xs, _, ks_s = fit_single(rts, SRT_FLOOR, P_CONTAM)
            use_mix, ks_m, xm = False, np.nan, None
            if ks_s > 0.10:                                   # only intervene where single fails
                xm, _, ks_m = fit_mixture(rts, SRT_FLOOR, P_CONTAM); pi = xm[0]
                em = (xm[3]+xm[2]/xm[1])*1000; rm = (xm[6]+xm[5]/xm[4])*1000
                use_mix = (ks_m < 0.10) and (0.10 <= pi <= 0.90) and ((rm - em) >= 30)
            if use_mix:
                pi, ve, ae, t0e, vr, ar, t0r = xm
                row = dict(pid=pid, spd=spd, n=n, model="mixture", ks=round(ks_m,4), ks_single=round(ks_s,4),
                           dip_p=round(dipp,4) if dipp==dipp else np.nan,
                           pi=round(pi,3), ve=round(ve,3), ae=round(ae,4), t0e=round(t0e*1000),
                           vr=round(vr,3), ar=round(ar,4), t0r=round(t0r*1000),
                           express_mode=round((t0e+ae/ve)*1000), reg_mode=round((t0r+ar/vr)*1000),
                           v=np.nan, a=np.nan, t0=np.nan)
            else:
                row = dict(pid=pid, spd=spd, n=n, model="single", ks=round(ks_s,4), ks_single=round(ks_s,4),
                           dip_p=round(dipp,4) if dipp==dipp else np.nan,
                           pi=np.nan, ve=np.nan, ae=np.nan, t0e=np.nan, vr=np.nan, ar=np.nan, t0r=np.nan,
                           express_mode=np.nan, reg_mode=np.nan,
                           v=round(xs[0],3), a=round(xs[1],4), t0=round(xs[2]*1000))
            pd.DataFrame([row]).to_csv(spath, mode="a", header=sh, index=False); sh = False
            print(f"SRT {pid} {spd}: {row['model']} ks={row['ks']:.3f} (single {ks_s:.3f})", flush=True)
    print("DDM_FIT_DONE", flush=True)

if __name__ == "__main__":
    main()
