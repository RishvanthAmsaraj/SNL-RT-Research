"""
Bayesian_HRT_fit.py  --  Hierarchical Bayesian model: hand RT (HRT)
====================================================================
Method B (Bayesian) for hand RT. Fits the single-boundary diffusion model
(shifted Wald, s = 1) as a hierarchical Bayesian model with partial pooling across
participants and weakly-informative, literature-centered priors. This resolves the
non-decision-time degeneracy on principled grounds (no hard drift cap needed) and
returns full posterior credible intervals.

Run:  python Bayesian_HRT_fit.py
Outputs: Bayesian_hrt_fits.csv (pid, spd, v, a, t0, t0_lo95, t0_hi95, conv_div, conv_rhat)
         Bayesian_hrt_ndt.csv  (focused NDT view: pid, spd, t0_ms, 95% CI, ci_width_ms, floored, diagnostics)
        Bayesian_hrt_posterior.nc  (full posterior, for audit)

MODEL (one "unit" = participant x speed; partial pooling across units):
  log f(tau) = log a - 0.5*log(2*pi) - 1.5*log(tau) - (a - v*tau)^2/(2 tau),  tau = RT - t0
  log v ~ Normal(mu_lv, s_lv);  mu_lv ~ Normal(log 10, 0.5);  s_lv ~ HalfNormal(0.5)
  log a ~ Normal(mu_la, s_la);  mu_la ~ Normal(log 1.0, 0.5); s_la ~ HalfNormal(0.5)
  t0 = FLOOR + (min_RT - FLOOR)*sigmoid(z);  z hierarchical;  FLOOR = 100 ms
       (FLOOR is a permissive manual non-decision floor on the fitted t0 PARAMETER;
        sub-100 ms responses are treated as anticipations / fast guesses
        [Whelan 2008; Luce 1986]; 100 ms is a data-cleaning threshold justified by
        summed irreducible delays, NOT a measured physiological constant.
        This is a model parameter floor; the raw-RT data cutoff of 150 ms is a
        separate, stricter anticipation filter applied before fitting -- see
        METHODOLOGICAL_JUSTIFICATION.md §5.3.
        A non-centered parametrization is used for sampling efficiency.)

Priors are weakly informative (data dominate). Boundary centered at a~1 matches the
rescaled Ratcliff & Tuerlinckx (2002) range under s=1 (their 0.08-0.16 at s=0.1).

CITATIONS: Wiecki, Sofer & Frank (2013) Front. Neuroinform. 7:14 (HDDM precedent);
  Ratcliff & Tuerlinckx (2002) Psychon. Bull. Rev. 9:438-481; Gelman et al., Bayesian
  Data Analysis (non-centered parametrization, partial pooling, r-hat). R-hat is
  attributed primarily to Gelman & Rubin (1992) Statist. Sci. 7:457-472; BDA3 is the
  textbook treatment. If the rank-normalized split R-hat is used, also cite
  Vehtari et al. (2021) Bayesian Analysis 16:667-718.
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try:
    import pymc as pm, pytensor.tensor as pt, arviz as az
except ModuleNotFoundError:
    sys.exit(
        "\n" + "=" * 72 + "\n"
        "  This Bayesian script needs PyMC, which is not installed.\n"
        + "=" * 72 + "\n"
        "PyMC's sampler uses a compiled backend, so on Windows the reliable way to\n"
        "install it is with conda (plain `pip install pymc` typically fails on the\n"
        "Windows Store Python because it has no C/C++ compiler). One-time setup:\n\n"
        "  1. Install Miniconda (no admin rights needed):\n"
        "       https://docs.conda.io/en/latest/miniconda.html\n"
        "  2. Open 'Anaconda Prompt' and run:\n"
        "       conda create -n snl python=3.11\n"
        "       conda activate snl\n"
        "       conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib\n"
        "  3. Run this script from that environment:\n"
        "       conda activate snl\n"
        "       python \"" + os.path.basename(__file__) + "\"\n\n"
        "You are NOT blocked in the meantime: the validated outputs of this script\n"
        "(the .csv fit tables and the .pdf/.png figures) are already saved in this\n"
        "folder, so the analysis is complete even without re-running it.\n"
        + "=" * 72 + "\n")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "pooled_data.csv")
OUT  = os.path.join(SCRIPT_DIR, "Bayesian_hrt_fits.csv")
SPEEDS = [0, 75, 150]
LO, HI = 150, 800            # HRT physiological filter (ms)
FLOOR = 0.100                # manual non-decision floor (100 ms)
DRAWS, TUNE, CHAINS, TARGET = 1500, 1500, 4, 0.95

def main():
    dfi = pd.read_csv(DATA); dfi = dfi[dfi["BlockType"] == "I"]
    units, rt, uidx, minrt = [], [], [], []; u = 0
    for spd in SPEEDS:
        for pid in sorted(dfi["Participant"].unique()):
            sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
            x = sub["HandRT_ms"].values.astype(float)
            x = x[(~np.isnan(x)) & (x >= LO) & (x <= HI)] / 1000.0
            if len(x) < 15: continue
            units.append((pid, spd)); minrt.append(x.min()); rt.extend(x.tolist()); uidx.extend([u]*len(x)); u += 1
    rt = np.array(rt); uidx = np.array(uidx); minrt = np.array(minrt)
    print(f"HRT: {len(rt)} trials across {len(units)} participant x speed units", flush=True)

    with pm.Model() as m:
        mu_lv = pm.Normal("mu_lv", np.log(10), 0.5); s_lv = pm.HalfNormal("s_lv", 0.5)
        mu_la = pm.Normal("mu_la", np.log(1.0), 0.5); s_la = pm.HalfNormal("s_la", 0.5)
        mu_z  = pm.Normal("mu_z", 0.5, 1.0);          s_z  = pm.HalfNormal("s_z", 1.0)
        lv = mu_lv + s_lv*pm.Normal("lv_raw", 0, 1, shape=len(units))
        la = mu_la + s_la*pm.Normal("la_raw", 0, 1, shape=len(units))
        z  = mu_z  + s_z *pm.Normal("z_raw",  0, 1, shape=len(units))
        v  = pm.Deterministic("v", pt.exp(lv))
        a  = pm.Deterministic("a", pt.exp(la))
        t0 = pm.Deterministic("t0", FLOOR + (minrt - FLOOR)*pm.math.sigmoid(z))
        tau = rt - t0[uidx]
        logp = pt.log(a[uidx]) - 0.5*np.log(2*np.pi) - 1.5*pt.log(tau) - (a[uidx]-v[uidx]*tau)**2/(2*tau)
        pm.Potential("lik", logp.sum())
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=TARGET, random_seed=7, progressbar=False)

    div = int(idata.sample_stats["diverging"].sum())
    rh = az.rhat(idata, var_names=["v","a","t0"])
    rmax = float(max(rh["v"].max(), rh["a"].max(), rh["t0"].max()))
    po = idata.posterior
    res = pd.DataFrame({
        "pid":[u[0] for u in units], "spd":[u[1] for u in units],
        "v":  np.round(po["v"].mean(("chain","draw")).values, 3),
        "a":  np.round(po["a"].mean(("chain","draw")).values, 4),
        "t0": np.round(po["t0"].mean(("chain","draw")).values*1000),
        "t0_lo95": np.round(po["t0"].quantile(0.025,("chain","draw")).values*1000),
        "t0_hi95": np.round(po["t0"].quantile(0.975,("chain","draw")).values*1000),
        "conv_div": div, "conv_rhat": round(rmax,3)})
    res.to_csv(OUT, index=False)
    # focused non-decision-time table (parallels Bayesian_srt_ndt.csv): t0 + 95% CI + floored flag
    ndt = res[["pid", "spd"]].copy()
    ndt["t0_ms"] = res.t0.astype(int)
    ndt["t0_lo95"] = res.t0_lo95.astype(int)
    ndt["t0_hi95"] = res.t0_hi95.astype(int)
    ndt["ci_width_ms"] = (res.t0_hi95 - res.t0_lo95).astype(int)
    ndt["floored"] = (np.abs(res.t0 - FLOOR * 1000) < 2).astype(int)   # 1 = pinned at the 100 ms floor
    ndt["conv_div"] = res.conv_div; ndt["conv_rhat"] = res.conv_rhat
    ndt.sort_values(["spd", "pid"]).to_csv(os.path.join(SCRIPT_DIR, "Bayesian_hrt_ndt.csv"), index=False)
    try: idata.to_netcdf(os.path.join(SCRIPT_DIR, "Bayesian_hrt_posterior.nc"))
    except Exception: pass
    print(f"divergences={div}  max r-hat={rmax:.3f}", flush=True)
    print(f"t0 at {int(FLOOR*1000)}ms floor: {int((np.abs(res.t0-FLOOR*1000)<2).sum())}/{len(res)}", flush=True)
    print("BAYESIAN_HRT_DONE", flush=True)

if __name__ == "__main__":
    main()
