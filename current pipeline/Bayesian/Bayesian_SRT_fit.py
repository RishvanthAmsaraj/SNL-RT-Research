"""
Bayesian_SRT_fit.py  --  Hierarchical Bayesian model: saccadic RT (SRT)
=======================================================================================
Produces the Bayesian model fits for saccadic RT (SRT), at the highest accuracy,
mirroring the DDM (Method A) per-cell model selection so the two methods are comparable:

  * UNIMODAL participant x speed cells  -> hierarchical single-boundary Wald with partial
    pooling across participants WITHIN each speed. Pooling + literature-centered priors
    resolve the non-decision-time degeneracy (no hard drift cap needed).
  * EXPRESS / BIMODAL cells             -> Bayesian two-component (express + regular)
    shifted-Wald mixture, fit per cell, with post-hoc relabeling by component mean
    (the standard fix for mixture label-switching).

Which cells are which is read from the DDM selection (DDM_srt_fits.csv, the
`model` column). If that file is absent, the script falls back to Hartigan's dip test.

DESIGN FOR RELIABILITY (so it can run to completion despite long runtimes)
  * Per-speed hierarchical models (each ~10-13 units) sample far faster and cleaner than
    one pooled 48-unit model, and per-speed pooling is statistically cleaner (it avoids
    the non-independence of the same participant appearing at three speeds).
  * RESUMABLE: every completed cell is appended to the output CSV immediately. On restart
    the script skips cells already present, so no work is ever lost. A per-speed
    hierarchical model is atomic (fit, then all its cells are written together).
  * Thorough sampling settings (high target_accept, long tuning) for accuracy.

MODEL (identical likelihood to the DDM; s = 1 single-boundary Wald)
  shifted-Wald logpdf(tau; v, a) = log a - 0.5*log(2*pi) - 1.5*log(tau) - (a - v*tau)^2/(2 tau)
  t0 parametrized as a fraction of each cell's fastest RT (always valid; smooth for NUTS).

PRIORS (weakly informative; data dominate) -- see Bayesian_Methods.md
  log v ~ Normal(log 10, 0.5);  log a ~ Normal(log 1.0, 0.5)  (a centered ~1 matches the
  rescaled Ratcliff & Tuerlinckx 2002 range under s=1);  t0 fraction logit hierarchical.

USAGE
  python Bayesian_SRT_fit.py            # all speeds
  python Bayesian_SRT_fit.py 150        # one speed only (optional)

OUTPUT
  Bayesian_srt_fits.csv  -- one row per participant x speed:
     pid, spd, n, model, conv_div, conv_rhat,
     single: v, a, t0(ms), t0_lo95, t0_hi95
     mixture: pi, pi_lo95, pi_hi95, express_mode(ms), express_mode_lo95, express_mode_hi95,
              reg_mode(ms), reg_mode_lo95, reg_mode_hi95
  Bayesian_srt_posterior_<speed>_unimodal.nc  -- full posterior per speed (for audit)

Requires: pymc, arviz, pytensor, numpy, pandas (sklearn/diptest only for the fallback).
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
DATA  = os.path.join(SCRIPT_DIR, "pooled_data.csv")
SELECT = os.path.join(SCRIPT_DIR, "DDM_srt_fits.csv")   # which cells are mixture
OUT   = os.path.join(SCRIPT_DIR, "Bayesian_srt_fits.csv")
FLOOR = 0.040            # validity floor for the mixture shift (s)
FLOOR_PHYS = 0.070       # physiological saccadic non-decision floor (70 ms) for the single-Wald t0
LO, HI = 80, 600         # SRT physiological filter (ms)
SPEEDS = [int(sys.argv[1])] if len(sys.argv) > 1 else [0, 75, 150]

# thorough sampling settings (accuracy prioritized; runtime is acceptable)
DRAWS, TUNE, CHAINS, TARGET = 1500, 1500, 4, 0.95
MIX_DRAWS, MIX_TUNE, MIX_TARGET = 1500, 1500, 0.95

ALL_COLS = ["pid","spd","n","model","conv_div","conv_rhat",
            "v","a","t0","t0_lo95","t0_hi95",
            "pi","pi_lo95","pi_hi95",
            "express_mode","express_mode_lo95","express_mode_hi95",
            "reg_mode","reg_mode_lo95","reg_mode_hi95"]

def append_row(d):
    d = {c: d.get(c, np.nan) for c in ALL_COLS}
    head = not os.path.exists(OUT)
    pd.DataFrame([d])[ALL_COLS].to_csv(OUT, mode="a", header=head, index=False)

def done_cells():
    if not os.path.exists(OUT): return set()
    t = pd.read_csv(OUT); return set(zip(t.pid, t.spd))

def load_cell(dfi, pid, spd):
    sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
    x = sub["GazeSRT_ms"].values.astype(float)
    return x[(~np.isnan(x)) & (x >= LO) & (x <= HI)] / 1000.0

# ---------------------------------------------------------------- model definitions
def hierarchical_single(rt, uidx, minrt, n_units):
    with pm.Model() as m:
        mu_lv = pm.Normal("mu_lv", np.log(10), 0.5); s_lv = pm.HalfNormal("s_lv", 0.5)
        mu_la = pm.Normal("mu_la", np.log(1.0), 0.5); s_la = pm.HalfNormal("s_la", 0.5)
        mu_z  = pm.Normal("mu_z", 0.0, 1.0);          s_z  = pm.HalfNormal("s_z", 1.0)
        lv = mu_lv + s_lv*pm.Normal("lv_raw", 0, 1, shape=n_units)
        la = mu_la + s_la*pm.Normal("la_raw", 0, 1, shape=n_units)
        z  = mu_z  + s_z *pm.Normal("z_raw",  0, 1, shape=n_units)
        v  = pm.Deterministic("v", pt.exp(lv))
        a  = pm.Deterministic("a", pt.exp(la))
        # t0 in [FLOOR_PHYS, min_RT]: enforces the 70 ms physiological saccadic floor
        t0 = pm.Deterministic("t0", FLOOR_PHYS + (minrt - FLOOR_PHYS)*pm.math.sigmoid(z))
        tau = rt - t0[uidx]
        logp = pt.log(a[uidx]) - 0.5*np.log(2*np.pi) - 1.5*pt.log(tau) - (a[uidx]-v[uidx]*tau)**2/(2*tau)
        pm.Potential("lik", logp.sum())
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=TARGET, random_seed=7, progressbar=False)
    return idata

def bayesian_mixture(rts):
    minrt = rts.min()
    def wlog(tau, v, a):
        return pt.log(a) - 0.5*np.log(2*np.pi) - 1.5*pt.log(tau) - (a - v*tau)**2/(2*tau)
    with pm.Model() as m:
        pi = pm.Beta("pi", 2, 2)
        v1 = pm.LogNormal("v1", np.log(15), 0.6); a1 = pm.LogNormal("a1", np.log(1.0), 0.6)
        v2 = pm.LogNormal("v2", np.log(12), 0.6); a2 = pm.LogNormal("a2", np.log(1.2), 0.6)
        z1 = pm.Normal("z1", 0, 1); z2 = pm.Normal("z2", 0, 1)
        t01 = pm.Deterministic("t01", FLOOR + (minrt-FLOOR)*pm.math.sigmoid(z1)*0.98)
        t02 = pm.Deterministic("t02", FLOOR + (minrt-FLOOR)*pm.math.sigmoid(z2)*0.98)
        lp1 = pt.log(pi)   + wlog(rts - t01, v1, a1)
        lp2 = pt.log(1-pi) + wlog(rts - t02, v2, a2)
        pm.Potential("lik", pt.sum(pm.math.logsumexp(pt.stack([lp1, lp2], axis=0), axis=0)))
        idata = pm.sample(MIX_DRAWS, tune=MIX_TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=MIX_TARGET, random_seed=11, progressbar=False)
    return idata

# ---------------------------------------------------------------- main
def main():
    dfi = pd.read_csv(DATA); dfi = dfi[dfi["BlockType"] == "I"]
    pids = sorted(dfi["Participant"].unique())

    # determine mixture cells from the DDM selection (preferred) or dip test (fallback)
    mixture_cells = set()
    if os.path.exists(SELECT):
        sel = pd.read_csv(SELECT)
        mixture_cells = set(zip(sel[sel.model == "mixture"].pid, sel[sel.model == "mixture"].spd))
        print(f"Loaded {len(mixture_cells)} mixture cells from the DDM selection.", flush=True)
    else:
        import diptest
        for spd in SPEEDS:
            for pid in pids:
                x = load_cell(dfi, pid, spd)
                if len(x) >= 15 and diptest.diptest(x*1000)[1] < 0.05:
                    mixture_cells.add((pid, spd))
        print(f"DDM selection not found; dip test flagged {len(mixture_cells)} cells.", flush=True)

    done = done_cells()

    for spd in SPEEDS:
        # ---- unimodal cells at this speed: ONE hierarchical model (partial pooling) ----
        uni = [p for p in pids if (p, spd) not in mixture_cells and len(load_cell(dfi, p, spd)) >= 15]
        uni_todo = [p for p in uni if (p, spd) not in done]
        if uni_todo:
            # build pooled arrays (fit ALL unimodal cells at this speed together for pooling)
            units, rt, uidx, minrt = [], [], [], []; u = 0
            for pid in uni:
                x = load_cell(dfi, pid, spd)
                units.append(pid); minrt.append(x.min()); rt.extend(x.tolist()); uidx.extend([u]*len(x)); u += 1
            rt = np.array(rt); uidx = np.array(uidx); minrt = np.array(minrt)
            print(f"\n[speed {spd}] hierarchical single-Wald: {len(units)} unimodal units, {len(rt)} trials ...", flush=True)
            idata = hierarchical_single(rt, uidx, minrt, len(units))
            div = int(idata.sample_stats["diverging"].sum())
            rh = az.rhat(idata, var_names=["v","a","t0"])
            rmax = float(max(rh["v"].max(), rh["a"].max(), rh["t0"].max()))
            po = idata.posterior
            t0m  = po["t0"].mean(("chain","draw")).values*1000
            t0lo = po["t0"].quantile(0.025,("chain","draw")).values*1000
            t0hi = po["t0"].quantile(0.975,("chain","draw")).values*1000
            vm   = po["v"].mean(("chain","draw")).values
            am   = po["a"].mean(("chain","draw")).values
            try: idata.to_netcdf(os.path.join(SCRIPT_DIR, f"Bayesian_srt_posterior_{spd}_unimodal.nc"))
            except Exception: pass
            for i, pid in enumerate(units):
                if (pid, spd) in done: continue
                append_row(dict(pid=pid, spd=spd, n=int((uidx==i).sum()), model="single",
                                conv_div=div, conv_rhat=round(rmax,3),
                                v=round(float(vm[i]),3), a=round(float(am[i]),4),
                                t0=round(float(t0m[i])), t0_lo95=round(float(t0lo[i])), t0_hi95=round(float(t0hi[i]))))
            print(f"[speed {spd}] unimodal done: div={div}, max r-hat={rmax:.3f}", flush=True)
        else:
            print(f"[speed {spd}] unimodal cells already complete; skipping.", flush=True)

        # ---- mixture cells at this speed: one Bayesian mixture each ----
        for pid in pids:
            if (pid, spd) not in mixture_cells: continue
            if (pid, spd) in done_cells(): continue
            rts = load_cell(dfi, pid, spd)
            if len(rts) < 15: continue
            print(f"\n[speed {spd}] Bayesian mixture: {pid} (n={len(rts)}) ...", flush=True)
            idata = bayesian_mixture(rts)
            div = int(idata.sample_stats["diverging"].sum())
            po = idata.posterior
            # per-(chain,draw) arrays, then relabel each draw by component mean
            def cd(n): return po[n].values            # shape (chain, draw)
            pi_, v1_, a1_, t01_, v2_, a2_, t02_ = [cd(n) for n in ["pi","v1","a1","t01","v2","a2","t02"]]
            m1 = t01_ + a1_/v1_; m2 = t02_ + a2_/v2_
            exp_is1 = m1 <= m2
            em_cd = np.where(exp_is1, m1, m2)*1000     # express mode (chain, draw)
            rm_cd = np.where(exp_is1, m2, m1)*1000     # regular mode
            pe_cd = np.where(exp_is1, pi_, 1-pi_)      # express weight
            def rhat_cd(x):                            # manual split-less r-hat on (chain,draw)
                mc, nn = x.shape
                if mc < 2: return np.nan
                cm = x.mean(1); W = x.var(1, ddof=1).mean()
                B = nn*cm.var(ddof=1); vh = (nn-1)/nn*W + B/nn
                return float(np.sqrt(vh/W)) if W > 0 else np.nan
            rmax = max(rhat_cd(em_cd), rhat_cd(pe_cd))  # convergence of the RELABELED estimates
            em = em_cd.reshape(-1); rm = rm_cd.reshape(-1); pe = pe_cd.reshape(-1)
            def ci(x): return round(float(np.mean(x)),3), round(float(np.percentile(x,2.5)),3), round(float(np.percentile(x,97.5)),3)
            pim, pil, pih = ci(pe); emm, eml, emh = ci(em); rmm, rml, rmh = ci(rm)
            append_row(dict(pid=pid, spd=spd, n=len(rts), model="mixture",
                            conv_div=div, conv_rhat=round(rmax,3),
                            pi=pim, pi_lo95=pil, pi_hi95=pih,
                            express_mode=round(emm), express_mode_lo95=round(eml), express_mode_hi95=round(emh),
                            reg_mode=round(rmm), reg_mode_lo95=round(rml), reg_mode_hi95=round(rmh)))
            print(f"[speed {spd}] {pid} mixture done: div={div}, express pi={pim:.2f}, modes {emm:.0f}/{rmm:.0f} ms", flush=True)

    print("\nBAYESIAN_SRT_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
