"""
Bayesian_SRT_ndt.py  --  Hierarchical Bayesian: saccadic NDT per participant

Estimates participant-level saccadic t0 (shared across speeds) via PyMC/NUTS.
Replaces per-cell flooring with partial pooling: t0 is shared across speeds per
participant, regularized toward the group mean.

Outputs: Bayesian_srt_ndt.csv (t0, 95% CI, trial count), Bayesian_srt_ndt.pdf/.png

For model details, see CODE_REFERENCE.md.

Run: python Bayesian_SRT_ndt.py  (needs DDM_srt_fits.csv first)
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
        "Install via conda: conda create -n snl python=3.11 && conda activate snl\n"
        "&& conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib\n"
        "\n"
        "Pre-validated outputs are already saved, so results are complete\n"
        "even without re-running.\n"
        + "=" * 72 + "\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family":_fam,"font.size":11,"pdf.fonttype":42,"ps.fonttype":42})

HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run DDM_fit.py first.")
    return p
LO, HI = 0.080, 0.600
CONTAM = 0.05; CDENS = CONTAM/(HI-LO)
FLOOR = 0.070                       # saccadic non-decision floor (70 ms): harmonized with the per-cell SRT fit and the saccadic dead-time literature (Bompas et al. 2017; Ludwig et al. 2007). Estimates are no longer allowed below the physiological saccadic minimum.
T0_PRIOR_MEAN, T0_PRIOR_SD = 0.070, 0.030   # weakly-informative physiological prior
DRAWS, TUNE, CHAINS, TARGET = 1000, 1500, 4, 0.95

def load(dfi, pid, spd):
    z = dfi[(dfi.Participant==pid)&(dfi.Speed_deg_per_s==spd)]
    x = z['GazeSRT_ms'].values.astype(float); return x[(~np.isnan(x))&(x>=80)&(x<=600)]/1000

def main():
    dfi = pd.read_csv(_need('pooled_data.csv')); dfi = dfi[dfi.BlockType=='I']
    s = pd.read_csv(_need('DDM_srt_fits.csv')); sing = s[s.model=='single']
    parts = sorted(sing.pid.unique()); pmap = {p:i for i,p in enumerate(parts)}
    cells = [(r.pid, int(r.spd)) for _, r in sing.iterrows()]
    rt=[]; cidx=[]; pidx_t=[]; minrt_p={p:1.0 for p in parts}
    for ci,(pid,spd) in enumerate(cells):
        x = load(dfi,pid,spd); rt.extend(x); cidx.extend([ci]*len(x)); pidx_t.extend([pmap[pid]]*len(x))
        minrt_p[pid] = min(minrt_p[pid], x.min())
    rt=np.array(rt); cidx=np.array(cidx); pidx_t=np.array(pidx_t)
    minrt_arr=np.array([minrt_p[p] for p in parts])
    ncell_p = np.array([sum(1 for pid,_ in cells if pmap[pid]==i) for i in range(len(parts))])
    print(f"{len(parts)} participants, {len(cells)} single cells, {len(rt)} trials", flush=True)

    with pm.Model() as m:
        mu_t0 = pm.Normal("mu_t0", T0_PRIOR_MEAN, T0_PRIOR_SD)      # population mean NDT
        sig_t0 = pm.HalfNormal("sig_t0", 0.025)                     # between-participant SD
        t0_p = pm.TruncatedNormal("t0_p", mu=mu_t0, sigma=sig_t0,
                                  lower=FLOOR, upper=minrt_arr-0.001, shape=len(parts))
        v = pm.Deterministic("v", pt.exp(pm.Normal("lv", np.log(12), 0.8, shape=len(cells))))
        a = pm.Deterministic("a", pt.exp(pm.Normal("la", np.log(1.2), 0.8, shape=len(cells))))
        tau = rt - t0_p[pidx_t]
        wlp = pt.log(a[cidx]) - 0.5*np.log(2*np.pi) - 1.5*pt.log(tau) - (a[cidx]-v[cidx]*tau)**2/(2*tau)
        comp = pt.stack([np.log(1-CONTAM)+wlp, pt.full(rt.shape, np.log(CDENS))], axis=0)
        pm.Potential("lik", pm.math.logsumexp(comp, axis=0).sum())
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=TARGET, random_seed=7, progressbar=False)

    div = int(idata.sample_stats["diverging"].sum())
    rh = float(az.rhat(idata, var_names=["t0_p"])["t0_p"].max())
    po = idata.posterior
    t0m = po["t0_p"].mean(("chain","draw")).values*1000
    t0lo = po["t0_p"].quantile(0.025,("chain","draw")).values*1000
    t0hi = po["t0_p"].quantile(0.975,("chain","draw")).values*1000
    vm = po["v"].mean(("chain","draw")).values; am = po["a"].mean(("chain","draw")).values
    print(f"convergence: divergences={div}, max t0 r-hat={rh:.3f}", flush=True)
    print(f"population mean t0={float(po['mu_t0'].mean())*1000:.0f}ms, between-participant SD={np.std(t0m):.0f}ms", flush=True)
    print(f"individual t0 range: {t0m.min():.0f}-{t0m.max():.0f}ms", flush=True)

    # per-participant t0 table
    prows = [dict(pid=p, t0_ms=round(t0m[i]), t0_lo95=round(t0lo[i]), t0_hi95=round(t0hi[i]),
                  ci_width_ms=round(t0hi[i]-t0lo[i]), n_cells=int(ncell_p[i]),
                  min_srt_ms=round(minrt_arr[i]*1000),
                  constraint=("well-constrained" if (t0hi[i]-t0lo[i])<35 else "regularized (wide CI)"))
             for i,p in enumerate(parts)]
    # per-cell v,a appended
    crows = [dict(pid=pid, spd=spd, v=round(float(vm[ci]),3), a=round(float(am[ci]),4),
                  t0_ms=round(t0m[pmap[pid]])) for ci,(pid,spd) in enumerate(cells)]
    pd.DataFrame(prows).to_csv(os.path.join(HERE,"Bayesian_srt_ndt.csv"), index=False)
    pd.DataFrame(crows).to_csv(os.path.join(HERE,"Bayesian_srt_ndt_cells.csv"), index=False)

    # forest plot of per-participant t0
    order = np.argsort(t0m)
    fig, ax = plt.subplots(figsize=(9, 7))
    for rank, i in enumerate(order):
        tight = (t0hi[i]-t0lo[i]) < 35
        c = '#2c7fb8' if tight else '#d95f0e'
        ax.plot([t0lo[i], t0hi[i]], [rank, rank], color=c, lw=2.2, alpha=0.85)
        ax.plot(t0m[i], rank, 'o', color=c, ms=6)
    ax.axvline(float(po['mu_t0'].mean())*1000, color='#444', ls='--', lw=1.3)
    ax.set_yticks(range(len(parts))); ax.set_yticklabels([parts[i] for i in order], fontsize=8.5)
    ax.set_xlabel('saccadic non-decision time $t_0$ (ms, posterior mean ± 95% CI)')
    ax.set_title('Per-participant saccadic non-decision time — ESTIMATED, not fixed\n'
                 'individual differences preserved; each estimate carries its own uncertainty', fontsize=11.5, fontweight='bold')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color='#2c7fb8',lw=2,label='tighter estimate (95% CI < 35 ms)'),
                       Line2D([0],[0],color='#d95f0e',lw=2,label='looser estimate (95% CI ≥ 35 ms)'),
                       Line2D([0],[0],color='#444',ls='--',label=f"population mean ({float(po['mu_t0'].mean())*1000:.0f} ms)")], fontsize=9, loc='lower right')
    ax.spines[['top','right']].set_visible(False); ax.grid(True, axis='x', ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE,"Bayesian_srt_ndt.pdf"), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(HERE,"Bayesian_srt_ndt.png"), dpi=140, bbox_inches='tight', facecolor='white')
    print("saved Bayesian_srt_ndt.csv, Bayesian_srt_ndt_cells.csv, Bayesian_srt_ndt.pdf/.png to", HERE, flush=True)
    print("BAYESIAN_SRT_NDT_DONE", flush=True)

if __name__ == "__main__":
    main()
