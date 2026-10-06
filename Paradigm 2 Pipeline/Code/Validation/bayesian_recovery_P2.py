"""
bayesian_recovery_P2.py  --  can the production Bayesian hand model recover a KNOWN change in hand t0?

Design = Paradigm 2's: the same 16 participants x 4 speeds and the real cell sizes. Truth: each participant's
Method B estimates at 75 deg/s (v, a, t0); t0 then falls 10 ms linearly from 75 to 150 deg/s (a Paradigm 1-sized
change, identical for everyone), v and a unchanged. Pure shifted-Wald RTs are simulated inside the 150-800 ms
window and fitted with the UNCHANGED Bayesian_HRT_fit.py (only addition in the working copy: it saves the posterior
draws of v, a, t0 so the decision-time summary can be checked). Scores: cell bias / RMSE / 95% coverage, the
recovered 150 - 75 change and its tests, the decision-time change (truth 0), the noise-only correlation of the two
changes, and what the RAW simulated data show.

Run:  python bayesian_recovery_P2.py                 (needs PyMC; ~5 min on 4 cores, ~7 min on 1)
      python bayesian_recovery_P2.py --analyze-only  (rescore an existing run in _recovery_run/)
Out:  bayesian_recovery_P2_summary.csv, bayesian_recovery_P2_cells.csv
"""
import os, sys, shutil, subprocess, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
HERE = os.path.dirname(os.path.abspath(__file__))
def find(name):
    for p in [os.path.join(HERE, name), os.path.join(HERE, "..", "Bayesian", name), os.path.join(HERE, "..", name)]:
        if os.path.exists(p): return p
    sys.exit(f"ERROR: {name} not found")
RUN = sys.argv[sys.argv.index("--run-dir") + 1] if "--run-dir" in sys.argv else os.path.join(HERE, "_recovery_run")
SP, SEED, DROP = [75, 100, 125, 150], 20261006, 10.0

def simulate():
    os.makedirs(RUN, exist_ok=True)
    base = pd.read_csv(find("Bayesian_hrt_fits.csv")); base = base[base.spd == 75].set_index("pid")
    d = pd.read_csv(find("pooled_data_P2.csv"), usecols=["Participant", "Speed_deg_per_s", "HandRT_ms"])
    ncell = d[d.HandRT_ms.between(150, 800)].groupby(["Participant", "Speed_deg_per_s"]).size()
    rng = np.random.default_rng(SEED); rows, truth = [], []
    for pid in base.index:
        v, a, t00 = base.loc[pid, "v"], base.loc[pid, "a"], base.loc[pid, "t0"] / 1000
        for s in SP:
            t0 = t00 - DROP / 1000 * (s - 75) / 75; n = int(ncell.loc[(pid, s)]); out = np.empty(0)
            while len(out) < n:
                x = t0 + stats.invgauss(mu=1 / (v * a), scale=a ** 2).rvs(size=2 * n, random_state=rng); out = np.concatenate([out, x[(x >= .150) & (x <= .800)]])
            rows += [dict(Participant=pid, BlockType="P2", Speed_deg_per_s=s, HandRT_ms=round(x * 1000, 3), GazeSRT_ms=np.nan) for x in out[:n]]
            truth.append(dict(pid=pid, spd=s, true_v=v, true_a=a, true_t0=t0 * 1000, true_dt=1000 * a / v))
    pd.DataFrame(rows).to_csv(os.path.join(RUN, "pooled_data_P2.csv"), index=False); pd.DataFrame(truth).to_csv(os.path.join(RUN, "truth.csv"), index=False)
    s = open(find("Bayesian_HRT_fit.py")).read(); old = "    po = idata.posterior\n"
    s = s.replace(old, old + "    np.savez(os.path.join(SCRIPT_DIR, 'posterior_draws.npz'), v=po['v'].values, a=po['a'].values, t0=po['t0'].values)\n", 1)
    open(os.path.join(RUN, "Bayesian_HRT_fit.py"), "w").write(s)
    subprocess.run([sys.executable, "Bayesian_HRT_fit.py"], cwd=RUN, check=True)

def analyze():
    f = pd.read_csv(os.path.join(RUN, "Bayesian_hrt_fits.csv")); m = f.merge(pd.read_csv(os.path.join(RUN, "truth.csv")), on=["pid", "spd"])
    e = m.t0 - m.true_t0; cov = (m.true_t0 >= m.t0_lo95) & (m.true_t0 <= m.t0_hi95)
    w = m.pivot_table(index="pid", columns="spd", values="t0"); d = w[150] - w[75]
    r = np.random.default_rng(0); b = [r.choice(d.values, len(d)).mean() for _ in range(3000)]
    m["dt"] = 1000 * m.a / m.v; wd = m.pivot_table(index="pid", columns="spd", values="dt"); dd = wd[150] - wd[75]
    raw = pd.read_csv(os.path.join(RUN, "pooled_data_P2.csv")); g = raw.groupby(["Participant", "Speed_deg_per_s"]).HandRT_ms
    q10, med = g.quantile(0.10).unstack(), g.median().unstack()
    rows = [("cell t0 bias (ms)", e.mean()), ("cell t0 RMSE (ms)", np.sqrt((e ** 2).mean())), ("cell 95% interval coverage (%)", 100 * cov.mean()),
            ("cells at the 130 ms floor", int(((m.t0 - 130).abs() < 2).sum()))]
    rows += [(f"group t0 at {s}: recovered / truth (ms)", f"{m[m.spd == s].t0.mean():.1f} / {m[m.spd == s].true_t0.mean():.1f}") for s in SP]
    rows += [("delta t0 150-75, recovered (truth -10.0)", d.mean()), ("delta t0 bootstrap 95% CI", f"[{np.percentile(b, 2.5):.1f}, {np.percentile(b, 97.5):.1f}]"),
             ("delta t0 participants negative", int((d < 0).sum())), ("delta t0 Wilcoxon p", wilcoxon(d).pvalue),
             ("Friedman p across 4 speeds", friedmanchisquare(*[w[s] for s in SP]).pvalue),
             ("delta decision time a/v 150-75, recovered (truth 0)", dd.mean()), ("delta decision time Wilcoxon p", wilcoxon(dd).pvalue),
             ("corr(delta t0, delta decision time) with no true variation", np.corrcoef(d, dd)[0, 1]),
             ("raw simulated data: delta 10th-percentile HRT 150-75 (ms)", (q10[150] - q10[75]).mean()),
             ("raw simulated data: delta median HRT 150-75 (ms)", (med[150] - med[75]).mean())]
    npz = os.path.join(RUN, "posterior_draws.npz")
    if os.path.exists(npz):
        z = np.load(npz); per_draw = (1000 * z["a"] / z["v"]).mean(axis=(0, 1))
        rows.append(("decision time: |ratio of posterior means - posterior mean of a/v| max (ms)", float(np.max(np.abs(1000 * f.a.values / f.v.values - per_draw)))))
    out = pd.DataFrame(rows, columns=["quantity", "value"]); out["value"] = [round(v, 4) if isinstance(v, float) else v for v in out.value]
    out.to_csv(os.path.join(HERE, "bayesian_recovery_P2_summary.csv"), index=False)
    m.assign(covered=cov).round(3).to_csv(os.path.join(HERE, "bayesian_recovery_P2_cells.csv"), index=False); print(out.to_string(index=False))

if __name__ == "__main__":
    if "--analyze-only" not in sys.argv: simulate()
    analyze()
