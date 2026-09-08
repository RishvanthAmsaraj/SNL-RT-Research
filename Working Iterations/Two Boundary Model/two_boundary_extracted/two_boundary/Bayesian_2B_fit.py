"""
Bayesian_2B_fit.py  --  Hierarchical Bayesian TWO-BOUNDARY DDM (Method B, two-boundary)

Drop-in successor to Bayesian_HRT_fit.py / Bayesian_SRT_fit.py for data that has a
per-trial binary outcome. Same architecture as the existing pipeline -- PyMC/NUTS,
partial pooling across participants, non-centred parametrisation, speed as a modelled
factor so group parameters come with credible intervals -- but the likelihood is the
Wiener first-passage-time density over (RT, response) instead of the single-boundary
shifted Wald over RT alone.

    a   boundary SEPARATION (barriers at 0 and a)   -- roughly 2x the single-boundary a
    v   drift rate, SIGN FREE                       -- +v drifts toward the upper barrier
    w   relative start point z/a in (0,1)           -- 0.5 unbiased; this is the new
                                                       parameter the single-boundary
                                                       model could not express
    t0  non-decision time                           -- same meaning, same floor as before

Run it once per effector so hand and eye are modelled identically:

    python Bayesian_2B_fit.py --data trials.csv --effector hand --sign-col HandErr_deg
    python Bayesian_2B_fit.py --data trials.csv --effector eye  --sign-col GazeErr_deg

Outputs
    Bayesian2B_{eff}_group.csv    group v, a, w, t0 per speed with 94% CIs
    Bayesian2B_{eff}_cells.csv    per participant x speed estimates
    Bayesian2B_{eff}_gof.csv      posterior predictive KS on RT + choice proportion
    Bayesian2B_{eff}_idata.nc     full posterior (re-analysable without refitting)
    Bayesian2B_{eff}_compare.csv  LOO: two-boundary DDM vs Wald + independent Bernoulli

Requires PyMC (conda install -c conda-forge pymc arviz -- pip-only PyMC on Windows
is not viable, per the pipeline notes).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from wfpt import numpy_wfpt_logpdf_vec, sample_ddm

SPEEDS = [0, 75, 150]
FLOORS = {"hand": 0.130, "eye": 0.070}      # Haith et al. 2016 / Bompas & Sumner 2011
WINDOWS = {"hand": (0.150, 0.800), "eye": (0.080, 0.600)}
LOGLIK_DRAWS = 400                           # thinned draws used for pointwise log-lik


# --------------------------------------------------------------------------- data
def load(args):
    df = pd.read_csv(args.data, sep=None, engine="python")
    if args.blocktype_col in df.columns and args.blocktype_keep:
        df = df[df[args.blocktype_col].astype(str) == args.blocktype_keep]

    rt_col = args.rt_col or ("HandRT_ms" if args.effector == "hand" else "GazeSRT_ms")
    missing = [c for c in (args.participant_col, args.speed_col, rt_col) if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing column(s) {missing}. Available: {list(df.columns)}")

    if args.choice_col:
        raw = df[args.choice_col]
        vals = sorted(pd.Series(raw.dropna().unique()).astype(str))
        if len(vals) != 2:
            sys.exit(f"ERROR: --choice-col {args.choice_col} has {len(vals)} unique values.")
        upper = args.upper_value if args.upper_value is not None else vals[1]
        ch = (raw.astype(str) == str(upper)).astype(float).where(raw.notna())
        label = f"{args.choice_col} == {upper}"
    elif args.sign_col:
        x = pd.to_numeric(df[args.sign_col], errors="coerce")
        ch = (x > 0).astype(float).where(x.notna() & (x != 0))
        label = f"sign({args.sign_col}) > 0"
    else:
        sys.exit("ERROR: a two-boundary DDM needs a per-trial binary outcome. "
                 "Pass --choice-col or --sign-col. Run two_boundary_readiness.py "
                 "in inspect mode to see what your file offers.")

    d = pd.DataFrame({
        "pid": df[args.participant_col].astype(str),
        "spd": pd.to_numeric(df[args.speed_col], errors="coerce"),
        "rt": pd.to_numeric(df[rt_col], errors="coerce") / 1000.0,
        "choice": ch,
    }).dropna()
    lo, hi = WINDOWS[args.effector]
    d = d[(d.rt >= lo) & (d.rt <= hi) & d.spd.isin(SPEEDS)]
    d["spd"] = d.spd.astype(int)
    return d.reset_index(drop=True), label


# --------------------------------------------------------------------------- model
def build_and_sample(d, floor, args):
    import pymc as pm
    import pytensor.tensor as pt
    from wfpt import pt_wfpt_logpdf

    pids = sorted(d.pid.unique())
    pid_idx = d.pid.map({p: i for i, p in enumerate(pids)}).values
    spd_idx = d.spd.map({s: i for i, s in enumerate(SPEEDS)}).values
    rt = d.rt.values.astype(float)
    choice = d.choice.values.astype(float)

    # per-cell minimum RT: t0 for a cell must stay below it or the density is undefined
    min_rt = np.full((len(SPEEDS), len(pids)), floor + 0.5)
    for (p, s), g in d.groupby(["pid", "spd"]):
        min_rt[SPEEDS.index(s), pids.index(p)] = g.rt.min()

    coords = {"speed": [str(s) for s in SPEEDS], "participant": pids}
    with pm.Model(coords=coords) as model:
        # --- drift: Normal, so the sign is free (an interception outcome can favour
        #     either response and the drift must be able to say so)
        mu_v = pm.Normal("mu_v", 0.0, 5.0, dims="speed")
        sd_v = pm.HalfNormal("sd_v", 2.0)
        z_v = pm.Normal("z_v", 0.0, 1.0, dims=("speed", "participant"))
        v = pm.Deterministic("v", mu_v[:, None] + sd_v * z_v, dims=("speed", "participant"))

        # --- boundary separation: positive, log-normal (cf. mu_la ~ N(log 1, 0.5) in the
        #     single-boundary scripts, shifted up because a is now a separation)
        mu_la = pm.Normal("mu_la", np.log(1.5), 0.5, dims="speed")
        sd_la = pm.HalfNormal("sd_la", 0.5)
        z_a = pm.Normal("z_a", 0.0, 1.0, dims=("speed", "participant"))
        a = pm.Deterministic("a", pt.exp(mu_la[:, None] + sd_la * z_a),
                             dims=("speed", "participant"))

        # --- relative start point, logit scale; mu_lw = 0 means unbiased
        mu_lw = pm.Normal("mu_lw", 0.0, 0.7, dims="speed")
        sd_lw = pm.HalfNormal("sd_lw", 0.5)
        z_w = pm.Normal("z_w", 0.0, 1.0, dims=("speed", "participant"))
        w = pm.Deterministic("w", pm.math.sigmoid(mu_lw[:, None] + sd_lw * z_w),
                             dims=("speed", "participant"))

        # --- non-decision time, same bounded transform as the existing scripts:
        #     t0 = FLOOR + (min RT - FLOOR) * sigmoid(.) * 0.98
        mu_lt = pm.Normal("mu_lt", 0.0, 1.0, dims="speed")
        sd_lt = pm.HalfNormal("sd_lt", 1.0)
        z_t = pm.Normal("z_t", 0.0, 1.0, dims=("speed", "participant"))
        t0 = pm.Deterministic(
            "t0", floor + (min_rt - floor) * pm.math.sigmoid(mu_lt[:, None] + sd_lt * z_t) * 0.98,
            dims=("speed", "participant"))

        tau = rt - t0[spd_idx, pid_idx]
        lp = pt_wfpt_logpdf(tau, v[spd_idx, pid_idx], a[spd_idx, pid_idx],
                            w[spd_idx, pid_idx], choice)
        if args.contamination > 0:
            span = float(rt.max() - rt.min())
            lp = pt.logaddexp(np.log1p(-args.contamination) + lp,
                              np.log(args.contamination / (2.0 * span)) + 0.0 * lp)
        pm.Potential("lik", pt.sum(lp))

        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=args.chains, target_accept=args.target_accept,
                          random_seed=args.seed, progressbar=True)
    return idata, pids, pid_idx, spd_idx, rt, choice, min_rt


# --------------------------------------------------------------------------- tables
def summarise(idata, pids, args):
    import arviz as az
    post = idata.posterior

    grows = []
    for i, s in enumerate(SPEEDS):
        row = {"speed": s}
        for name, lab in [("mu_v", "v"), ("mu_la", "a"), ("mu_lw", "w"), ("mu_lt", "t0_ms")]:
            x = post[name].isel(speed=i).values.ravel()
            if name == "mu_la":
                x = np.exp(x)
            elif name == "mu_lw":
                x = 1.0 / (1.0 + np.exp(-x))
            elif name == "mu_lt":
                # report the group t0 as the mean of the per-participant t0 at this speed
                x = post["t0"].isel(speed=i).values.reshape(-1, len(pids)).mean(axis=1) * 1000.0
            lo, hi = az.hdi(x, hdi_prob=0.94)
            row[lab] = float(x.mean()); row[f"{lab}_lo"] = float(lo); row[f"{lab}_hi"] = float(hi)
        grows.append(row)
    group = pd.DataFrame(grows)

    crows = []
    for i, s in enumerate(SPEEDS):
        for j, p in enumerate(pids):
            r = {"pid": p, "spd": s}
            for name in ("v", "a", "w", "t0"):
                x = post[name].isel(speed=i, participant=j).values.ravel()
                lo, hi = az.hdi(x, hdi_prob=0.94)
                scale = 1000.0 if name == "t0" else 1.0
                r[name] = float(x.mean()) * scale
                r[f"{name}_lo"] = float(lo) * scale
                r[f"{name}_hi"] = float(hi) * scale
            crows.append(r)
    cells = pd.DataFrame(crows)
    return group, cells


def pointwise_loglik(idata, pids, pid_idx, spd_idx, rt, choice, n_draws=LOGLIK_DRAWS):
    """Pointwise log-likelihood of the two-boundary DDM, for LOO."""
    post = idata.posterior
    nchain, ndraw = post.sizes["chain"], post.sizes["draw"]
    flat = lambda name: post[name].values.reshape(nchain * ndraw, len(SPEEDS), len(pids))
    V, A, W, T = flat("v"), flat("a"), flat("w"), flat("t0")
    total = V.shape[0]
    sel = np.linspace(0, total - 1, min(n_draws, total)).astype(int)
    out = np.empty((len(sel), len(rt)))
    for k, s in enumerate(sel):
        out[k] = numpy_wfpt_logpdf_vec(rt - T[s][spd_idx, pid_idx], V[s][spd_idx, pid_idx],
                                       A[s][spd_idx, pid_idx], W[s][spd_idx, pid_idx], choice)
    return out, sel


def gof(idata, d, pids, args, n_sim_draws=40):
    """Posterior predictive: KS on RT and the choice proportion, per cell."""
    from scipy.stats import ks_2samp
    post = idata.posterior
    rng = np.random.default_rng(args.seed)
    rows = []
    for (p, s), g in d.groupby(["pid", "spd"]):
        i, j = SPEEDS.index(s), pids.index(p)
        V = post["v"].isel(speed=i, participant=j).values.ravel()
        A = post["a"].isel(speed=i, participant=j).values.ravel()
        W = post["w"].isel(speed=i, participant=j).values.ravel()
        T = post["t0"].isel(speed=i, participant=j).values.ravel()
        sel = rng.choice(len(V), size=min(n_sim_draws, len(V)), replace=False)
        sim_rt, sim_ch = [], []
        per = max(20, int(np.ceil(len(g) * 4 / len(sel))))
        for k in sel:
            r, c = sample_ddm(per, V[k], A[k], W[k], T[k], rng=rng)
            ok = np.isfinite(r)
            sim_rt.append(r[ok]); sim_ch.append(c[ok])
        sim_rt = np.concatenate(sim_rt); sim_ch = np.concatenate(sim_ch)
        rows.append(dict(pid=p, spd=s, n=len(g),
                         ks=float(ks_2samp(g.rt.values, sim_rt).statistic),
                         p_upper_obs=float(g.choice.mean()),
                         p_upper_pred=float(sim_ch.mean())))
    return pd.DataFrame(rows)


def compare_against_wald(d, floor, args, ll_2b, sel):
    """
    LOO comparison on the SAME joint data (RT, choice):
        M1  two-boundary DDM             -- RT and choice coupled by one diffusion
        M2  shifted Wald x Bernoulli     -- RT and choice modelled independently
    If M1 does not win, the coupling the two-boundary model adds is not earning its
    keep and the single-boundary model plus a separate choice model is the honest report.
    """
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    pids = sorted(d.pid.unique())
    pid_idx = d.pid.map({p: i for i, p in enumerate(pids)}).values
    spd_idx = d.spd.map({s: i for i, s in enumerate(SPEEDS)}).values
    rt, choice = d.rt.values, d.choice.values
    min_rt = np.full((len(SPEEDS), len(pids)), floor + 0.5)
    for (p, s), g in d.groupby(["pid", "spd"]):
        min_rt[SPEEDS.index(s), pids.index(p)] = g.rt.min()

    coords = {"speed": [str(s) for s in SPEEDS], "participant": pids}
    with pm.Model(coords=coords):
        mu_lv = pm.Normal("mu_lv", np.log(10), 0.5, dims="speed")
        sd_lv = pm.HalfNormal("sd_lv", 0.5)
        zv = pm.Normal("zv", 0, 1, dims=("speed", "participant"))
        v = pt.exp(mu_lv[:, None] + sd_lv * zv)

        mu_la = pm.Normal("mu_la", np.log(1.0), 0.5, dims="speed")
        sd_la = pm.HalfNormal("sd_la", 0.5)
        za = pm.Normal("za", 0, 1, dims=("speed", "participant"))
        a = pt.exp(mu_la[:, None] + sd_la * za)

        mu_lt = pm.Normal("mu_lt", 0, 1, dims="speed")
        sd_lt = pm.HalfNormal("sd_lt", 1.0)
        zt = pm.Normal("zt", 0, 1, dims=("speed", "participant"))
        t0 = pm.Deterministic("t0", floor + (min_rt - floor)
                              * pm.math.sigmoid(mu_lt[:, None] + sd_lt * zt) * 0.98,
                              dims=("speed", "participant"))

        mu_lp = pm.Normal("mu_lp", 0, 1.5, dims="speed")
        sd_lp = pm.HalfNormal("sd_lp", 1.0)
        zp = pm.Normal("zp", 0, 1, dims=("speed", "participant"))
        pr = pm.Deterministic("p_upper", pm.math.sigmoid(mu_lp[:, None] + sd_lp * zp),
                              dims=("speed", "participant"))

        tau = rt - t0[spd_idx, pid_idx]
        vi, ai, pi_ = v[spd_idx, pid_idx], a[spd_idx, pid_idx], pr[spd_idx, pid_idx]
        wald = (pt.log(ai) - 0.5 * np.log(2 * np.pi) - 1.5 * pt.log(tau)
                - (ai - vi * tau) ** 2 / (2 * tau))
        bern = choice * pt.log(pi_) + (1 - choice) * pt.log(1 - pi_)
        pm.Potential("lik", pt.sum(wald + bern))
        idata_w = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                            cores=args.chains, target_accept=args.target_accept,
                            random_seed=args.seed + 1, progressbar=True)

    post = idata_w.posterior
    nc, nd = post.sizes["chain"], post.sizes["draw"]
    fl = lambda n: post[n].values.reshape(nc * nd, len(SPEEDS), len(pids))
    T = fl("t0"); P = fl("p_upper")
    with np.errstate(all="ignore"):
        Vv = np.exp((post["mu_lv"].values[..., None] + post["sd_lv"].values[..., None, None]
                     * post["zv"].values).reshape(nc * nd, len(SPEEDS), len(pids)))
        Aa = np.exp((post["mu_la"].values[..., None] + post["sd_la"].values[..., None, None]
                     * post["za"].values).reshape(nc * nd, len(SPEEDS), len(pids)))
    ll_w = np.empty((len(sel), len(rt)))
    for k, s in enumerate(sel):
        tau = rt - T[s][spd_idx, pid_idx]
        vi, ai, pi_ = Vv[s][spd_idx, pid_idx], Aa[s][spd_idx, pid_idx], P[s][spd_idx, pid_idx]
        with np.errstate(all="ignore"):
            wl = (np.log(ai) - 0.5 * np.log(2 * np.pi) - 1.5 * np.log(np.maximum(tau, 1e-12))
                  - (ai - vi * tau) ** 2 / (2 * np.maximum(tau, 1e-12)))
        ll_w[k] = np.where(tau > 0, wl, -np.inf) + \
            choice * np.log(pi_) + (1 - choice) * np.log(1 - pi_)

    def as_idata(ll):
        return az.from_dict(posterior={"dummy": np.zeros((1, ll.shape[0]))},
                            log_likelihood={"obs": ll[None, ...]})

    cmp = az.compare({"two_boundary_DDM": as_idata(ll_2b),
                      "Wald_x_Bernoulli": as_idata(ll_w)}, ic="loo")
    return cmp.reset_index().rename(columns={"index": "model"})


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--effector", default="hand", choices=["hand", "eye"])
    ap.add_argument("--participant-col", default="Participant")
    ap.add_argument("--speed-col", default="Speed_deg_per_s")
    ap.add_argument("--blocktype-col", default="BlockType")
    ap.add_argument("--blocktype-keep", default="I")
    ap.add_argument("--rt-col", default=None)
    ap.add_argument("--choice-col", default=None)
    ap.add_argument("--upper-value", default=None)
    ap.add_argument("--sign-col", default=None)
    ap.add_argument("--contamination", type=float, default=0.0,
                    help="uniform contamination share (DDM_fit.py Method A uses 0.05)")
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the LOO comparison against Wald x Bernoulli")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    try:
        import pymc, arviz as az       # noqa: F401
    except Exception:
        sys.exit("ERROR: PyMC is required.\n"
                 "  conda install -c conda-forge pymc arviz\n"
                 "Pip-only alternatives are not adequate substitutes here.")
    import arviz as az

    d, label = load(args)
    floor = FLOORS[args.effector]
    eff = args.effector
    os.makedirs(args.out, exist_ok=True)

    n_min = d.groupby(["pid", "spd"]).choice.apply(lambda c: min(c.sum(), len(c) - c.sum()))
    print("=" * 78)
    print(f"Hierarchical two-boundary DDM -- {eff}   (outcome: {label})")
    print("=" * 78)
    print(f"{len(d):,} trials, {d.pid.nunique()} participants, "
          f"{len(d.groupby(['pid', 'spd']))} cells")
    print(f"overall upper-barrier share: {d.choice.mean():.3f}")
    print(f"minority responses: {int(n_min.sum()):,} total, "
          f"median {n_min.median():.0f} per cell")
    if n_min.sum() < 200:
        print("\nWARNING: very few minority responses. The boundary separation and the "
              "\nstart point will be driven by the priors, not the data. Run "
              "\ntwo_boundary_readiness.py and read its verdict before trusting this fit.\n")

    idata, pids, pid_idx, spd_idx, rt, choice, _ = build_and_sample(d, floor, args)

    summ = az.summary(idata, var_names=["mu_v", "mu_la", "mu_lw", "mu_lt",
                                        "sd_v", "sd_la", "sd_lw", "sd_lt"])
    max_rhat = float(summ["r_hat"].max())
    n_div = int(idata.sample_stats["diverging"].values.sum())
    print(f"\nmax R-hat {max_rhat:.4f}   divergences {n_div}"
          f"   {'CONVERGED' if (max_rhat < 1.01 and n_div == 0) else 'CHECK THIS'}")

    group, cells = summarise(idata, pids, args)
    print("\nGroup parameters by speed (94% HDI):")
    print(group.round(3).to_string(index=False))

    g = gof(idata, d, pids, args)
    print(f"\nposterior predictive: median KS on RT = {g.ks.median():.3f}; "
          f"median |observed - predicted| choice proportion = "
          f"{(g.p_upper_obs - g.p_upper_pred).abs().median():.3f}")

    group.to_csv(os.path.join(args.out, f"Bayesian2B_{eff}_group.csv"), index=False)
    cells.to_csv(os.path.join(args.out, f"Bayesian2B_{eff}_cells.csv"), index=False)
    g.to_csv(os.path.join(args.out, f"Bayesian2B_{eff}_gof.csv"), index=False)
    try:
        idata.to_netcdf(os.path.join(args.out, f"Bayesian2B_{eff}_idata.nc"))
    except Exception as e:
        print(f"WARNING: could not save NetCDF posterior ({e}); continuing to LOO comparison.")

    if not args.no_compare:
        print("\nFitting the comparison model (shifted Wald x independent Bernoulli)...")
        ll_2b, sel = pointwise_loglik(idata, pids, pid_idx, spd_idx, rt, choice)
        cmp = compare_against_wald(d, floor, args, ll_2b, sel)
        cmp.to_csv(os.path.join(args.out, f"Bayesian2B_{eff}_compare.csv"), index=False)
        print("\nLOO comparison (does coupling RT and choice buy anything?):")
        print(cmp.round(2).to_string(index=False))

    print(f"\nwrote Bayesian2B_{eff}_*.csv and Bayesian2B_{eff}_idata.nc to {args.out}")


if __name__ == "__main__":
    main()
