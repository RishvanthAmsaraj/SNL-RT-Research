"""
Per-speed hierarchical shifted-Wald models -- the two model-structure upgrades
from the roadmap.

Unlike the pooled model (one global prior over all participant x speed units),
these treat target speed as a modelled factor: a group mean per speed plus
participant random effects. That yields group-level drift, boundary, and
non-decision time PER SPEED with credible intervals, and (optionally) models the
participant effects as correlated across parameters with an LKJ prior.

  fit_per_speed(df, effector, correlated=False)  ->  (idata, group_summary[, corr])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._speeds import SPEEDS, PHYSIO_FLOOR

_LOG2PI = float(np.log(2.0 * np.pi))


def _units_pc(df: pd.DataFrame, effector: str):
    """Build per-(participant,condition) unit index arrays for the trial vector."""
    sub = df[df["effector"] == effector]
    pids = sorted(sub["participant"].unique())
    pid_map = {p: i for i, p in enumerate(pids)}
    rt, uidx, u_pid, u_cid, minrt = [], [], [], [], []
    u = 0
    for pid in pids:
        for c in range(len(SPEEDS)):
            x = sub[(sub.participant == pid) & (sub.condition == c)]["rt"].values.astype(float)
            x = x[np.isfinite(x)]
            if len(x) < 15:
                continue
            u_pid.append(pid_map[pid]); u_cid.append(c); minrt.append(float(x.min()))
            rt.extend(x.tolist()); uidx.extend([u] * len(x)); u += 1
    return (pids, np.array(rt), np.array(uidx), np.array(u_pid, dtype="int64"),
            np.array(u_cid, dtype="int64"), np.array(minrt))


def fit_per_speed(df: pd.DataFrame, effector: str, correlated: bool = False,
                  draws=1000, tune=1000, chains=4, target_accept=0.95, cores=1,
                  seed=7, progressbar=False):
    """
    Fit the per-speed hierarchical model. Returns (idata, group_summary) and, when
    correlated=True, also the participant-effect correlation matrix.
    """
    import pymc as pm
    import pytensor.tensor as pt

    floor = PHYSIO_FLOOR[effector]
    mu_z_mean = 0.5 if effector == "hand" else 0.0
    pids, rt, uidx, u_pid, u_cid, minrt = _units_pc(df, effector)
    n_part, n_cond = len(pids), len(SPEEDS)
    coords = {"participant": pids, "condition": [SPEEDS[c] for c in range(n_cond)],
              "param": ["log_v", "log_a", "z"]}
    mean_minrt = np.array([minrt[u_cid == c].mean() if np.any(u_cid == c) else floor + 0.05
                           for c in range(n_cond)])

    with pm.Model(coords=coords) as model:
        mu_lv = pm.Normal("mu_lv_cond", np.log(10), 0.5, dims="condition")
        mu_la = pm.Normal("mu_la_cond", np.log(1.0), 0.5, dims="condition")
        mu_z = pm.Normal("mu_z_cond", mu_z_mean, 1.0, dims="condition")

        if correlated:
            chol, corr, sigmas = pm.LKJCholeskyCov(
                "chol", n=3, eta=2.0, sd_dist=pm.HalfNormal.dist(0.3), compute_corr=True)
            z_raw = pm.Normal("z_raw", 0, 1, dims=("participant", "param"))
            offset = pm.Deterministic("offset", z_raw @ chol.T, dims=("participant", "param"))
            off_lv, off_la, off_z = offset[:, 0], offset[:, 1], offset[:, 2]
            pm.Deterministic("corr", corr)
        else:
            s_lv = pm.HalfNormal("s_lv", 0.3); s_la = pm.HalfNormal("s_la", 0.3)
            s_z = pm.HalfNormal("s_z", 0.5)
            off_lv = s_lv * pm.Normal("z_lv", 0, 1, dims="participant")
            off_la = s_la * pm.Normal("z_la", 0, 1, dims="participant")
            off_z = s_z * pm.Normal("z_z", 0, 1, dims="participant")

        lv = mu_lv[u_cid] + off_lv[u_pid]
        la = mu_la[u_cid] + off_la[u_pid]
        z = mu_z[u_cid] + off_z[u_pid]
        v = pt.exp(lv); a = pt.exp(la)
        t0 = floor + (minrt - floor) * pm.math.sigmoid(z)

        tau = rt - t0[uidx]
        logp = (pt.log(a[uidx]) - 0.5 * _LOG2PI - 1.5 * pt.log(tau)
                - (a[uidx] - v[uidx] * tau) ** 2 / (2 * tau))
        pm.Potential("lik", logp.sum())

        # group-level per-speed quantities (participant offset = 0)
        pm.Deterministic("v_group", pt.exp(mu_lv), dims="condition")
        pm.Deterministic("a_group", pt.exp(mu_la), dims="condition")
        pm.Deterministic("t0_group_ms",
                         (floor + (mean_minrt - floor) * pm.math.sigmoid(mu_z)) * 1000.0,
                         dims="condition")

        idata = pm.sample(draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar)

    # summarise group-level per speed with 94% credible intervals
    post = idata.posterior
    rows = []
    for i in range(n_cond):
        row = {"effector": effector, "speed": SPEEDS[i], "t0_floor_ms": floor * 1000.0}
        for name, lab in [("v_group", "v"), ("a_group", "a"), ("t0_group_ms", "t0_ms")]:
            vals = post[name].isel(condition=i).values.flatten()
            lo, hi = np.percentile(vals, [3, 97])
            row[lab] = float(np.mean(vals)); row[f"{lab}_lo"] = float(lo); row[f"{lab}_hi"] = float(hi)
        rows.append(row)
    group = pd.DataFrame(rows)

    if correlated:
        cm = post["corr"].mean(("chain", "draw")).values
        labels = ["log_v", "log_a", "z(t0)"]
        corr_df = pd.DataFrame(cm, index=labels, columns=labels).round(3)
        return idata, group, corr_df
    return idata, group
