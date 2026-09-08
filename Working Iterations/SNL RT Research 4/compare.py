"""
compare.py -- fair comparison between non-nested models on the same RT data.

The professor's instruction was: fit LATER to the arm data and compare goodness of
fit; if LATER is better, that's reason enough. Right instinct. But the naive version
of that comparison is unfair in three ways at once, and each one has a fix:

  1. PARAMETER COUNTS DIFFER. Plain LATER has 2 free parameters, the shifted Wald has
     3. A raw log-likelihood or raw KS comparison rewards the Wald for flexibility.
     Fix: compare like-for-like (shifted LATER vs shifted Wald, both 3), and report
     AIC/BIC when counts differ.

  2. CONTAMINATION CONVENTIONS DIFFER. Your Wald likelihood carries a 5% uniform
     contaminant; LATER as normally fit does not. The contaminated Wald gets to
     explain away outliers LATER is forced to fit. Fix: compare both under the SAME
     convention. These functions default to pure likelihoods for both.

  3. KS NULLS DIFFER BY MODEL. A KS of 0.08 does not mean the same thing under the
     Wald as under LATER, because the estimation-induced shrinkage of the null
     differs. Fix: parametric bootstrap each model against ITS OWN null. Same lesson
     as audit finding A1.

WHICH STATISTIC TO LEAD WITH
----------------------------
For two non-nested models with the same parameter count, the Vuong test is the right
answer and is what a reviewer will expect. It tests whether the two models are
equally close to the truth, and its sign says which is favoured. Cross-validated
log-likelihood is the robust cross-check. Bootstrap KS answers a different question
-- whether each model is adequate in absolute terms -- and both can fail at once.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm


# --------------------------------------------------------------------------- #
def aic(ll, k):
    return float(2 * k - 2 * ll)


def bic(ll, k, n):
    return float(k * np.log(n) - 2 * ll)


# --------------------------------------------------------------------------- #
def vuong_test(ll1_i, ll2_i, k1=0, k2=0, correction="none"):
    """
    Vuong (1989) test for two NON-NESTED models.

    ll1_i, ll2_i are per-observation log-likelihoods. Returns z and a two-sided p.
    z > 0 favours model 1; z < 0 favours model 2. Under the null (both equally close
    to the truth) z is asymptotically standard normal.

    correction='aic' or 'bic' penalises the model with more parameters, which is what
    you want when the counts differ.
    """
    ll1_i = np.asarray(ll1_i, float)
    ll2_i = np.asarray(ll2_i, float)
    n = ll1_i.size
    m = ll1_i - ll2_i
    if correction == "aic":
        m = m - (k1 - k2) / n
    elif correction == "bic":
        m = m - (k1 - k2) * np.log(n) / (2 * n)
    sd = m.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return {"z": 0.0, "p": 1.0, "favours": "tie", "n": n}
    z = float(np.sqrt(n) * m.mean() / sd)
    p = float(2 * norm.sf(abs(z)))
    fav = "model1" if z > 0 else ("model2" if z < 0 else "tie")
    if p >= 0.05:
        fav = "tie (not distinguishable)"
    return {"z": z, "p": p, "favours": fav, "n": n}


# --------------------------------------------------------------------------- #
def cv_loglik(rt, fit_fn, logpdf_fn, k_folds=5, seed=0):
    """
    K-fold cross-validated total log-likelihood. Fits on the training folds and
    scores the held-out fold, so it penalises overfitting without assuming an
    asymptotic parameter penalty. Higher is better.

    fit_fn(train_rt) -> params object
    logpdf_fn(test_rt, params) -> per-observation log density
    """
    rt = np.asarray(rt, float)
    n = rt.size
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k_folds)
    total = 0.0
    for f in folds:
        mask = np.ones(n, bool)
        mask[f] = False
        try:
            p = fit_fn(rt[mask])
            lp = logpdf_fn(rt[f], p)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(lp)):
            return -np.inf
        total += float(np.sum(lp))
    return total


# --------------------------------------------------------------------------- #
def bootstrap_ks_generic(rt, fit_fn, ks_fn, rvs_fn, B=200, seed=0):
    """
    Parametric-bootstrap p-value for the KS statistic of ANY model.

    Each model gets its own null, because the estimation-induced shrinkage of the KS
    distribution depends on the model being fitted. Comparing raw KS across model
    families is not meaningful; comparing bootstrap p-values is.
    """
    rt = np.asarray(rt, float)
    fit = fit_fn(rt)
    obs = ks_fn(np.sort(rt), fit)
    n = rt.size
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        sim = np.sort(rvs_fn(fit, n, rng))
        try:
            fb = fit_fn(sim)
            null[b] = ks_fn(sim, fb)
        except Exception:
            null[b] = np.nan
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"ks": obs, "ks_p": np.nan, "crit95": np.nan, "rejected": None}
    p = float((np.sum(null >= obs) + 1) / (null.size + 1))
    return {"ks": float(obs), "ks_p": p, "p_floor": 1.0 / (null.size + 1),
            "crit95": float(np.percentile(null, 95)),
            "rejected": bool(p < 0.05), "B": int(null.size)}


# --------------------------------------------------------------------------- #
def compare_models(rt, m1, m2, B=200, seed=0, cv_folds=5):
    """
    Full head-to-head between two model specs.

    Each spec is a dict:
        name, fit (rt -> params), logpdf (rt, params -> per-obs log density),
        ks (sorted_rt, params -> KS), rvs (params, n, rng -> sample), k (n params)

    Returns every statistic needed to defend a model choice, plus a verdict that
    only fires when the evidence actually separates them.
    """
    rt = np.asarray(rt, float)
    n = rt.size
    f1, f2 = m1["fit"](rt), m2["fit"](rt)
    l1 = np.asarray(m1["logpdf"](rt, f1), float)
    l2 = np.asarray(m2["logpdf"](rt, f2), float)
    ll1, ll2 = float(l1.sum()), float(l2.sum())
    k1, k2 = m1["k"], m2["k"]

    corr = "none" if k1 == k2 else "aic"
    v = vuong_test(l1, l2, k1, k2, correction=corr)

    cv1 = cv_loglik(rt, m1["fit"], m1["logpdf"], cv_folds, seed)
    cv2 = cv_loglik(rt, m2["fit"], m2["logpdf"], cv_folds, seed)

    b1 = bootstrap_ks_generic(rt, m1["fit"], m1["ks"], m1["rvs"], B, seed)
    b2 = bootstrap_ks_generic(rt, m2["fit"], m2["ks"], m2["rvs"], B, seed + 1)

    if v["p"] < 0.05:
        verdict = m1["name"] if v["z"] > 0 else m2["name"]
    else:
        verdict = "indistinguishable"

    return {
        "n": n,
        "m1": m1["name"], "m2": m2["name"],
        "ll1": ll1, "ll2": ll2,
        "k1": k1, "k2": k2,
        "aic1": aic(ll1, k1), "aic2": aic(ll2, k2),
        "bic1": bic(ll1, k1, n), "bic2": bic(ll2, k2, n),
        "cv_ll1": cv1, "cv_ll2": cv2,
        "cv_favours": (m1["name"] if cv1 > cv2 else m2["name"]),
        "vuong_z": v["z"], "vuong_p": v["p"], "vuong_correction": corr,
        "ks1": b1["ks"], "ks1_p": b1["ks_p"], "ks1_rejected": b1["rejected"],
        "ks2": b2["ks"], "ks2_p": b2["ks_p"], "ks2_rejected": b2["rejected"],
        "both_rejected": bool(b1["rejected"] and b2["rejected"]),
        "neither_rejected": bool(not b1["rejected"] and not b2["rejected"]),
        "verdict": verdict,
    }


def summarise(res, indent="    "):
    print(f"{indent}{res['m1']} (k={res['k1']})  vs  {res['m2']} (k={res['k2']})   "
          f"n={res['n']}")
    print(f"{indent}  log-lik      {res['ll1']:10.3f}   {res['ll2']:10.3f}")
    print(f"{indent}  AIC          {res['aic1']:10.3f}   {res['aic2']:10.3f}")
    print(f"{indent}  BIC          {res['bic1']:10.3f}   {res['bic2']:10.3f}")
    print(f"{indent}  CV log-lik   {res['cv_ll1']:10.3f}   {res['cv_ll2']:10.3f}"
          f"   -> {res['cv_favours']}")
    print(f"{indent}  KS           {res['ks1']:10.4f}   {res['ks2']:10.4f}")
    print(f"{indent}  KS boot p    {res['ks1_p']:10.4f}   {res['ks2_p']:10.4f}")
    print(f"{indent}  Vuong z = {res['vuong_z']:+.3f}, p = {res['vuong_p']:.4f} "
          f"({res['vuong_correction']} corrected)")
    print(f"{indent}  VERDICT: {res['verdict']}")
    if res["both_rejected"]:
        print(f"{indent}  !! both models are rejected in absolute terms -- a relative")
        print(f"{indent}     win here does not mean the winner is adequate.")
