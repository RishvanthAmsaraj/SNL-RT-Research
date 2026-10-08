"""Two-boundary DDM with across-trial DRIFT VARIABILITY (sv).

The plain 4-parameter DDM cannot produce enough right skew in the marginal RT
distribution, which is where it was losing. Drift variability is the standard fix:
integrating the WFPT density over v ~ Normal(v, sv) lengthens the right tail.

    f(t,c | v,sv,a,w,t0) = INTEGRAL f_wfpt(t,c | nu,a,w,t0) * N(nu; v,sv) dnu

evaluated by Gauss-Hermite quadrature.
"""
import sys, numpy as np
sys.path.insert(0,'/home/claude/two_boundary')
from wfpt import numpy_wfpt_logpdf_vec
from scipy.optimize import minimize
from scipy.special import logsumexp

_X,_W = np.polynomial.hermite_e.hermegauss(15)
_LOGW = np.log(_W/np.sqrt(2*np.pi))

def loglik_sv(rt, ch, v, sv, a, w, t0):
    tau = rt - t0
    if np.any(tau <= 0): return -np.inf
    if sv < 1e-6:
        lp = numpy_wfpt_logpdf_vec(tau, v, a, w, ch)
        return -np.inf if not np.all(np.isfinite(lp)) else float(lp.sum())
    comps = np.empty((len(_X), len(rt)))
    for k, x in enumerate(_X):
        comps[k] = numpy_wfpt_logpdf_vec(tau, v + sv*x, a, w, ch) + _LOGW[k]
    tot = logsumexp(comps, axis=0)
    return -np.inf if not np.all(np.isfinite(tot)) else float(tot.sum())

def fit_sv(rt, ch, start, floor=0.130):
    """start = (v,a,w,t0) from the plain 2B fit. Returns (v,sv,a,w,t0), loglik."""
    hi = max(np.percentile(rt,3)-0.002, floor+1e-3)
    def unpack(z):
        v = z[0]; sv = np.exp(z[1]); a = np.exp(z[2])
        w = 1/(1+np.exp(-z[3])); t0 = floor + (hi-floor)/(1+np.exp(-z[4]))
        return v, sv, a, w, t0
    def nll(z):
        ll = loglik_sv(rt, ch, *unpack(z))
        return 1e10 if not np.isfinite(ll) else -ll
    v0,a0,w0,t00 = start
    frac = np.clip((t00-floor)/(hi-floor), .02, .98)
    best=None
    for sv0 in (0.3, 1.0, 2.0):
        z0=np.array([v0, np.log(sv0), np.log(a0), np.log(w0/(1-w0)), np.log(frac/(1-frac))])
        for meth in ("Nelder-Mead","Powell"):
            r=minimize(nll,z0,method=meth,options=dict(maxiter=6000,maxfev=6000,xatol=1e-6,fatol=1e-8))
            if best is None or r.fun<best.fun: best=r
            z0=r.x
    return unpack(best.x), -best.fun
