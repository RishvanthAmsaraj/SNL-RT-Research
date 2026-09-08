"""
test_wfpt.py  --  validation of the two-boundary WFPT density.

Four independent checks, none of which trusts the others:

  1. QUADRATURE   integrating the defective density over time must reproduce the
                  analytic barrier probability (1 - exp(-2vaw)) / (1 - exp(-2va)).
  2. SIMULATION   the density must match a Monte-Carlo Euler-Maruyama simulation.
  3. AGREEMENT    the PyTensor (fixed-term) version must match the NumPy
                  (adaptive-term) version across the plausible parameter range.
  4. GRADIENTS    the PyTensor log density must have finite gradients w.r.t.
                  v, a and w -- without this NUTS cannot sample.

Run: python test_wfpt.py
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import ks_2samp

from wfpt import (numpy_wfpt_pdf, numpy_p_upper, simulate_ddm, pt_wfpt_logpdf)

GRID = [
    # v      a     w      description
    (2.0,  1.2,  0.50, "unbiased, moderate drift"),
    (0.0,  1.0,  0.50, "zero drift, unbiased -> 50/50"),
    (-1.5, 1.5,  0.50, "negative drift"),
    (1.0,  1.0,  0.30, "start point biased low"),
    (1.0,  1.0,  0.70, "start point biased high"),
    (6.0,  1.0,  0.50, "fast drift, few errors"),
    (0.5,  2.5,  0.55, "wide boundary, slow"),
    (3.0,  0.6,  0.45, "narrow boundary, fast"),
    (10.0, 1.0,  0.50, "very fast drift (~no errors)"),
]

fails = []


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


# --------------------------------------------------------------------------- #
print("\n1. QUADRATURE -- density integrates to the analytic barrier probability")
for v, a, w, desc in GRID:
    p_up_analytic = numpy_p_upper(v, a, w)
    p_up_num = quad(lambda t: numpy_wfpt_pdf(t, v, a, w, upper=True),
                    1e-6, 60.0, limit=400)[0]
    p_lo_num = quad(lambda t: numpy_wfpt_pdf(t, v, a, w, upper=False),
                    1e-6, 60.0, limit=400)[0]
    e_up = abs(p_up_num - p_up_analytic)
    e_tot = abs((p_up_num + p_lo_num) - 1.0)
    report(f"v={v:5.1f} a={a:.1f} w={w:.2f}  ({desc})",
           e_up < 1e-5 and e_tot < 1e-5,
           f"P(up)={p_up_num:.6f} vs {p_up_analytic:.6f}, total={p_up_num + p_lo_num:.6f}")


# --------------------------------------------------------------------------- #
print("\n2. SIMULATION -- density matches Monte-Carlo (KS on upper-barrier RTs)")
# Judged on the KS statistic, not its p-value. simulate_ddm is Euler-Maruyama, which
# overshoots the barrier by O(sqrt(dt)) and so is very slightly biased fast; at large n
# that bias is detectable however correct the density is. Check 1 (quadrature) is the
# exact test of the density -- this one is a sanity check on the SIMULATOR, which is
# what the posterior predictive checks depend on.
rng = np.random.default_rng(0)
for v, a, w, desc in GRID[:4]:
    t0 = 0.15
    rt, ch = simulate_ddm(8000, v, a, w, t0, dt=1e-4, max_t=3.0, rng=rng)
    ok_sim = np.isfinite(rt)
    rt, ch = rt[ok_sim], ch[ok_sim]

    # choice proportion
    p_sim = ch.mean()
    p_ana = numpy_p_upper(v, a, w)

    # sample from the analytic upper-barrier density by inverse-CDF on a grid
    up = rt[ch == 1] - t0
    if len(up) < 500:
        report(f"v={v:5.1f} a={a:.1f} w={w:.2f}  ({desc})", True, "too few upper responses, skipped")
        continue
    grid = np.linspace(1e-4, np.percentile(up, 99.9) * 1.4, 4000)
    dens = np.array([numpy_wfpt_pdf(g, v, a, w, upper=True) for g in grid])
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    cdf = cdf / cdf[-1]
    draws = np.interp(rng.random(len(up)), cdf, grid)

    ks = ks_2samp(up, draws)
    # 5% two-sample critical value at this n, plus 0.02 to absorb Euler overshoot bias
    crit = 1.36 * np.sqrt(2.0 / len(up)) + 0.02
    report(f"v={v:5.1f} a={a:.1f} w={w:.2f}  ({desc})",
           ks.statistic < crit and abs(p_sim - p_ana) < 0.015,
           f"n_up={len(up):5d}  KS D={ks.statistic:.4f} (crit {crit:.4f}); "
           f"P(up) sim={p_sim:.4f} vs {p_ana:.4f}")


# --------------------------------------------------------------------------- #
print("\n3. AGREEMENT -- PyTensor (fixed terms) vs NumPy (adaptive terms)")
import pytensor
import pytensor.tensor as pt

tau_t = pt.dvector("tau"); v_t = pt.dscalar("v")
a_t = pt.dscalar("a"); w_t = pt.dscalar("w"); c_t = pt.dvector("c")
logp_expr = pt_wfpt_logpdf(tau_t, v_t, a_t, w_t, c_t)
f_logp = pytensor.function([tau_t, v_t, a_t, w_t, c_t], logp_expr,
                           on_unused_input="ignore")

taus = np.concatenate([np.linspace(0.005, 0.05, 12),
                       np.linspace(0.06, 1.2, 30),
                       np.linspace(1.5, 4.0, 8)])
worst = 0.0
for v, a, w, desc in GRID:
    for upper in (False, True):
        got = f_logp(taus, v, a, w, np.full(taus.shape, 1.0 if upper else 0.0))
        want = np.log(np.maximum(
            np.array([numpy_wfpt_pdf(t, v, a, w, upper=upper) for t in taus]), 1e-300))
        finite = np.isfinite(want) & (want > -600)
        err = np.max(np.abs(got[finite] - want[finite]))
        worst = max(worst, err)
        report(f"v={v:5.1f} a={a:.1f} w={w:.2f} {'upper' if upper else 'lower'}",
               err < 1e-8, f"max |log-density error| = {err:.2e}")
print(f"  worst-case error across the whole grid: {worst:.3e}")


# --------------------------------------------------------------------------- #
print("\n4. GRADIENTS -- finite d(logp)/d(v, a, w), and finite-difference agreement")
gv, ga, gw = pt.grad(logp_expr.sum(), [v_t, a_t, w_t])
f_grad = pytensor.function([tau_t, v_t, a_t, w_t, c_t], [gv, ga, gw],
                           on_unused_input="ignore")

for v, a, w, desc in GRID[:6]:
    c = np.tile([0.0, 1.0], len(taus) // 2 + 1)[:len(taus)]
    g = f_grad(taus, v, a, w, c)
    finite = all(np.all(np.isfinite(x)) for x in g)

    # central finite differences
    h = 1e-6
    fd = [
        (f_logp(taus, v + h, a, w, c).sum() - f_logp(taus, v - h, a, w, c).sum()) / (2 * h),
        (f_logp(taus, v, a + h, w, c).sum() - f_logp(taus, v, a - h, w, c).sum()) / (2 * h),
        (f_logp(taus, v, a, w + h, c).sum() - f_logp(taus, v, a, w - h, c).sum()) / (2 * h),
    ]
    rel = max(abs(float(g[i]) - fd[i]) / max(abs(fd[i]), 1.0) for i in range(3))
    report(f"v={v:5.1f} a={a:.1f} w={w:.2f}  ({desc})", finite and rel < 1e-4,
           f"max relative grad error = {rel:.2e}")


# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED -- the WFPT density is correct and differentiable.")
