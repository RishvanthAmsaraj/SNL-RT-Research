# Migration Notes: Deprecated Ver 2 → Current Pipeline

## Overview

This document details the specific technical and methodological improvements
that led from Ver 2 to the current production pipeline. It is intended for
researchers who need to understand why certain approaches were abandoned and
what replaced them.

---

## 1. SRT Non-Decision Time: The Floor-Piling Problem

### Ver 2 Approach
```python
# Per-cell fitting: each participant × speed fitted independently
def fit_wald_contamination(rts):
    t0_max = max(np.percentile(rts, 3) - 0.002, T0_MIN + 0.001)
    # ... optimize v, a, t0
```

**Problem:** For fast saccades (many cells with median SRT < 150 ms), the
three parameters (v, a, t₀) are not jointly identifiable. The optimizer slides
t₀ to whatever minimum is imposed, inflates v to compensate, and reports a
deceptively good fit.

**Evidence:**
- 19/33 SRT cells floored at 70 ms even with Bayesian per-cell fitting
- Changing t₀ floor from 50→70→90 ms shifted estimates proportionally
- KS statistics remained good despite incorrect t₀

### Current Pipeline Solution

**Hierarchical participant-level t₀:**
```python
# Bayesian_SRT_ndt.py
with pm.Model() as m:
    mu_t0 = pm.Normal("mu_t0", 0.070, 0.030)  # population mean
    sig_t0 = pm.HalfNormal("sig_t0", 0.025)   # between-participant SD
    t0_p = pm.TruncatedNormal("t0_p", mu=mu_t0, sigma=sig_t0,
                              lower=FLOOR, upper=minrt_arr-0.001,
                              shape=len(parts))
```

**Why this works:**
1. One t₀ per participant (shared across 3 speeds) = 3× more trials per estimate
2. Partial pooling: fast-saccade participants regularized toward population mean
3. Wide credible intervals honestly flag unidentifiable cases
4. Population mean mildly prior-dependent (documented transparently)

**Result:** 0/16 participants floored; individual differences preserved

---

## 2. From Point Estimates to Credible Intervals

### Ver 2 Approach
Maximum-likelihood via `scipy.optimize.differential_evolution`:
```python
res = differential_evolution(neg_ll, bounds, seed=42, maxiter=1500)
v, a, t0 = res.x  # Single best-fit value only
```

**Problems:**
- No uncertainty quantification
- Cannot distinguish well-identified from poorly identified parameters
- Local optima possible (addressed with multi-start but not eliminated)

### Current Pipeline Solution

**Bayesian posterior sampling:**
```python
with pm.Model() as m:
    # ... model definition ...
    idata = pm.sample(1500, tune=1500, chains=4, cores=4,
                      target_accept=0.95, random_seed=7)
    
# Posterior summary
v_mean = idata.posterior["v"].mean(("chain", "draw"))
v_ci_lo = idata.posterior["v"].quantile(0.025, ("chain", "draw"))
v_ci_hi = idata.posterior["v"].quantile(0.975, ("chain", "draw"))
```

**Advantages:**
- Full uncertainty propagation
- R-hat detects non-convergence (R-hat > 1.01 = problem)
- Divergence tracking flags model/data mismatches
- Can report "estimate is regularized (wide CI)" honestly

---

## 3. Mixture Model Selection

### Ver 2 Approach
Relied on BIC or Hartigan's dip-test for bimodal detection:
```python
# BIC comparison (tends to over-penalize simplicity)
# OR
import diptest
dip_p = diptest.diptest(rts)[1]
if dip_p < 0.05:
    # Flag as bimodal
```

**Problems:**
- BIC: penalizes single-component models too strongly → false positives
- Dip-test: insensitive to small fast components → false negatives
- No structural validation of recovered components

### Current Pipeline Solution

**Fit-driven + structural validation:**
```python
# 1. Fit single Wald first
xs, _, ks_s = fit_single(rts, SRT_FLOOR, P_CONTAM)

# 2. Only consider mixture if single fails
if ks_s > 0.10:
    xm, _, ks_m = fit_mixture(rts, SRT_FLOOR, P_CONTAM)
    pi = xm[0]
    em = (xm[3] + xm[2]/xm[1]) * 1000  # express mode
    rm = (xm[6] + xm[5]/xm[4]) * 1000  # regular mode
    
    # 3. Adopt only if ALL criteria met
    use_mix = (ks_m < 0.10) and \
              (0.10 <= pi <= 0.90) and \
              ((rm - em) >= 30)  # well-separated modes
```

**Advantages:**
- Conservative: only adopts mixture when single demonstrably fails
- Structural checks prevent degenerate solutions (π ≈ 0 or 1)
- Mode separation ensures components are psychologically meaningful

---

## 4. Path Handling

### Ver 2 Issue
```python
# Hard-coded absolute paths
CSV_PATH = r'C:\Users\Rishv\Desktop\SNL Lab\...'
```

**Problem:** Not portable. Breaks on any machine except the original.

### Current Pipeline Solution
```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "pooled_data.csv")
```

**Advantage:** Works on any machine; relative to script location.

---

## 5. Code Organization

### Ver 2 Structure
```
Deprecated Ver 2/
├── Bayesian Model/          # Early Bayesian (per-cell, no SRT_ndt)
├── DDM Model/
│   ├── Deprecated/
│   │   ├── Extra/           # Kinarm processor, READMEs
│   │   ├── HRT/             # Separate HRT script
│   │   ├── SRT/             # Separate SRT script
│   │   └── PyDDM/           # Even older PyDDM code
│   ├── DDM_fit.py           # (same as current)
│   └── DDM_figures.py       # (similar to current)
├── NDT Code/
├── Vincentile Code/
└── Verification Code/
```

**Problems:**
- Deep nesting of deprecated code
- Multiple versions of same functionality
- No clear separation of concerns

### Current Pipeline Structure
```
Current Pipeline/
├── Bayesian/               # All Bayesian models
├── DDM/                    # Frequentist comparison
├── NDT/                    # NDT visualizations
├── Vincentile/             # Model-free analysis
└── RUN_GUIDE.md            # Single entry point
```

**Advantages:**
- Flat, logical organization
- Each folder self-contained
- Clear run order in RUN_GUIDE.md

---

## 6. Key Files Retained from Ver 2

The following files were carried forward essentially unchanged:

| File | Status | Notes |
|------|--------|-------|
| `DDM_fit.py` | Unchanged | Already mature in Ver 2 |
| `DDM_figures.py` | Minor updates | Added Bayesian comparison panels |
| `vincentile_figures.py` | Unchanged | Model-free, no methodology change |

---

## 7. Files Deliberately Abandoned

| File | Reason |
|------|--------|
| `ddm_fit_final.py` | Superseded by `DDM_fit.py` (same core, better organized) |
| `DDM HRT.py` / `DDM SRT.py` | Merged into unified `DDM_fit.py` |
| `ddm_diagnostics.py` | Superseded by `DDM_figures.py` + `Bayesian_figures.py` |
| `DDM_kinarm_multi_trial_processor.py` | One-time data preprocessing; not part of analysis pipeline |
| Per-cell Bayesian SRT | Replaced by `Bayesian_SRT_ndt.py` (participant-level t₀) |

---

## Summary Table: Ver 2 vs Current

| Aspect | Ver 2 | Current (v3) |
|--------|-------|--------------|
| **Framework** | Maximum likelihood | Hierarchical Bayesian |
| **SRT t₀** | Per-cell (floored) | Per-participant (shared across speeds) |
| **Uncertainty** | None | Full posterior CIs |
| **Convergence** | Not checked | R-hat + divergences |
| **Mixture selection** | BIC / dip-test | Fit-driven + structural |
| **Portability** | Hard-coded paths | Relative paths |
| **Organization** | Deep nesting | Flat, logical |
| **Runtime** | ~3-6 min | ~10-15 min (Bayesian) |
| **Output** | Point estimates | Estimates + CIs + diagnostics |

---

## Recommendation

**For new analyses:** Use the Current Pipeline exclusively.

**For reproducing earlier results:** Ver 2 outputs are preserved in the
repository history. The DDM (Method A) outputs from `DDM_fit.py` are
intentionally identical between Ver 2 and Current for direct comparison.

**For understanding methodology:** Read `RUN_GUIDE.md` in the Current Pipeline
for the definitive description of the recommended approach.
