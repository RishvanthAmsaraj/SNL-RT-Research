# Changelog — SNL RT Research Pipeline

## Version History & Evolution

This document traces the evolution of the analysis pipeline from initial prototypes
to the current production-ready system. Each version represents a major methodological
advancement driven by empirical validation and peer review.

---

## Current Pipeline (v3.0) — Active

**Status:** Production-ready | **Date:** June 2024

### Key Advancements

1. **Hierarchical Bayesian Framework**
   - Replaces per-cell maximum-likelihood fitting with partial-pooling Bayesian models
   - Resolves non-decision time (t₀) floor-piling through principled regularization
   - Returns full posterior credible intervals for all parameters
   - Handles fast responders correctly (wide CIs instead of hard floors)

2. **SRT Non-Decision Time: Participant-Level Estimation**
   - **Problem:** Per-cell SRT fitting cannot identify t₀ — estimates slide to whatever
     floor is imposed (verified: 19/33 cells floor at 70 ms even with Bayesian per-cell)
   - **Solution:** Hierarchical model estimates one t₀ per participant, shared across
     all three target speeds. Triples the trial count informing each estimate and
     enforces cross-speed consistency.
   - **Result:** 0/16 participants floored (vs ~50% with per-cell); population mean
     ~60 ms with honest individual differences preserved

3. **Express/Regular Saccade Mixture (Data-Driven Selection)**
   - Single Wald fit attempted first; mixture only adopted when single fails (KS > 0.10)
   - Mixture adoption requires: KS < 0.10, substantial components (0.10 ≤ π ≤ 0.90),
     and well-separated modes (≥ 30 ms apart)
   - Avoids BIC over-detection and dip-test under-detection

4. **Methodological Rigor**
   - Literature-grounded parameter bounds (Ratcliff & Tuerlinckx 2002)
   - 5% uniform contamination term for robustness
   - Convergence diagnostics (divergences, R-hat) on every fit
   - Resumable fitting (no work lost on interruption)

### Files
- `Bayesian/Bayesian_HRT_fit.py` — Hierarchical HRT model
- `Bayesian/Bayesian_SRT_fit.py` — Hierarchical SRT model (per-speed)
- `Bayesian/Bayesian_SRT_ndt.py` — Participant-level SRT t₀ estimation
- `Bayesian/Bayesian_figures.py` — Publication-quality summary figures
- `Bayesian/Bayesian_conceptual.py` — Process schematics
- `DDM/DDM_fit.py` — Frequentist comparison (Method A)
- `DDM/DDM_figures.py`, `DDM_conceptual.py` — DDM diagnostics
- `NDT/NDT_barchart.py`, `NDT_barchart_bayesian.py` — NDT visualizations
- `Vincentile/vincentile_figures.py` — Model-free distribution analysis

---

## Deprecated Ver 2 — Previous Lab Pipeline

**Status:** Superseded | **Period:** 2023–2024

### What Worked
- Correctly identified the need for per-participant fitting (vs pooled)
- Introduced contamination mixture for robustness
- Developed visualization pipeline for conceptual figures
- Created comprehensive diagnostic PDF (9 pages)

### Critical Issues Identified

1. **SRT t₀ Floor-Piling (Fundamental)**
   - Per-cell fitting cannot identify t₀ for fast saccades
   - Estimates pile at hard floor (70 ms) regardless of true value
   - ~50% of cells floored; no way to distinguish "genuinely fast" from "unidentifiable"
   - **Fix in v3:** Participant-level hierarchical estimation with partial pooling

2. **No Credible Intervals**
   - Maximum-likelihood returns point estimates only
   - Cannot quantify uncertainty or detect poorly identified parameters
   - **Fix in v3:** Full Bayesian posterior with 95% CIs

3. **Fixed t₀ Removes Individual Differences**
   - Alternative fix (fixing t₀ at constant) removes real variation between participants
   - **Fix in v3:** Hierarchical model preserves individual differences with honest uncertainty

4. **Hard-Coded Paths**
   - Absolute Windows paths (`C:\Users\Rishv\...`) throughout
   - Not portable across machines
   - **Fix in v3:** Relative paths using `SCRIPT_DIR`

5. **No Convergence Diagnostics**
   - Could not detect sampling failures or model misspecification
   - **Fix in v3:** R-hat and divergence tracking on every fit

6. **Bimodal Detection: BIC vs Dip-Test Tension**
   - BIC tends to over-detect mixtures (penalizes simplicity too strongly)
   - Hartigan's dip-test tends to under-detect (insensitive to small fast components)
   - **Fix in v3:** Fit-driven + structural validation (KS + component separation)

### Files
- `Deprecated Ver 2/Bayesian Model/` — Early Bayesian implementations
- `Deprecated Ver 2/DDM Model/` — Previous DDM analysis (superseded by current DDM/)
- `Deprecated Ver 2/NDT Code/` — Earlier NDT calculations
- `Deprecated Ver 2/Vincentile Code/` — Previous vincentile methods
- `Deprecated Ver 2/Verification Code/ddm_diagnostics.py` — Comprehensive diagnostics

---

## Deprecated Ver 1 — Initial PyDDM Prototypes

**Status:** Archived | **Period:** 2022–2023

### What It Was
- Proof-of-concept using the PyDDM library
- Synthetic data generation for validation
- Single-choice and dual-choice task implementations
- Basic visualization with matplotlib

### Critical Issues Identified

1. **PyDDM Dependency**
   - External library with limited flexibility
   - Difficult to customize for specific experimental paradigms
   - Slower than native implementations on large datasets
   - **Fix in v2/v3:** Native scipy optimization with custom likelihoods

2. **Synthetic Data Only**
   - No real Kinarm data processing pipeline
   - Hard-coded parameters in data generators
   - **Fix in v2:** Real data loading from pooled_data.csv

3. **No Statistical Validation**
   - No goodness-of-fit metrics (KS, etc.)
   - No parameter recovery simulations
   - **Fix in v2:** KS statistics, diagnostic PDFs, mixture validation

4. **Basic Visualization**
   - Simple histograms and trajectory plots
   - No publication-quality figures
   - **Fix in v2/v3:** Vector PDFs with editable text, condition colors, proper typography

5. **No Hierarchical Structure**
   - Each participant fitted in isolation
   - No sharing of information across participants
   - **Fix in v3:** Partial pooling with weakly informative priors

6. **Incorrect Model for Task**
   - Used two-boundary DDM (choice tasks) for interception (single-boundary)
   - **Fix in v2/v3:** Single-boundary Wald (first-passage) model

### Files
- `Deprecated Ver 1/DualChoice*.py` — Dual choice task models
- `Deprecated Ver 1/SingleChoice*.py` — Single choice task models
- `Deprecated Ver 1/*DataGen.py` — Synthetic data generators

---

## Migration Guide

### From Ver 1 → Ver 2
- Replace PyDDM with native scipy optimization
- Switch from two-boundary to single-boundary (Wald) model
- Add real data loading from CSV
- Implement KS goodness-of-fit
- Add contamination mixture

### From Ver 2 → Current (Ver 3)
- Replace MLE with hierarchical Bayesian (PyMC)
- Move SRT t₀ from per-cell to per-participant estimation
- Add credible intervals to all reports
- Implement resumable fitting
- Add convergence diagnostics
- Replace BIC/dip-test with fit-driven mixture selection

---

## Citation

If you use this pipeline, please cite the version appropriate to your analysis:

**Current Pipeline (Bayesian):**
```
Amsaraj, R. (2024). SNL RT Research Pipeline v3.0 — Hierarchical Bayesian
Drift Diffusion Models for KINARM Interception Tasks.
Sensorimotor Neuroscience Laboratory.
```

**Earlier Versions:**
See individual script headers for method-specific citations.

---

## References

- Ratcliff, R. & Tuerlinckx, F. (2002). Estimating parameters of the diffusion model:
  Approaches to dealing with contaminant reaction times and parameter variability.
  *Psychonomic Bulletin & Review*, 9(3), 438–481.

- Wiecki, T.V., Sofer, I. & Frank, M.J. (2013). HDDM: Hierarchical Bayesian estimation
  of the Drift-Diffusion Model in Python. *Frontiers in Neuroinformatics*, 7, 14.

- Gelman, A. et al. *Bayesian Data Analysis* (3rd ed.). CRC Press.

- Shinn, M., Lam, N.H. & Murray, J.D. (2020). A flexible framework for simulating
  and fitting generalized drift-diffusion models. *eLife*, 9, e56938. (PyDDM)
