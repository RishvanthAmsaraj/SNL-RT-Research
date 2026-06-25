# SNL RT Research — Issues & Improvements (Current Pipeline v3.0)

**Known limitations, planned improvements, and architectural notes for the active production pipeline.**

---

## Status

The Current Pipeline reflects the completed evolution through Phases 0–2 (see [`DEVELOPMENT_HISTORY.md`](../DEVELOPMENT_HISTORY.md)). It is considered **stable for internal reporting** with the following caveats.

---

## Known Limitations

### 1. Sampling settings: reduced-chain preview

The preview CSVs and figures distributed with the repository were generated with reduced MCMC sampling (≈600 draws / 600 tune / 4 chains) for speed. Point estimates match the full run exactly — HRT t₀ values are identical at 168/156/147 — but final paper numbers should come from a full-draw run (1500 draws / 1500 tune / 4 chains). This is clearly documented in `RUN_GUIDE.md`.

### 2. The saccadic NDT collapse

The per-participant saccadic NDT model (`Bayesian_SRT_ndt.py`) collapses to the floor (70 ms) for all participants when the floor is applied. This is **not a bug** — it is a diagnosis that saccadic t₀ is not identifiable above the physiological floor. The decision to report saccadic t₀ as fixed at 70 ms is handled transparently, but future work could explore alternative saccade-specific models.

### 3. Separate conda environment requirement

The Bayesian scripts require a separate conda environment due to PyMC's compilation dependency on non-pip packages. The `RUN_GUIDE.md` documents the workaround, but it adds friction. Containerization (Docker/Singularity) would be a clean solution.

### 4. DDM diagnostic consistency

The DDM figures are labeled as "comparison/diagnostic" but some viewers may mistake them for primary results. The narrative framing addresses this explicitly, but it requires reader discipline.

### 5. No cross-validation

The pipeline does not include a cross-validation framework (e.g., LOO-CV, WAIC). The current diagnostics (R-hat, divergence counts, KS statistics) confirm convergence and fit quality but do not quantify out-of-sample predictive performance.

### 6. Individual trial-level pooling

The model pools parameters at the participant level but does not model trial-level effects (e.g., trial order, within-session fatigue). These are assumed to be averaged out across 7,676 trials.

---

## What Was Fixed in This Version

Compared to Deprecated Ver 3:

- **Knox & Wolohan DOI** corrected (`e0133595` → `e0120437`)
- **V_MAX reduced** from 40 to 20 (Tran 2020 literature anchor)
- **Hand t₀ floor raised** from 100 to 130 ms (Haith 2016)
- **SRT per-participant NDT floor raised** from 35 to 70 ms (saccadic dead time)
- **v/a bounds re-attributed** from Ratcliff & Tuerlinckx to Tran 2020
- Figure refinements: NDT chart zoomed y-axes, bar→point switch, floor line labels corrected
- **LATER model** added as complementary saccade analysis
- **`why_saccadic_t0_floors.py`** — mechanism diagnostic added
- Saccadic t₀ reported as **fixed at 70 ms**, not estimated per participant

---

## Planned Improvements (Not Yet Implemented)

### High priority

- [ ] Full-draw MCMC run with 1500/1500/4 chains for final publication figures
- [ ] Dockerfile for reproducible environment (eliminate conda/pip split)

### Medium priority

- [ ] LOO-CV or WAIC for cross-validation
- [ ] Bootstrap or permutation tests for the dissociation result (supplementing Friedman)
- [ ] Sensitivity analysis on the express-saccade mixture threshold

### Low priority / future work

- [ ] Trial-level covariate modelling (trial order, inter-trial interval)
- [ ] Re-analysis with a dedicated saccade model (LATER) as primary for saccades
- [ ] Expand to additional KINARM tasks (if data becomes available)

---

## Dependencies & Versions

| Package | Version | Notes |
|---|---|---|
| Python | ≥3.8 | 3.11 recommended for PyMC stability |
| scipy | ≥1.7 | MLE optimization |
| pandas | ≥1.3 | Data handling |
| matplotlib | ≥3.4 | Figures (300 DPI, pdf.fonttype=42) |
| PyMC | ≥5.0 | Bayesian models (conda only) |
| ArviZ | ≥0.15 | Posterior diagnostics |
| scikit-learn | ≥1.0 | Diptest for bimodality |
| numpy | ≥1.21 | Numerical backbone |
| numba | ≥0.55 | Optional speed-up for DDM likelihood |
