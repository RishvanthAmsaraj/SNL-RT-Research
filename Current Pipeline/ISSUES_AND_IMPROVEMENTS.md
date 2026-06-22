# Issues & Improvement Opportunities

## Current Pipeline (v3.0)

This document tracks known issues, planned improvements, and potential
enhancements for the current production pipeline. Items are categorized by
priority and estimated effort.

---

## 🔴 Critical (Address Before Publication)

### 1. Research Paper Citations
**Status:** Pending | **Effort:** Medium

The pipeline implements several methods from published literature but lacks
proper inline citations:

- [ ] Add DOI links to all cited papers in script headers
- [ ] Create a `REFERENCES.bib` file for LaTeX integration
- [ ] Verify all parameter bounds against original sources
- [ ] Add page numbers for specific claims (e.g., Ratcliff & Tuerlinckx bounds)

**Files affected:** All Bayesian scripts, `DDM_fit.py`

---

### 2. Parameter Recovery Validation
**Status:** Not implemented | **Effort:** High

No systematic parameter recovery simulations have been run:

- [ ] Generate synthetic data with known parameters
- [ ] Test recovery accuracy across parameter space
- [ ] Document recovery bias (if any) for each parameter
- [ ] Add recovery validation script to repository

**Priority justification:** Essential for establishing method validity in
peer review.

---

## 🟡 High Priority (Improve Robustness)

### 3. Cross-Validation Framework
**Status:** Not implemented | **Effort:** High

Currently no out-of-sample validation:

- [ ] Implement leave-one-participant-out cross-validation
- [ ] Compare predictive accuracy: DDM vs Bayesian
- [ ] Report log-likelihood on held-out data
- [ ] Add CV script to `Verification Code/`

---

### 4. Prior Sensitivity Analysis
**Status:** Partial | **Effort:** Medium

Bayesian results depend on prior choices:

- [ ] Systematically vary prior centers (±1 SD)
- [ ] Document which estimates are prior-sensitive
- [ ] Add prior sensitivity report to output
- [ ] Consider weakly-informative vs informative prior comparison

**Current status:** SRT t₀ prior sensitivity partially documented in
`Bayesian_SRT_ndt.py` comments.

---

### 5. Model Comparison Metrics
**Status:** Partial | **Effort:** Medium

- [ ] Compute WAIC/LOO for Bayesian models (ArviZ)
- [ ] Compare single vs mixture models per cell
- [ ] Document model selection criteria
- [ ] Add comparison table to output

---

## 🟢 Medium Priority (Enhance Usability)

### 6. Configuration File Support
**Status:** Not implemented | **Effort:** Low

Hard-coded constants scattered across scripts:

- [ ] Create `config.yaml` with all tunable parameters
- [ ] Load config at runtime instead of hard-coding
- [ ] Document each parameter's purpose and valid range
- [ ] Add config validation on startup

**Example parameters to externalize:**
```yaml
fitting:
  hrt_floor_ms: 100
  srt_floor_ms: 70
  contamination: 0.05
  ks_threshold: 0.10
  
bayesian:
  draws: 1500
  tune: 1500
  chains: 4
  target_accept: 0.95
  
mixture:
  min_component_weight: 0.10
  max_component_weight: 0.90
  min_mode_separation_ms: 30
```

---

### 7. Progress Logging
**Status:** Minimal | **Effort:** Low

- [ ] Add structured logging (JSON format)
- [ ] Log per-cell fit time, convergence status, warnings
- [ ] Create processing report after pipeline completion
- [ ] Add progress bars for long-running operations

---

### 8. Unit Tests
**Status:** None | **Effort:** High

- [ ] Test Wald PDF/CDF against analytic solutions
- [ ] Test mixture selection logic with synthetic bimodal data
- [ ] Test hierarchical model on simulated data
- [ ] Add GitHub Actions CI for automated testing

---

### 9. Documentation Improvements
**Status:** Ongoing | **Effort:** Medium

- [ ] Add mathematical derivation of single-boundary Wald
- [ ] Document the non-centered parametrization (why it matters)
- [ ] Add FAQ section to README
- [ ] Create video tutorial for pipeline execution

---

## 🔵 Low Priority (Nice to Have)

### 10. Parallel Processing Optimization
**Status:** Partial (PyMC parallelizes chains) | **Effort:** Medium

- [ ] Parallelize DDM fitting across cells (embarrassingly parallel)
- [ ] Add multiprocessing support to `DDM_fit.py`
- [ ] Benchmark speedup on multi-core systems

---

### 11. Interactive Visualization
**Status:** Not implemented | **Effort:** High

- [ ] Create Jupyter notebook with interactive plots
- [ ] Add slider for prior parameter exploration
- [ ] Enable participant-level drill-down
- [ ] Export to HTML for sharing

---

### 12. Data Format Flexibility
**Status:** Fixed format | **Effort:** Medium

Currently requires exact column names in `pooled_data.csv`:

- [ ] Add column name mapping config
- [ ] Support alternative input formats (Excel, MATLAB .mat)
- [ ] Validate input schema before processing
- [ ] Provide clear error messages for missing columns

---

## 📊 Performance Benchmarks

### Current Runtime (M2 MacBook Air, 2024)

| Step | Runtime | Bottleneck |
|------|---------|------------|
| DDM_fit.py | ~2 min | Differential evolution per cell |
| Bayesian_HRT_fit.py | ~8 min | NUTS sampling (4 chains × 1500 draws) |
| Bayesian_SRT_fit.py | ~12 min | Mixture models + per-speed hierarchies |
| Bayesian_SRT_ndt.py | ~10 min | Participant-level t₀ sampling |
| Figure generation | ~1 min | Matplotlib rendering |
| **Total** | **~33 min** | **Bayesian sampling** |

### Target Improvements

| Optimization | Estimated Speedup |
|-------------|-------------------|
| Parallel DDM fitting | 3-4× |
| Reduced draws for exploration | 2× (with quality trade-off) |
| GPU sampling (PyMC JAX backend) | 5-10× |

---

## 🐛 Known Bugs

### Bug 1: Font Fallback on Linux
**Status:** Workaround exists | **Priority:** Low

Arial font not available on Linux systems:
```python
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
```

**Fix:** Better font detection or bundled font file.

---

### Bug 2: Large File Warnings
**Status:** Ignored | **Priority:** Low

`Bayesian_hrt_posterior.nc` can exceed 100 MB:
```python
try: idata.to_netcdf(...)
except Exception: pass  # Silently fails on disk full
```

**Fix:** Check disk space before write; compress netCDF.

---

### Bug 3: Mixture Label Switching
**Status:** Handled | **Priority:** Medium

Bayesian mixture models suffer from label switching (component identities
swap across chains). Currently handled by post-hoc relabeling by mode:

```python
m1 = t01_ + a1_/v1_; m2 = t02_ + a2_/v2_
exp_is1 = m1 <= m2  # express = faster component
```

**Potential improvement:** Use constrained priors or ordered transform.

---

## 📝 Contribution Guidelines

When addressing items in this list:

1. Create a feature branch: `git checkout -b feature/issue-N-description`
2. Update this document to mark item as "In Progress"
3. Add tests if applicable
4. Update CHANGELOG.md
5. Submit pull request with before/after benchmarks

---

## 📅 Timeline

| Quarter | Focus |
|---------|-------|
| Q3 2024 | Critical items (citations, recovery validation) |
| Q4 2024 | High priority (CV, prior sensitivity) |
| Q1 2025 | Medium priority (config, logging, tests) |
| Q2 2025 | Low priority (interactive viz, optimizations) |

---

## Contact

For questions about priorities or to suggest new items, open a GitHub issue
or contact the maintainer.

**Maintainer:** Rishvanth Amsaraj  
**Last Updated:** June 2024
