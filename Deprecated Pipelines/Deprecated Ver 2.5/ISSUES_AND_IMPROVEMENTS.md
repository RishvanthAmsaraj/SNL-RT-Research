# SNL RT Research — Deprecated Ver 2.5 Issues & Improvements

**Why this version was superseded, and what replaced it.**

---

## Summary

**Deprecated Ver 2.5** was the bridge between DDM and full Bayesian — it introduced per-cell Bayesian models and improved the express-saccade detection but stopped short of the hierarchical approach needed to resolve the floor-piling. The saccadic t₀ still floored 19/33 cells even in the Bayesian fits, confirming the need for participant-level pooling.

**Superseded by:** [Deprecated Ver 3](../Deprecated%20Ver%203/)

---

## Problems

### 1. Per-cell SRT t₀ floors — Bayesian can't fix it

Even with Bayesian estimation, the per-cell SRT fit still floored **19 of 33 cells at 70 ms**. The problem is not DDM vs. Bayesian — it's per-cell vs. pooled. The information to separate non-decision time from decision time simply isn't in individual saccadic cells.

*Severity: Critical. Fixed in Ver 3 by pooling t₀ to the participant level.*

### 2. No full credible intervals

The per-cell Bayesian model produced point estimates but not full posterior credible intervals. Could not distinguish well-identified cells from poorly identified ones.

*Severity: Methodological. Fixed in Ver 3 with full posterior sampling and ArviZ diagnostics.*

### 3. No convergence diagnostics

No R-hat, no divergence tracking. Could not confirm that the per-cell Bayesian chains had converged (many had not — the floor-piling was a sign).

*Severity: Methodological. Fixed in Ver 3 with full R-hat, ESS, and divergence monitoring.*

### 4. SRT NDT floor too low (35 ms)

The per-participant SRT NDT model used a 35 ms floor — below the saccadic dead time (~70–90 ms). This produced overscattered, uninterpretable estimates.

*Severity: Methodological. Fixed in Ver 3 by raising to 70 ms, then definitively in Current Pipeline with the fixed-at-70 reporting.*

### 5. No structured pipeline

Like Ver 2, scripts were run individually with no defined workflow or output chaining.

*Severity: Operational. Fixed in Ver 3 with ordered pipeline and RUN_GUIDE.*

### 6. No narrative framework for results

The results existed but had no narrative framework for presentation — no way to explain the three-category figure split (Bayesian vs. DDM vs. vincentile) to readers.

*Severity: Communication. Fixed in Ver 3.*

---

## What Was Right (and Not Changed)

- **Fit-driven + structural mixture detection** — correct approach (fixed the BIC over-detection from Ver 2)
- **Fast/slow saccadic mixture model** — correct handling of express saccades
- **First Bayesian implementation** — established the architecture later scaled to full hierarchical
- **Relative paths** — fixed the portability issue from Ver 2

---

## Files In This Folder

- `DDM/DDM_fit.py` — DDM MLE (mature at this point)
- `DDM/DDM_figures.py` — Publication figures (mature)
- `Bayesian/Bayesian_HRT_fit.py` — First pass at hand Bayesian (per-cell)
- `Bayesian/Bayesian_SRT_fit.py` — Per-cell saccadic Bayesian with mixture
- `Bayesian/Bayesian_SRT_ndt.py` — Per-participant SRT NDT (35 ms floor, overscattered)
- `Vincentile/vincentile_figures.py` — Model-free figures (mature)
- `NDT/NDT_barchart.py`, `NDT_barchart_bayesian.py` — NDT visualization
- `RUN_GUIDE.md` — Structured run order
- `METHODOLOGICAL_JUSTIFICATION.md` — Detailed model selection and bound reasoning
- `ISSUES_AND_IMPROVEMENTS.md` — Original issues document (this folder)
- `MIGRATION_NOTES.md` — Transition notes to Ver 3
