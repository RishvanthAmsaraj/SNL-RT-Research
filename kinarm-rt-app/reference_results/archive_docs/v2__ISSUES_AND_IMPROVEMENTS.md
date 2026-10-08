# SNL RT Research — Deprecated Ver 2 Issues & Improvements

**Why this version was superseded, and what replaced it.**

---

## Summary

**Deprecated Ver 2** was the first real production pipeline — native scipy MLE of the shifted Wald on `pooled_data.csv`, with the contamination mixture, correct data filters, and publication-quality figures. It remains structurally sound for DDM (frequentist) analysis.

**However**, the central problem — saccadic t₀ floor-piling — motivated the move to Bayesian methods. The DDM result (p = 0.047 for the HRT speed effect) was fragile, resting entirely on three floored cells.

**Superseded by:** [Deprecated Ver 2.5](../Deprecated%20Ver%202.5/)

---

## Problems

### 1. SRT t₀ floor-piling (the central problem)

Per-cell saccadic t₀ estimates piled at the floor (70 ms). This is an identifiability limit of the Wald on near-symmetric saccadic distributions — not solvable by tuning within the MLE framework.

*Severity: Critical (makes SRT analysis unreliable). First addressed in Ver 2.5 with per-cell Bayesian models; fully fixed in Ver 3 with participant-level hierarchical pooling.*

### 2. Fragile DDM HRT result

The hand speed effect (p = 0.047) rested on three floored cells (`CMT001`, `CMT002`, `CMT010` at 150 deg/s). Dropping them gave p = 0.199. Not publishable.

*Severity: Critical. Fixed in Ver 3 with hierarchical Bayesian estimation (p = 0.0016, 0 floored cells).*

### 3. No hierarchical structure

Each participant fitted independently. No partial pooling to regularize noisy estimates.

*Severity: Methodological. Fixed in Ver 3 with PyMC/NUTS hierarchical Bayesian models.*

### 4. No convergence diagnostics

No way to distinguish well-identified from poorly identified estimates. Could not detect the floor-piling as a convergence failure.

*Severity: Methodological. Fixed in Ver 3 with R-hat and divergence tracking.*

### 5. Express-saccade detection using BIC

BIC-based bimodality detection over-identified mixtures (detected bimodality where none existed). Fixed in Ver 2.5.

*Severity: Methodological. Fixed in Ver 2.5 by replacing BIC with a fit-driven + structural validation approach.*

### 6. Hard-coded absolute Windows paths

Scripts contained hard-coded paths that broke on other machines.

*Severity: Portability. Fixed in Ver 2.5 by replacing with `SCRIPT_DIR`-based relative paths.*

### 7. No structured pipeline

Scripts were run ad-hoc with no defined run order or output-consumption chain. Could produce inconsistent figures if re-run out of sequence.

*Severity: Operational. Fixed in Ver 3 with RUN_GUIDE.md and an ordered fits→figures→diagnostics workflow.*

---

## What Was Right (and Not Changed)

- The shifted-Wald model choice (single-boundary) — **correct from this version onward**
- The contamination mixture (95% Wald + 5% uniform) — **correct and kept**
- The SRT filter (80–600 ms) — **correct and kept**
- The figure house style (Arial, 300 DPI, pdf.fonttype=42) — **evolved but not discarded**
- KS goodness-of-fit testing — **core diagnostic kept throughout**

---

## Files In This Folder

- `DDM Model/` — Core fitting (DDM_fit.py), figures, conceptual schematics
- `NDT Code/` — Early NDT calculations
- `Vincentile Code/` — Early vincentile methods
- `Verification Code/` — Full 9-page diagnostic suite
- `Bayesian Model/` — Early per-cell Bayesian implementations (seed of Method B)
- `Deprecated/` — Even older code preserved for reference
