# SNL RT Research — Deprecated Ver 1 Issues & Improvements

**Why this version was superseded, and what replaced it.**

---

## Summary

**Deprecated Ver 1** was the proof-of-concept phase — two-choice DDM on synthetic data using the **PyDDM library** (Shinn et al. 2020). It demonstrated the fitting machinery worked but was not suitable for real KINARM data. All code here is preserved for historical reference; none of it should be used for production analysis.

**Superseded by:** [Deprecated Ver 2](../Deprecated%20Ver%202/)

---

## Problems

### 1. Wrong model class

Two-choice DDM requires binary correct/incorrect outcomes. The KINARM interception task is **go-type** (no choice). The correct model is a **single-boundary diffusion (shifted Wald)**.

*Severity: Foundational. Fixed in Ver 2 by switching to native scipy MLE of the shifted Wald.*

### 2. PyDDM library dependency

PyDDM is a research library with its own update cycle, bugs, and limitations:
- Inflexible for custom likelihoods (e.g., the contamination mixture)
- Version conflicts with newer Python releases
- Slower than native numpy/scipy on large datasets
- Abstracted away the mathematical detail needed for diagnostic work

*Severity: Operational. Fixed in Ver 2 by implementing native scipy MLE.*

### 3. Synthetic data only

No connection to real KINARM data. No empirical validation.

*Severity: Foundational. Fixed in Ver 2 by loading `pooled_data.csv`.*

### 4. No statistical validation

No KS goodness-of-fit, no parameter recovery, no diagnostic plots. Unclear whether parameter estimates were trustworthy.

*Severity: Methodological. Fixed in Ver 2 by adding the full diagnostic suite.*

### 5. No express-saccade handling

Saccadic RT bimodality existed in the data but was not detected or modelled.

*Severity: Methodological. First addressed in Ver 2 (SRT filter fix), fully addressed in Ver 2.5 (mixture models).*

### 6. Basic visualization

Simple matplotlib histograms without publication-quality formatting. No vector output.

*Severity: Cosmetic. Fixed in Ver 2 with 300 DPI PDFs, proper fonts, and the house style.*

### 7. SRT filter too aggressive

The ≥150 ms lower cutoff removed genuine fast saccades (including express saccades). Fixed to 80 ms in Ver 2.

*Severity: Data integrity. Fixed in Ver 2.*

---

## Files In This Folder

- `DualChoice*.py` — Two-choice DDM implementations (PyDDM and native)
- `SingleChoice*.py` — Single-boundary implementations (both libraries)
- `*DataGen.py` — Synthetic data generators for parameter recovery tests
- `MIGRATION_NOTES.md` — Notes written at the time of the Ver 1 → Ver 2 transition
