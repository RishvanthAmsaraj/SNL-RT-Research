# SNL RT Research — Development History

**A complete record of how this project evolved**, from the earliest PyDDM prototype through to the current literature-anchored, hierarchical-Bayesian pipeline. Every major decision, the problem that motivated it, the numbers that were derived, and — equally important — what we deliberately chose *not* to change.

---

## Scope

This project fits a **single-boundary shifted-Wald drift-diffusion model** to hand reaction time (HRT) and saccadic reaction time (SRT) from a **KINARM interception task** — 16 participants, 3 target speeds (0 / 75 / 150 deg/s), using both frequentist MLE and hierarchical Bayesian (PyMC/NUTS) estimation. The headline result is a **hand-vs-eye dissociation**: hand non-decision time (t₀) decreases with target speed (170 → 158 → 148 ms, p = 0.003), while saccadic t₀ does not (fixed at the 70 ms physiological floor — not identifiable above it).

---

## The Arc at a Glance

| Phase | What Happened | Repo Folder | End State |
|---|---|---|---|
| **0 — Foundations** | PyDDM toy → real data; two-choice → shift-Wald; SRT filter fix; contamination mixture | Ver 1 → Ver 2 | Working DDM on real data; express saccades identified; first Bayesian seed |
| **1 — Bayesian Pipeline** | Hierarchical Bayesian; flooring investigation; dissociation result; ordered pipeline | Ver 2.5 → Ver 3 | Headline dissociation; 3-category figure suite; RUN_GUIDE |
| **1.5 — Citation Audit** | Every citation verified against published record; DOI fix; monkey vs. human species tagging | Ver 3 | Methodological References document; clean citation chain |
| **2 — Literature Anchoring** | Bounds from systematic review; *why* saccadic t₀ floors; LATER alternative; NDT chart refinement | Ver 3 → Current | Literature-anchored bounds; mechanism diagnosis; LATER figures; fixed-70 ms reporting |

Two structural shifts define the entire journey:

- **From a toy to the right model for the task.** The earliest work used a *two-choice* DDM (standard textbook form, requires binary correct/incorrect choice). The real interception task is *go-type* — no binary choice — so the model was changed to a **single-boundary shifted Wald**, the correct descriptive model for one-sided reaction times.
- **From ad hoc scripts to an ordered pipeline.** Early work ran scripts one-off. The current state has a defined run order (fits → figures → diagnostics) where each stage produces CSV outputs consumed by the next.

---

## Phase 0 — Foundations

*Repo folders: Deprecated Ver 1 → Deprecated Ver 2*

### 0.1 The Rudimentary Start (Synthetic Data, Two-Choice DDM)

The project began as a proof-of-concept: a **two-choice diffusion model fit to synthetic data** using the **PyDDM library** (Shinn et al. 2020). This confirmed that the machinery worked and parameters could be recovered. No real data, no real pipeline, no statistical validation.

**Problem that drove deprecation:** PyDDM was too inflexible for custom likelihoods (e.g., the contamination mixture we later needed), had version conflicts, and was slower than native implementations on large datasets.

### 0.2 The Key Early Modelling Decision: DDM → Shifted Wald

Moving toward the real task forced the first critical decision. The KINARM interception task does not have a binary choice outcome — participants initiate a movement and we record *when*. A two-choice DDM is the wrong object. The correct model is the **single-boundary diffusion**, whose first-passage-time density is the **shifted Wald** (inverse Gaussian with a temporal shift t₀).

This equivalence — single-boundary diffusion = shifted Wald — became the settled foundation of everything afterward. The three parameters carried forward:

| Parameter | Meaning |
|---|---|
| **Drift rate `v`** | Rate of evidence accumulation toward the action threshold |
| **Boundary `a`** | Evidence required before the response is initiated |
| **Non-decision time `t₀`** | Sensory + motor time outside the decision process |

### 0.3 Moving to the Real KINARM Dataset

The synthetic data was replaced with the real dataset: **16 participants** (`CMT001–CMT010`, `CMT0011`, `CMT0012`, `CMT0014–CMT0017`; the set skips `CMT0013`), each performing the interception task at three target speeds (0, 75, 150 deg/s), with both hand RT and saccadic RT per trial. Per-participant files (`CMT*_MASTER_Summary.csv`) were concatenated into `pooled_data.csv` (**7,676 trials**).

### 0.4 The First Real Pipeline & Deliverable Suite

Phase 0 produced the first real deliverable suite (Ver 2):
- **~10 scripts** across DDM fitting, visualization, diagnostics
- **7 figure families** — vincentile plots, RT histograms, KDE overlays, DDM conceptual schematics, NDT bar charts, diagnostic suite
- **House visual style** — pale green/red/blue per speed, Arial font, 300 DPI PDFs with `pdf.fonttype=42`
- **Cached outputs** (`.npz`) for reproducibility

### 0.5 First Major Fix — The SRT Data Filter

**Problem.** The saccadic RT filter was initially set too high (≥150 ms lower cutoff), removing genuine fast saccades — including the express-saccade range.

**Fix.** SRT filter corrected to **80–600 ms**. 80 ms is the human anticipation threshold; 600 ms removes lapses. Hand filter set to **150–800 ms**.

> **Why this mattered:** The first lesson that data-cleaning cutoffs are distinct, consequential modelling choices — and that an over-aggressive cutoff can silently throw away the very trials (express saccades) that later prove scientifically important.

### 0.6 Second Major Fix — The Contamination Mixture Model

**Problem.** A pure shifted-Wald MLE is sensitive to outliers (a handful of implausible responses can drag the fit).

**Fix.** Following Ratcliff & Tuerlinckx (2002), the likelihood became a **95% shifted Wald + 5% uniform contamination mixture**. The uniform component absorbs contaminant trials mathematically while keeping **all participant data** — a standing project principle.

### 0.7 Discovering the SRT Floor-Piling Artifact

**Problem.** Saccadic t₀ estimates piled up against their lower bound ("floor-piling") — many cells sat exactly on the floor rather than taking a range of values. This is the central technical thread of the entire project.

**First approach.** Hierarchical Bayesian estimation in **PyMC** was introduced to address it — the seed of what became Method B.

### 0.8 Express Saccades Identified

**`CMT0012`** and **`CMT002`** were identified as express-saccade-dominant (bimodal saccadic distributions, `CMT0012` measured at 63–80% express). **Decision held for the entire project:** keep these participants; handle bimodality with mixture models.

### 0.9 Environment Workaround

PyMC does not install cleanly via pip on Windows. A **conda-based installation workaround** was documented with pre-saved validated results.

---

## Phase 1 — The Bayesian Refinement and the Headline Result

*Repo folders: Deprecated Ver 2.5 → Deprecated Ver 3*

### 1.1 The Flooring Investigation

Examining the improved Bayesian NDT bar chart raised two questions:
- **(a)** The SRT t₀ piling around 70 ms
- **(b)** Three HRT cells apparently floored at 100 ms at 150 deg/s

#### The SRT "70 ms Pile"

A visual red herring was corrected: an old version had floored at 40 ms and the chart *label* still said 40 ms, but the actual floor in the code was **70 ms**. The pile was real — **24 of 48 cells (50%) sat exactly at 70 ms**. Diagnosed as the **identifiability limit**: fast, low-variance saccades carry almost no information to separate non-decision time from decision time.

#### The HRT Three Floored Cells

Exactly three cells — **`CMT001`, `CMT002`, `CMT010` at 150 deg/s** — sat at the 100 ms floor.

#### The Consequential Discovery

Those three floored cells were the *only* reason the DDM hand speed effect was statistically significant:
- DDM: Friedman **p = 0.047**
- Dropping those three participants: **p = 0.199**

The headline effect rested entirely on three floored values — a fragile result we could not have defended.

### 1.2 The Bayesian Re-Fit — Result That Reversed Expectations

The expectation was that lifting the floored cells would *weaken* the speed effect. **The opposite happened.**

- The Bayesian HRT model floored **0 cells** (DDM floored 3), converged cleanly (0 divergences, r-hat ≈ 1.002), and lifted the three problem cells: `CMT001` 100→142 ms, `CMT002` 100→151 ms, `CMT010` 100→138 ms.
- The speed effect got **stronger**: Friedman **p = 0.047 (DDM) → 0.0016 (Bayesian)**.
- 14 of 16 participants showed lower t₀ at 150 deg/s than at 0 (median drop ≈ 22 ms).

**Why the reversal:** The flooring was not manufacturing a false effect — it was one symptom of broad per-cell MLE noise scrambling the within-subject ranks the Friedman test depends on, *hiding* a real effect. Partial pooling regularized noisy estimates, the noise dropped out, and the underlying trend surfaced.

**Two checks ruled out a parameterization artifact:**
1. The fastest HRT itself drops with speed (200→186→166 ms, p = 0.001)
2. The pooled, speed-agnostic fraction of the fast-RT window that t₀ occupies also drops (0.687→0.673→0.616, p = 0.039)

### 1.3 The Dissociation

Putting the effectors together gave the headline result: **hand t₀ decreases with target speed (p = 0.0016, robust, no floored cells), while saccadic t₀ does not change with speed** (non-significant, and not identifiable in the first place).

| Speed | Bayesian HRT t₀ (95% CI) | Saccadic t₀ |
|---|---|---|
| 0 deg/s | 168 ms (154–182) | Fixed at 70 ms |
| 75 deg/s | 156 ms (139–174) | Fixed at 70 ms |
| 150 deg/s | 147 ms (130–162) | Fixed at 70 ms |
| Friedman p | 0.0016 | (not testable — fixed) |

### 1.4 Resolving "Which Model Per Participant" Confusion

Important clarification: **Every participant, every variable, uses the same hierarchical Bayesian model.** The evidence that flooring is a *per-cell identifiability* phenomenon is that even the **Bayesian per-cell SRT fit still floors 19 of 33 cells at 70 ms**. What removes the flooring is pooling t₀ to the participant level, which floors 0 cells.

### 1.5 The Pipeline Architecture

Phase 1 turned scripts into a **pipeline**:
- **4 fitting scripts**: `DDM_fit.py` (Method A), `Bayesian_HRT_fit.py`, `Bayesian_SRT_fit.py`, `Bayesian_SRT_ndt.py` (Method B)
- **7 figure scripts** with DDM/Bayesian pairs
- **3-category figure split**: Bayesian (results), DDM (comparison/diagnostic), vincentile (raw data)
- **RUN_GUIDE.md** — installation + run order
- Guidelines for how to present each panel without overclaiming

---

## Phase 1.5 — Citations and the Methodology Document

*Repo folder: Deprecated Ver 3*

### What Was Verified / Fixed

- **One hard error:** The **Knox & Wolohan (2015) DOI** was wrong in the code — `e0133595` pointed to an unrelated HIV-vaccine paper; corrected to `e0120437`.
- Ratcliff & Tuerlinckx (2002) validated as source for boundary range and contamination mixture.
- Wiecki et al. (2013) (HDDM) confirmed as hierarchical-Bayesian precedent.
- Gelman et al. for non-centered parametrization, partial pooling, and R-hat.

### Two Conceptual Clarifications

1. **Monkey vs. human.** The ~70 ms saccade figure originates in monkey neurophysiology (Fischer & Boch 1983; Dorris, Paré & Munoz 1997). The code already applied human-appropriate sourcing (80 ms anticipation threshold).
2. **Two kinds of lower bound.** Data-cleaning cutoffs (150 ms / 80 ms on raw RTs) ≠ fitted-parameter floors (130 ms / 70 ms on t₀). Conflating them is a common reviewer trap.

A `Methodological_References` document was produced capturing all of this. None of these changes altered a single fitted number — they were about correctness and defensibility.

---

## Phase 2 — Literature-Anchored Bounds, Flooring Diagnosis, and Alternatives

*Repo folder: Current Pipeline*

### 2.1 Re-Deriving the Bounds from a Systematic Review

The key new source was **Tran et al. (2020)**, a systematic review pooling published DDM fits and rescaling to s = 1. Cross-referencing our fitted values against its envelopes drove several changes:

| Parameter | Old Value | New Value | Source | Impact |
|---|---|---|---|---|
| **Drift cap V_MAX** | 40 | 20 | Tran (2020): \|v\| ≲ 18.5 at s = 1 | Never bound at 40; no result change |
| **Hand t₀ floor** | 100 ms | 130 ms | Haith et al. (2016): reach preparation ≈ 130 ms | Barely binds (fitted min 129 ms); result survives |
| **v/a bounds** | Ratcliff & Tuerlinckx (informal) | Tran (2020) systematic envelopes | Tran (2020) | Re-attribution; no numerical change |

These were classed as **safe changes** — they do not alter fitted results.

### 2.2 The SRT NDT Collapse and "Fixed at 70 ms" Decision

One change was *not* safe: raising the per-participant SRT NDT floor from 35 to 70 ms produced a **collapse** — all 14 participants pinned at 70–71 ms, between-subject variance went to zero. The population-mean hyperparameter wanted ≈ 40 ms, *below* the floor.

**What it means:** This is not a bug. It confirms that **saccadic non-decision time is not identifiable above the physiological floor**. The decision: report saccadic t₀ as **fixed at 70 ms**, not estimated per participant — sharpening the dissociation.

### 2.3 Diagnosing *Why* Saccadic t₀ Floors — The Mechanism

A central Phase-2 investigation produced a clean data-grounded mechanism. A shifted Wald ties skewness to spread (for a pure Wald, skewness = 3 × CV), which forces an implied non-decision time:

```
implied t₀ = mean RT − 3 · SD / skewness
```

| Effector | skew / CV | Shape-Implied t₀ | Outcome |
|---|---|---|---|
| Hand | ≈ 12.9 (strongly right-skewed) | ≈ 191 ms (above 130 floor) | Identified |
| Eye | ≈ 3.4 (near-symmetric) | ≈ 20–30 ms (below 70 floor) | Floors |

**Three conclusions:**
1. **Not a sample-size problem.** Saccadic cells carry median 110 trials (range 79–118) — ample.
2. **Expected behaviour.** Saccadic latencies are near-reciprocal-normal — the saccade field built LATER precisely because of this.
3. **Fixed 70 ms is realistic.** Free estimate (~20–30 ms) is below saccadic conduction time (~70–90 ms).

### 2.4 Validating the Code Against the Field

The hierarchical-Bayesian-PyMC architecture **matches HDDM (Wiecki 2013) and HSSM** in structure (NUTS, subject + group parameters, partial pooling). The single difference: HDDM/HSSM fit the *two-choice* DDM; ours is a single-boundary shifted Wald, appropriate for a go-type task.

### 2.5 The LATER Alternative for Saccades

The saccade field's native LATER model was implemented on the data:

- Saccadic latencies fall on the **straight reciprobit line** (median r² = 0.98)
- Raw goodness-of-fit: **LATER and Wald are about tied** (KS ≈ 0.12 each)
- LATER's advantage: **no separate non-decision-time parameter** — the floor question does not arise
- Cost: LATER parameters (rate, threshold) do not map onto Wald parameters

**Recommendation:** Keep the single Wald for both effectors as primary analysis (preserves the clean dissociation); include LATER as complementary saccade analysis.

### 2.6 Figure Refinements

- DDM NDT chart "physiological min" line: 100 → 130 ms
- Bayesian NDT chart floor label: "100 ms" → "130 ms"
- Y-axes zoomed to populated ranges
- t₀-by-speed panels: bars → mean-markers-with-dots (truncated bars exaggerate differences)

---

## Parameter-Value Changelog

A focused, value-by-value record of how each bound/setting evolved. "Stable" means it was set once on a defensible basis.

| Parameter | Ver 1 | Ver 2 | Ver 2.5 | Ver 3 | **Current** | Source |
|---|---|---|---|---|---|---|
| Model class | Two-choice DDM | Single-boundary Wald | ←same | ←same | Single-boundary Wald | Go-type task |
| SRT filter | ≥150 ms | 80–600 ms | ←same | ←same | 80–600 ms | Human anticipation threshold |
| HRT filter | 150–800 ms | ←same | ←same | ←same | 150–800 ms | Anticipation/lapse removal |
| Likelihood | Pure Wald | 95%W + 5%U | ←same | ←same | Contamination mixture | Ratcliff & Tuerlinckx 2002 |
| V_MAX | (none) | 40 | ←same | ←same | **20** | Tran et al. 2020 |
| Hand t₀ floor | (none) | 100 ms | ←same | ←same | **130 ms** | Haith et al. 2016 |
| SRT per-t₀ floor | (none) | 70 ms | ←same | ←same | 70 ms | Bompas 2017; Ludwig 2007 |
| SRT ppt NDT floor | (none) | — | 35 ms | ←same | **70 ms** (→ fixed) | Saccadic dead time |
| Knox DOI | — | — | — | e0133595 | **e0120437** | Corrected via citation audit |
| Estimation | PyDDM | MLE (scipy) | MLE + cell Bayes | Hier. Bayes | Hier. Bayes | Wiecki 2013 |

**HRT t₀ results at each stage:**

| Version | 0 deg/s | 75 deg/s | 150 deg/s | p-value | Floored cells |
|---|---|---|---|---|---|
| Ver 2 (DDM, floor 100) | 163 ms | 152 ms | 141 ms | 0.047 | 3 |
| Ver 3 (Bayesian, floor 100) | 168 ms | 156 ms | 147 ms | 0.0016 | 0 |
| Current (Bayesian, floor 130) | 170 ms | 158 ms | 148 ms | 0.0034 | 0 |

---

## Decisions We Deliberately Did NOT Change

Equally important for the repository record: things we considered and chose to keep.

1. **Keeping express-dominant participants (`CMT0012`, `CMT002`, `CMT003`, `CMT004`).** Excluding them would clean up saccadic distributions, but they are real data. Bimodality is handled with mixture models.

2. **The single-boundary Wald, not a two-choice DDM.** Two-choice would let us cite the larger DDM literature, but it is the wrong object for a go-type task.

3. **The same model for both effectors.** Using LATER for saccades would fit them more naturally, but would break the cross-effector comparison that is the core result.

4. **One uniform Bayesian estimator across all participants.** Never split participants between DDM and Bayesian; the ones you'd hand to the DDM are the ones that floor hardest.

5. **No across-trial variability parameters (sv, sz, st₀).** Hard to estimate at our trial counts; literature cautions against over-claiming (Böhm et al. 2018).

6. **Reporting the non-significant / non-identifiable saccadic result honestly.** A clean "saccadic t₀ is not testable (fixed at the floor)" is more defensible than a strained effect.

---

## Pipeline Today: File Map and Run Order

### Current Pipeline (folder: `Current Pipeline/`)

**Fitting scripts (run first):**
- `DDM/DDM_fit.py` — Method A, frequentist MLE (pip-installable; comparison/diagnostic)
- `Bayesian/Bayesian_HRT_fit.py` — hierarchical Bayesian hand fit (the reported HRT result)
- `Bayesian/Bayesian_SRT_fit.py` — hierarchical Bayesian per-cell saccadic fit with mixture handling
- `Bayesian/Bayesian_SRT_ndt.py` — per-participant saccadic NDT model (collapses to 70 ms)

**Diagnostic scripts:**
- `Bayesian/SRT_identifiability_check.py` — shows saccadic t₀ is unidentifiable
- `Bayesian/SRT_fixed_t0_analysis.py` — drift-by-speed pattern unchanged across fixed t₀ values
- `Bayesian/why_saccadic_t0_floors.py` — the skew/spread mechanism figure
- `Bayesian/LATER_analysis.py` — the saccade-native LATER alternative

**Figure scripts:**
- `DDM/DDM_figures.py`, `DDM_conceptual.py` — DDM comparison/diagnostic figures
- `Bayesian/Bayesian_figures.py`, `Bayesian_conceptual.py` — reported results
- `NDT/NDT_barchart.py`, `NDT_barchart_bayesian.py` — NDT visualizations
- `Vincentile/vincentile_figures.py` — model-free raw-RT figures

**Data & docs:**
- `pooled_data.csv` — canonical input (7,676 trials)
- `RUN_GUIDE.md` — installation and run order

**Run order:** fits → figures → diagnostics

### Deprecated Pipelines (folder: `Deprecated Pipelines/`)

Each deprecated version is preserved in its own subfolder with its own `ISSUES_AND_IMPROVEMENTS.md` documenting why it was superseded:

| Folder | Contents | Superseded by |
|---|---|---|
| `Deprecated Ver 1/` | PyDDM prototypes, synthetic data generators | Ver 2 |
| `Deprecated Ver 2/` | Native scipy MLE, first real data pipeline | Ver 2.5 |
| `Deprecated Ver 2.5/` | Early Bayesian, express saccade handling | Ver 3 |
| `Deprecated Ver 3/` | Full hierarchical Bayesian, dissociation result | Current Pipeline |

---

## Recurring Principles and Lessons

1. **Floor-piling is an identifiability symptom, not a tuning problem.** No bound, prior, or pooling recovers information not in the data.

2. **Per-cell MLE noise can hide effects, not just inflate them.** The hand speed effect got *stronger* under partial pooling because pooling removed rank-scrambling noise.

3. **Distinguish the two kinds of lower bound.** Data-cleaning cutoffs (150 ms / 80 ms on raw RTs) ≠ fitted-parameter floors (130 ms / 70 ms on t₀). Reviewers conflate them.

4. **Anchor bounds to systematic, independent sources — and don't cherry-pick.** The Phase-2 bounds come from a systematic review (Tran 2020) and task-specific physiology (Haith 2016), not from papers selected to flatter the result.

5. **Verify citations against the record.** The Knox & Wolohan DOI error (pointing to an unrelated HIV-vaccine paper) is the cautionary example.

6. **Report the hard cases honestly.** Saccadic non-identifiability and the moving-target interpretation caveat are stated plainly.

7. **Keep the data; handle the complexity.** Express-dominant participants were kept and modelled, never excluded.

---

*This document reflects the final project state: literature-anchored bounds (V_MAX = 20, hand t₀ floor 130 ms, saccadic t₀ fixed at the 70 ms physiological floor), the hand-vs-eye dissociation as the headline result (HRT t₀ 170 → 158 → 148 ms, p = 0.003), and the LATER model available as a complementary saccade analysis.*
