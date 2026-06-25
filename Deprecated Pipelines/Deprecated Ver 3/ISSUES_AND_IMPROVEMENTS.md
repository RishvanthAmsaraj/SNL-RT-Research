# SNL RT Research — Deprecated Ver 3 Issues & Improvements

**Why this version was superseded, and what replaced it.**

---

## Summary

**Deprecated Ver 3** was the first version to produce the headline dissociation (hand t₀ decreases with speed, saccadic t₀ does not). It had the full hierarchical Bayesian architecture, convergence diagnostics, ordered pipeline, and the 3-category figure suite. The results were correct and the analysis was defensible.

**Nevertheless**, the bounds were not yet anchored to a systematic literature review, and several diagnostics (why saccadic t₀ floors, LATER alternative) had not been developed. These were refinements that strengthened an already-sound foundation without changing the headline result.

**Superseded by:** [Current Pipeline](../../Current%20Pipeline/)

---

## Problems

### 1. Bounds not anchored to systematic review

- V_MAX = 40 was set from the informal Ratcliff & Tuerlinckx (2002) range. The Tran et al. (2020) systematic review showed `|v| ≲ 18.5` at s = 1. Our 40 never bound — fitted v ≈ 4.7–13.8 — but the source was less defensible than needed for publication.
- Hand t₀ floor of 100 ms was informal. Haith et al. (2016) showed reach-preparation minimum ≈ 130 ms.
- v/a bounds attributed to Ratcliff & Tuerlinckx were actually more consistent with Tran 2020 envelopes.

*Severity: Methodological (defensibility, not correctness). None of these changes altered a single fitted value, but they strengthened the publication basis. Fixed in Current Pipeline.*

### 2. Saccadic t₀ was still estimated (unreliably)

The per-participant SRT NDT model used a 35 ms floor that produced overscattered estimates. While the main SRT model fixed t₀ at 70 ms per cell, the narrative still implied saccadic t₀ was estimable at the participant level.

*Severity: Methodological. Fixed in Current Pipeline by raising floor to 70 ms, demonstrating the collapse, and reporting as fixed at 70 ms.*

### 3. No mechanism diagnosis for saccadic flooring

The pipeline knew *that* saccadic t₀ floors but not *why*. This left the explanation as "identifiability" — a label, not a mechanism.

*Severity: Scientific communication. Fixed in Current Pipeline with `why_saccadic_t0_floors.py` — the skew/spread figure showing the Wald's implied t₀ formula.*

### 4. No LATER alternative

The saccade field's native LATER model (reciprobit plots) was not available. A reviewer familiar with the saccade literature would reasonably ask for it.

*Severity: External validation. Fixed in Current Pipeline with `LATER_analysis.py`.*

### 5. DOI error in code

Knox & Wolohan (2015) DOI was `e0133595` — which points to an unrelated HIV-vaccine paper. The correct express-saccade reference is `e0120437`.

*Severity: Citation integrity. Fixed in Current Pipeline.*

### 6. NDT figure axes could mislead

NDT bar charts had unzoomed y-axes that included the full 0–300 ms range, compressing the interesting region (100–180 ms). Bars for the by-speed NDT panels could exaggerate differences at small scales.

*Severity: Visual communication. Fixed in Current Pipeline with zoomed axes and bar→point switch.*

### 7. Implementation not validated against field tools

No explicit comparison to HDDM/HSSM architecture or PyDDM for historical reference.

*Severity: External validation. Fixed in Current Pipeline with architecture comparison notes.*

---

## What Was Right (and Not Changed)

- **Hierarchical Bayesian architecture** — correct, kept
- **Participant-level t₀ pooling** — the fix for floor-piling, kept
- **3-category figure split** — Bayesian (results), DDM (diagnostic), vincentile (raw data) — kept
- **Contamination mixture (95% Wald + 5% uniform)** — kept
- **SRT filter 80–600 ms** — kept
- **HRT filter 150–800 ms** — kept
- **Express-dominant participants kept and modelled with mixtures** — kept
- **Same Bayesian model for all participants** — kept
- **RUN_GUIDE.md** — kept and updated for new scripts
- **Headline dissociation** — unchanged (values shifted slightly from floor changes but survived all checks)

---

## Files In This Folder

- `DDM/` — DDM_fit.py, DDM_figures.py, DDM_conceptual.py
- `Bayesian/` — Bayesian_HRT_fit.py, SRT_fit.py, SRT_ndt.py, Bayesian_figures.py, conceptual.py, SRT_identifiability_check.py, SRT_fixed_t0_analysis.py
- `NDT/` — NDT_barchart.py, NDT_barchart_bayesian.py
- `Vincentile/` — vincentile_figures.py
- `RUN_GUIDE.md`
