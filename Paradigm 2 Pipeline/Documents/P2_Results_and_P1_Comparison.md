# Paradigm 2 (CIR) — Results, Takeaways, and Comparison to Paradigm 1

**Rishvanth Amsaraj · 2026-09-24 · companion to `P2_Technical_Breakdown.md`**

This document synthesizes the Paradigm 2 analysis for the lab: what was found, what it means, what still needs answering, and how it lines up against Paradigm 1. Every number is drawn from the computed tables in `Code/` (see the technical breakdown for the raw tables and the validation logs).

---

## 1. What Paradigm 2 is

| | Paradigm 1 (CMT) | Paradigm 2 (CIR) |
|---|---|---|
| Participants | 16 (CMT001–CMT017, no CMT004/CMT013) | 16 (CIR001–CIR017, no CIR004) |
| Speeds | 0 / 75 / 150 deg/s | 75 / 100 / 125 / 150 deg/s |
| Design | 3 speeds interleaved per block | 4 speeds interleaved per block |
| Block type | `I` | `P2` |
| Trials kept | ~7,676 rows | 10,158 hand + 10,017 saccade (after P1 RT windows) |
| Model | Single-boundary shifted Wald | **Same, unchanged** |

The pipeline is the single-boundary shifted Wald that Paradigm 1 settled on (Method A = MLE with 5% contamination; Method B = hierarchical Bayesian via NUTS). Only configuration changed: speeds, block type, input file, and four-speed figure layout. Likelihood, priors, bounds, floors (hand 130 / saccade 70 ms), RT windows (150–800 / 80–600 ms), optimiser and sampler settings are byte-identical.

---

## 2. Headline results

1. **The pipeline ports cleanly.** Paradigm 2 ran with no methodological changes.
2. **The code and environment are verified.** Re-running the original Paradigm 1 scripts on Paradigm 1 data here reproduces Method A *exactly* (48/48 hand and 48/48 saccade cells, identical model choices) and Method B to Monte Carlo error (hand t₀ r = 0.9995, largest cell difference 1 ms).
3. **Hand t₀ is identified, as in Paradigm 1.** 0 divergences, max R-hat 1.003, 0/64 Bayesian cells at the floor; the floor sweep shows the hand is set by the data, not the bound.
4. **The Paradigm 1 speed effect does NOT replicate.** Hand t₀ is flat across speeds (≈147–152 ms), and the small change that exists is in the *opposite* direction to Paradigm 1. This is the central finding.
5. **Saccadic t₀ is still not identifiable — but for a new reason.** In Paradigm 1 every participant's estimate hit the 70 ms floor. In Paradigm 2, 10/15 hit the floor and 6/15 hit the ceiling (the participant's fastest saccade − 1 ms). Reporting saccadic t₀ as fixed at 70 ms remains the defensible choice.
6. **Left vs Right changes where the hand goes, not when it starts.** Right targets draw a ~13° larger lead, but there is no reliable difference in reaction time or any decision parameter.
7. **The extraction's QA flags don't matter.** Excluding the 48 flagged saccades moves saccadic t₀ by ≤ 1.6 ms and flips the model in 1/25 borderline cells.

---

## 3. Results in detail

### 3.1 Hand non-decision time

| Speed | Method B t₀ (mean [95% CI]) | Method A t₀ (mean ± SD) |
|---|---|---|
| 75 | 146.7 [138.8, 154.8] | 148.1 ± 21.6 |
| 100 | 147.0 [138.7, 155.4] | 146.4 ± 14.4 |
| 125 | 151.7 [142.1, 160.7] | 156.5 ± 17.7 |
| 150 | 150.6 [141.0, 159.9] | 146.5 ± 15.9 |

Hand t₀ is essentially flat: **+3.9 ms from 75 to 150 deg/s [0.8, 6.9]**, and the per-participant trend is not reliable (slope +1.63 ms per 25 deg/s, p = 0.074). Method A — the noisier estimator — shows nothing at all (Friedman p = 0.398).

Method A floors 19/64 cells; the recovery study explains this as estimation noise (true values sit only ~17–22 ms above the 130 ms floor, and per-cell RMSE is 12–14 ms). Partial pooling (Method B) removes it.

### 3.2 Saccadic non-decision time

Saccadic t₀ remains **fixed at 70 ms**. The participant-level model produces posterior means of 71–110 ms, but only CIR014 (94 ms [76, 107]) is clear of *both* the 70 ms floor and the fastest-saccade ceiling; the rest press against one bound or the other.

The shape mechanism that diagnosed Paradigm 1 still holds: pooled saccade skew/CV 3.9 implies t₀ ≈ 44 ms (below the floor); hand skew/CV 12.9 implies ≈ 181 ms (above the floor).

**Express saccades.** Six two-component cells have a genuinely express fast component (< 130 ms): CIR008 at every speed (~105 ms) and CIR007 at two (~125 ms). CIR015's "fast" component (~181 ms) and CIR003's closely-spaced components are two-component fits, not express saccades. CIR001 is the least well-described participant (single-Wald KS 0.110–0.158 at every speed).

### 3.3 Left vs Right and aiming

| Measure (Right − Left, pooled) | Value | p |
|---|---|---|
| Median HRT | +4.79 ms | 0.453 |
| Median SRT | +3.40 ms | 0.860 |
| Method A hand t₀ | +0.16 ms | 0.597 |
| Signed error (SignedError_deg) | **+13.19°** | **0.004** |
| Signed error (HRT50) | **+10.00°** | **< 0.001** |

**Movement start does not differ by direction** — HRT, SRT, t₀, and boundary separation are all indistinguishable across Left and Right. **Aim does differ** — the hand leads the target on 91–95% of Left trials but 98–99% of Right trials, with a ~13° larger lead for Right targets.

This matters for the motor-plan-vs-biomechanics question: whatever produces the Left/Right launch-angle asymmetry acts on the *content or execution* of the movement, not on *when* it starts. The RT data cannot separate plan content from biomechanics — the planned submovement/kinematic analysis is the right tool.

---

## 4. Validation

**Paradigm 1 parity** (original P1 scripts rerun in this environment on P1 data):

| Table | Match |
|---|---|
| `DDM_hrt_fits.csv` | 48/48 cells, exact (0.0 difference on every column) |
| `DDM_srt_fits.csv` | 48/48 cells, exact; model choice 48/48 identical |
| `Bayesian_hrt_fits.csv` | 48/48 cells, t₀ within 1 ms, r = 0.9995 |

**Parameter recovery** (simulate-and-refit at n = 158): hand t₀ recovered with bias −5.4 to +2.2 ms and RMSE 12–14 ms. Saccadic t₀ at a true 90 ms is recovered with ~20 ms RMSE and 30–42% of cells on the floor; a true 44 ms (below the floor) returns ~70–72 ms in 80–90% of cells. Per-cell saccadic t₀ is therefore weakly identified at best.

**Convergence:** hand model 0 divergences, R-hat 1.003; saccade models R-hat ≤ 1.007 with 5 divergences in the 125 deg/s unimodal model (0.08% of draws); two-component and participant-level models 0 divergences, R-hat ≤ 1.005.

---

## 5. Takeaways

1. **The pipeline is portable and reproducible.** The single-boundary shifted-Wald pipeline transferred to a new paradigm with zero methodological changes and passed a byte-level parity check. This is a strong foundation for the rest of the project.

2. **The Paradigm 1 dissociation did not replicate.** Hand t₀ *decreasing* with target speed — the headline of Paradigm 1 — is absent in Paradigm 2. Hand t₀ is flat within ~5 ms across 75–150 deg/s. This is the finding that most needs attention, because it reframes Paradigm 1's central claim.

3. **Hand t₀ identification is robust across cohorts.** Whatever the speed effect, hand non-decision time is cleanly estimated in both paradigms (0 floored cells under Bayesian pooling).

4. **Saccadic t₀ is fundamentally hard to identify — now for two reasons, not one.** Paradigm 1 showed the floor; Paradigm 2 adds the ceiling. Fixing saccadic t₀ at 70 ms survives as the right call, but the ceiling binding is a genuine modeling gap worth addressing (see below).

5. **The Left/Right asymmetry is an aiming effect, not a timing effect.** It acts on movement direction (signed error), not initiation (RT/t₀). This points the next analysis at motor planning/execution rather than the decision stage.

6. **QA flags are benign.** The extraction's quality flags have no material effect on the results, which de-risks the dataset.

---

## 6. Questions that need answering

1. **Why does the Paradigm 1 speed effect not replicate?** The leading candidates are (a) different cohorts (CMT vs CIR are different people), and (b) the task range — Paradigm 1's largest step was *stationary → moving* (0 → 75 deg/s), and a stationary condition may change how participants prepare on every trial. Paradigm 2 has no stationary condition. This needs a decision on how to frame the Paradigm 1 result (cohort-specific vs. task-specific vs. not robust).

2. **Measurement definitions in the v0_1_36 extraction.** Before any aiming result is interpreted, we need the exact timing point of `HandDir_deg` / `TargetDir_deg` and the `_HRT50` variants (now 99.5% populated vs 25% in Paradigm 1). The within-cell HRT-vs-signed-error correlation is mechanically confounded by this, so it must not be interpreted until the definition is confirmed.

3. **The participant-level saccadic t₀ ceiling.** Its upper bound is the participant's single fastest saccade − 1 ms, so the estimate is tied to one trial. Adding the 5% contamination term would remove that dependence, but it changes the model — a decision to make with the professors before anyone edits it.

4. **Five divergences at 125 deg/s.** 0.08% of draws in one model; a higher-`target_accept` refit is the standard check if a reviewer asks.

5. **The Streamlit app hard-codes speeds 0/75/150** (`kinarm_rt/_speeds.py`), so it will not run Paradigm 2 without a small change.

6. **Inherited, unchanged for parity:** the KS < 0.10 mixture trigger is uncalibrated (the true 5% critical value at n ≈ 158 is lower), and the hand / per-speed saccade Bayesian likelihoods have no contamination term.

---

## 7. Paradigm 1 vs Paradigm 2

### At the shared speeds (75, 150 deg/s)

| | 75 deg/s | 150 deg/s |
|---|---|---|
| Hand t₀ (Method B) — Paradigm 1 | 158.0 ms | 147.9 ms |
| Hand t₀ (Method B) — Paradigm 2 | 146.7 ms | 150.6 ms |

Different participants, so this is a between-cohort comparison, not a within-subject replication.

### What replicates

- Hand t₀ identified and stable under Bayesian pooling.
- Hand RT skew/CV = 12.9 (identical to Paradigm 1).
- Saccadic t₀ not identifiable at the participant level.
- Express-saccade-dominant individuals require two-component mixtures.

### What does not replicate

- **The decrease of hand t₀ with speed** (Paradigm 1: 158 → 148 ms; Paradigm 2: flat at ~147–152 ms).
- **Saccadic flooring is weaker**, and a new ceiling-bind appears at the participant level (6/15 vs 0/14 in Paradigm 1).
- **Trial composition:** Paradigm 1's lead-vs-lag comparison (~35% lag) cannot be repeated — lag trials are only 1–9% of Paradigm 2, so Paradigm 2 effectively measures a lead-dominant regime.

### Bottom line

Paradigm 2 **replicates the methodology** and the *identifiability* structure (hand clean, saccade floor-bound), but **not the substantive Paradigm 1 effect**. The speed-dependence of hand non-decision time looks, on this evidence, specific to Paradigm 1's cohort or its stationary-to-moving transition — a result that should be stated plainly and investigated rather than assumed to generalize.
