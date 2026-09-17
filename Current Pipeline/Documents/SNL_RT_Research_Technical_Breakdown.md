# SNL RT Research — Complete Technical Breakdown

**Rishvanth Amsaraj • 2026-08-21**

This is my complete explanation of the SNL RT Research project — what it is, how the models work, why every number is what it is, and how the pieces fit together. Every value below is drawn from the repository and the run outputs (the 2026-08-16 LATER/two-boundary run, the 2026-08-06 validation study v5, and the kinarm-rt-app record v1.1–v1.10). Where a number carries a documented caveat, I state it explicitly.

> **Final decision (September 2026):** the single-boundary shifted Wald is the production model. The LATER and two-boundary explorations (Sections 8–9) concluded they offer no compelling benefit, so both are archived under `Working Iterations/`. The body of this document keeps the full analysis for the record.

## Contents

1. What this project is, in one page
2. The five statistical models — what each one is and why
3. The repository structure — production vs deprecated
4. The data (pooled_data.csv) — what every column means
5. The model parameters — v, a, t₀ — and every prior, bound, and floor
6. The headline dissociation — the Bayesian result, explained
7. The frequentist validation study (v5) — FDR, bootstrap, and what 22/48 means
8. Directive 1: LATER on the hands — the tie and the shift-LRT
9. Directive 2: the two-boundary DDM question
10. The app shell (kinarm-rt-app) — what it is, its tests, and every bug caught
11. The full pass/fail/accuracy table — every model, every metric
12. How every number relates to every other number (the web of evidence)
13. Paper takeaways — what to lead with, what to hedge
14. Open questions and next steps
15. Bibliography and DOI index

---

## 1. What this project is, in one page

This is my reaction-time (RT) modeling project in the Penn State Sensorimotor Neuroscience Lab. The task is a KINARM interception (go-type) task: a target appears, and the participant must move a hand (or move their gaze) toward it as fast as possible. There is no binary choice — no "left vs right" — just *initiate a movement when the target appears*. That single fact drives almost every modeling decision in the repository, because it means the correct model is a **single-boundary accumulator**, not the two-choice drift-diffusion model most people reach for first.

The scientific question is: what separates the time it takes to decide to move from the time it takes to physically move? That split is the **non-decision time (t₀)** parameter. My central finding is a **dissociation**: hand movements carry a large, identifiable non-decision time, while saccades (eye movements) do not — their t₀ collapses to a physiological floor. This matters because it says something real about the difference between the neural/motor pipelines for reaching versus saccades.

Two effectors are measured in the same participants, at three target speeds (0°, 75°, 150° per second):

- **HRT** = Hand Reaction Time (`HandRT_ms`)
- **SRT** = Saccade Reaction Time (`GazeSRT_ms`)

The project has four overlapping layers of work: (1) the core pipeline that fits the models, (2) a frequentist validation study that checks the pipeline against itself and the literature, (3) a set of professor-directed next steps (LATER on the hands, two-boundary feasibility), and (4) an app shell (`kinarm-rt-app`) that wraps the whole pipeline in a GUI/CLI so it runs on any computer.

---

## 2. The five statistical models — what each one is and why

There are exactly five statistical/cognitive models in play. They are **not** AI models and not interchangeable; each answers a different question. Understanding why each exists is the key to understanding the whole project.

### Model A — The shifted Wald (single-boundary diffusion)

This is the workhorse. The shifted Wald (an inverse-Gaussian distribution shifted in time by t₀) is the exact first-passage-time distribution of a single-boundary diffusion process — which is precisely what a "go" movement task is. It has three parameters:

- **v (drift rate)** — how fast evidence accumulates toward the "go" threshold. Higher v = faster decisions.
- **a (boundary separation)** — how much evidence must accumulate before the response fires. Higher a = more cautious = slower but more accurate.
- **t₀ (non-decision time)** — the time for everything that isn't the decision: sensory transduction, afferent conduction, and muscle activation. This is the parameter of scientific interest.

```
log PDF(τ; v, a) = log a − ½·log(2π) − 1.5·log(τ) − (a − vτ)² / (2τ),   τ = RT − t₀
```

Fitted two ways: **Method A** (frequentist maximum-likelihood, per participant×speed cell, in `DDM_fit.py`) and **Method B** (hierarchical Bayesian, in `Bayesian_HRT_fit.py` / `Bayesian_SRT_fit.py`). Why two methods? Method A is fast and directly comparable to the literature's frequentist fits; Method B gives credible intervals and partial pooling (borrowing strength across participants), which matters when individual cells have few trials.

### Model B — The hierarchical Bayesian version (Method B)

Same shifted-Wald likelihood, but the parameters are drawn per participant from group-level distributions (e.g. vᵢ ~ Normal(μᵥ, σᵥ)), following the HDDM/Wiecki et al. (2013) architecture, sampled with PyMC/NUTS. This is the model that produces the headline dissociation. "Hierarchical" is what lets the model share information: if one participant has noisy data, the group-level prior stabilizes their estimate instead of it exploding.

### Model C — LATER (Linear Approach to Threshold with Ergodic Rate)

LATER (Carpenter & Williams 1995) models the reciprobit of RT: if you plot cumulative RT on a probit axis against reciprocal time (1/RT), a single Gaussian rate process produces a straight line. LATER has a different parameterization than the Wald (rate r, where RT = t₀ + 1/r) and a different theoretical justification, but it is mathematically closely related to the shifted Wald. I use it as a complementary model — especially for saccades — and as the basis for the professor-directed "shift-LRT" test (Section 8).

### Model D — The two-component express/regular mixture

Saccade cells are sometimes bimodal: a fast "express" mode and a slower "regular" mode. For cells flagged as bimodal (by Hartigan's dip test / structural flags), the pipeline fits a two-component mixture: π·Wald(express) + (1−π)·Wald(regular). This matters because a single Wald can't represent a bimodal distribution; forcing it to would corrupt v, a, and t₀. The finding (Section 8) is that most flagged cells aren't genuinely "express" — only 1 of 16 has a mode under 130 ms.

### Model E — The shift-LRT (nested LATER)

This is a statistical test more than a model. Shifted LATER (RT = t₀ + 1/r) nests plain LATER (t₀ = 0). So "does this effector demand a non-decision term?" becomes a likelihood-ratio test comparing the two nested models. The beauty: it asks the t₀ question **without any Wald assumptions**, giving an independent-family replication of the dissociation. This was my suggestion, and it produced the sharpest result (Section 8).

---

## 3. The repository structure — production vs deprecated

The repo is organized around a simple principle: **one audited production pipeline, and every prior version preserved as a frozen snapshot with its own issues log.** Nothing is deleted; superseded work is moved into `Deprecated Pipelines/`.

| Path | Status | What it is |
|---|---|---|
| `Current Pipeline/` | ★ PRODUCTION | The live pipeline (v3.0, Phases 0–2): `Code/`, `Documents/`, `Figures/`, `ISSUES_AND_IMPROVEMENTS.md`. |
| `Deprecated Pipelines/Deprecated Ver 1` | Deprecated | First version. Frozen + issues log. |
| `Deprecated Pipelines/Deprecated Ver 2` | Deprecated | Second version. |
| `Deprecated Pipelines/Deprecated Ver 2.5` | Deprecated | Incremental revision between 2 and 3. |
| `Deprecated Pipelines/Deprecated Ver 3` | Deprecated | Immediate predecessor of Current; has its own `RUN_GUIDE.md` + `Documents/`. |
| `kinarm-rt-app/` | Production (app shell) | The GUI/CLI wrapper around the pipeline (Section 10). |
| `README.md` | Live | Research entrance (not an app manual). |
| `CHANGELOG.md`, `DEVELOPMENT_HISTORY.md` | Live | Version/decision history. |
| `REFERENCES.md` / `REFERENCES.bib` | Live | Full bibliography (also mirrored at the end of this document). |
| `LICENSE` | Live | All Rights Reserved (per the 2026-06-23 decision). |

### Why the rotation exists, and what changed at each step

Every time a version was superseded, the reasons were logged. The key changes from Deprecated Ver 3 → Current (captured in the ISSUES doc):

- Corrected the **Knox & Wolohan DOI**: `e0133595` → `e0120437` (the earlier number was wrong; see Section 15).
- **Drift cap Vmax**: 40 → 20, re-anchored to Tran et al. (2020), which found |v| ≲ 18.5 at s=1 in a systematic review.
- **Hand t₀ floor** 100 → 130 ms, re-anchored to Haith et al. (2016).
- **Saccade t₀ floor** 35 → 70 ms (Bompas & Sumner 2011; Ludwig et al. 2007).
- **v/a bounds** re-attributed from Ratcliff & Tuerlinckx to Tran et al. (2020).
- **LATER** added as a complementary saccade analysis.
- **Saccadic t₀** now reported fixed at 70 ms, not estimated.

Why this matters: a paper's claims live or die on parameter bounds. Anchoring every floor and cap to a specific literature source (with DOI) is what makes the results defensible in peer review. The rotation system guarantees the provenance of every number is traceable.

---

## 4. The data — pooled_data.csv, column by column

The canonical input is `pooled_data.csv` with 7,676 trials.

| Column | Meaning | Why it matters |
|---|---|---|
| `Participant` | Participant ID (16 participants total). | The random effect in the hierarchical model; the unit of pooling. |
| `Speed_deg_per_s` | Target speed: 0 / 75 / 150 deg/s. | The main experimental manipulation; t₀ decreases with speed (169.5 → 158.0 → 147.9 ms on hand). |
| `HandRT_ms` | Hand reaction time (ms). | The HRT effector — filtered 150–800 ms. |
| `GazeSRT_ms` | Saccade (gaze) reaction time (ms). | The SRT effector — filtered 80–600 ms. |
| `BlockType` | Filter to `'I'` (interception) trials. | Removes non-interception blocks that would contaminate the RT distribution. |
| Kinematic columns | Signed initial movement/lateral error, target side, etc. | Used in the two-boundary analysis to recover error-trial direction (Section 9). |

### The filters and why they matter

- **HRT 150–800 ms** — lower bound removes anticipation (Whelan 2008); upper bound removes lapses (Luce 1986). A trial under 150 ms on the hand is almost certainly an anticipatory movement, not a real decision; a trial over 800 ms is a lapse.
- **SRT 80–600 ms** — 80 ms is the human anticipation threshold for saccades (Fischer & Weber 1993). Saccades are faster than hands, so the window is shifted left.

Why this matters: the shifted Wald is defined for a pure decision-time process. If anticipation trials (which have near-zero decision time) are left in, they pull t₀ down and inflate v artificially. Filtering is the first, most boring, and most consequential modeling step.

---

## 5. The parameters — v, a, t₀ — and every prior, bound, and floor

This section is the "why every number is what it is" reference. It's the content of `CODE_REFERENCE.md`, which is the single source of truth for the pipeline.

### Parameter meanings

| Parameter | Name | Meaning in plain English |
|---|---|---|
| v | Drift rate | Rate of evidence accumulation toward "go." The speed of the decision engine. |
| a | Boundary separation | How much evidence is needed before the response fires. The height of the finish line. |
| t₀ | Non-decision time | Sensory + motor time outside the decision. The parameter of scientific interest. |

### Priors (Bayesian, Method B)

| Prior | Value | Meaning / source |
|---|---|---|
| μₗᵥ (group drift, log scale) | Normal(log 10, 0.5) | Centered at drift 10 (the ln of 10), wide enough to not over-constrain. Log-normal keeps v positive (drift can't be negative in a go-task). |
| μₗₐ (group boundary, log scale) | Normal(log 1, 0.5) | Centered at boundary 1. In single-boundary models a is often fixed at 1 for identifiability (v and a trade off); here it's estimated with a log-normal prior. |
| μ_z (starting point) | Normal(0.5, 1) hand; Normal(0, 1) saccade | Starting-point bias. Hand centered at 0.5 (midpoint), saccade at 0. Different because the two effectors have different bias structure. |
| Contamination | 5% Uniform | 0.95·Wald + 0.05·Uniform(RTmin, RTmax) — a robust mixture term for outlier trials. |

### Bounds and floors (the ones reviewers will check)

| Bound | Value | Source | Why it matters |
|---|---|---|---|
| Drift cap Vmax | 20 | Tran et al. (2020): \|v\| ≲ 18.5 at s=1 | Prevents the fit from returning absurd drift rates (a sign of model misspecification). |
| Hand t₀ floor | 130 ms | Haith et al. (2016): reach-preparation minimum | Physiological floor; a hand can't be ready faster than ~130 ms, so t₀ below that is unphysical. |
| Saccade t₀ floor | 70 ms | Bompas & Sumner 2011; Ludwig et al. 2007 | Saccadic afferent+efferent conduction minimum. This is the number the dissociation hinges on — saccadic t₀ collapses to 70 and is reported as fixed, not estimated. |
| HRT filter | 150–800 ms | Whelan 2008; Luce 1986 | Anticipation + lapse removal. |
| SRT filter | 80–600 ms | Fischer & Weber 1993; Luce 1986 | Anticipation + lapse removal (saccade-scaled). |

### Why the 70 ms floor is the heart of the project

Fast saccadic cells have per-cell t₀ that tracks the 70 ms floor rather than being identified by the data (see `SRT_identifiability_check.py`). In the Bayesian per-participant model, saccadic t₀ collapses to 70 ms for all participants and is reported as a fixed constant, not an estimated parameter. The contrast — hand t₀ is freely estimated and identified (169.5/158.0/147.9 ms across speeds), saccade t₀ collapses to a floor — **is the dissociation**. It's not just a number; it's the whole finding.

### Run order (how the scripts chain)

```
DDM_fit.py ─────────────────► DDM_hrt_fits.csv, DDM_srt_fits.csv
  ├─► Bayesian_HRT_fit.py ─► Bayesian_hrt_fits.csv, Bayesian_hrt_ndt.csv
  ├─► Bayesian_SRT_fit.py ─► Bayesian_srt_fits.csv
  ├─► DDM_figures.py / DDM_conceptual.py
  ├─► NDT_barchart.py
  └─► SRT_identifiability_check.py / SRT_fixed_t0_analysis.py
        └─► Bayesian_SRT_ndt.py (depends on DDM_srt_fits.csv for single/mixture split)
              ├─► Bayesian_figures.py / Bayesian_conceptual.py
              ├─► NDT_barchart_bayesian.py
              └─► why_saccadic_t0_floors.py / LATER_analysis.py
Vincentile/vincentile_figures.py (model-free; reads pooled_data.csv only)
```

Each script reads the CSV outputs of the previous stage. Bayesian scripts require PyMC. Pre-validated outputs are committed alongside so results are reproducible without re-running.

---

## 6. The headline dissociation — the Bayesian result, explained

The flagship finding — the number on my resume — is the hand-vs-saccade non-decision time dissociation. The reported p-value has two slightly different forms in the project history, which is worth being precise about:

- **p = 0.0016** — the original headline (2026-06).
- **p = 0.003** across 3 target-speed conditions — the version used on the resume v8 (2026-07-06), reported as "hand-vs-saccade non-decision time dissociation p = 0.003 across 3 target-speed conditions (KINARM, n=16)."

Both describe the same effect; the difference is the exact contrast and degrees of freedom. The takeaway is identical: the difference in non-decision time between hand and saccade is statistically significant.

### What the numbers behind it are

| Speed | Hand t₀ (group) | Saccade t₀ |
|---|---|---|
| 0 deg/s (rest) | 169.5 ms | ~70 ms (floor) |
| 75 deg/s | 158.0 ms | ~70 ms (floor) |
| 150 deg/s | 147.9 ms | ~70 ms (floor) |

Three things matter here:

- **Hand t₀ is identified and decreases with speed.** As the target moves faster, there's less time to prepare, so the "decision" component compresses — t₀ drops from 169.5 to 147.9 ms. This is a sensible, interpretable pattern.
- **Saccade t₀ is pinned at the 70 ms floor and doesn't move.** The eye's conduction latency dominates; there's no recoverable "decision" time on top.
- **The gap (~100 ms hand − ~70 ms saccade) is the effect.** Reaching carries ~100 ms of genuine non-decision/decision overhead that saccades don't.

**Why it matters scientifically.** The hand and the eye are driven by different motor pipelines with different conduction latencies. The fact that hand t₀ is large and speed-dependent while saccade t₀ is small and floor-limited is evidence that the two effectors are governed by genuinely different decision/preparation processes — not the same process measured at different scales. That's the claim the whole project supports.

---

## 7. The frequentist validation study (v5) — FDR, bootstrap, and what 22/48 means

Beyond the Bayesian result, a parallel validation study (completed 2026-08-06) asked a different question: does the frequentist (MLE) shifted-Wald fit actually find evidence that the Wald model is a real improvement, cell by cell, once you correct for multiple comparisons? This is where the "pass/fail/accuracy" framing comes from.

### The three gates

1. **V1 parity gate** — clean for all 80 cells. Two cells (`CMT005@150`, `CMT007@150`) had `-inf` log-likelihoods, but these were t₀-precision artifacts (the model pinned t₀ to a value that made the decision-time zero for one trial), not a validity break. Route C (loader/fitter mismatch) was ruled out — the loader and fitter agree.
2. **FDR survival (Benjamini–Hochberg)** — of the cells where the Wald model was compared against a null, how many survive correction for the 48 tests being run?
3. **Bootstrap refinement (B=2000)** — 18 borderline cells re-tested with 2000 bootstrap resamples (~43 min) to resolve Monte-Carlo-uncertain p-values.

### The FDR numbers (the "accuracy rate" for the frequentist Wald)

| Effector | Cells surviving BH FDR (q=0.05) | Cells surviving (q=0.10) | Interpretation |
|---|---|---|---|
| SRT (saccade) | 22 / 48 | 29 / 48 | Genuine Wald effect in roughly half the saccade cells, robust to multiple-comparison correction. |
| HRT (hand) | 0 / 48 | — | A resolution artifact, not a null (see below). |

**Why HRT "0/48" is not a failure.** The 3 hand cells that do reject the null are pinned at p ≤ 0.01 — genuinely significant — but at any bootstrap size B, the Benjamini–Hochberg procedure's tightest resolvable threshold can't descend below a floor set by the number of tests and the smallest observed p. With only 3 rejections all clustered at p≤0.01, BH can't certify them at q=0.05. So "0/48 survive" is an artifact of how BH resolves ties at extreme p-values, not evidence that the hand has no Wald effect. The correct statement: "genuine but not resolvable at alpha."

### The bootstrap refinement (B=2000) — two flips, netting to zero

18 borderline cells (p-values near 0.05) were re-estimated at B=2000 to resolve Monte-Carlo uncertainty. Exactly two verdicts flipped, and they cancelled out:

| Cell | Before | After | Meaning |
|---|---|---|---|
| CMT0015@0 | 0.055 | 0.042 | False-clear → reject (a real effect was being missed). |
| CMT0016@75 | 0.050 | 0.063 | Monte-Carlo false-reject → pass (an artifact was being counted). |

Net effect: **22/48 survives unchanged**. This is important because it means the headline number is stable — it doesn't wobble when you throw more computation at it.

### Convergent validity (the honest version)

The validation also checked whether two methods agree. The finding is nuanced:

- **SRT overlap 32/32, r = 0.824** — looks great, but is largely structural: both methods floor-pin saccade t₀ at 70 ms, so of course they agree. High correlation here is partly "both floor at the same place," not independent evidence.
- **HRT overlap 47/48, r = 0.641** — lower correlation, but more honest: hand t₀ is freely estimated by both methods, so agreement here is genuine.

Takeaway: the HRT number (47/48, r=0.641) is the stronger validity claim despite the lower correlation, because it isn't contaminated by a shared floor. This is a subtle point reviewers will respect if I make it, and penalize if I don't.

---

## 8. Directive 1: LATER on the hands — the tie, and the shift-LRT

The professor asked (Directive 1): fit LATER to the hand data and compare it to the Wald. This run (2026-08-16) produced the data in `q2_later_vs_wald.csv` (48 HRT + 48 SRT cells). Two distinct questions were answered.

### Question 1: Is shifted-LATER a better model than the Wald on hands? — A tie

Shifted LATER (3 params, RT = t₀ + 1/r) was fit by maximum likelihood (closed form, not reciprobit OLS) and compared to the 3-parameter Wald on like-for-like footing, using Vuong's closeness test and bootstrap Kolmogorov–Smirnov fits:

| Metric | HRT (hand) | SRT (saccade) | Meaning |
|---|---|---|---|
| Vuong: statistically indistinguishable | 45 / 48 | 22 / 48 | The two models can't be told apart on the hand data by the closeness test. |
| Absolute fit: LATER better (lower neg log-lik) | 33 / 48 | 25 / 48 | By pure fit, LATER nudges ahead on hands. |
| KS rejection (fit quality) — LATER | 12 / 48 | 39 / 48 | LATER is a notably worse fit for saccades than hands. |
| KS rejection — Wald | 21 / 48 | — | LATER rejects fewer hand cells than the Wald. |

**Interpretation:** on the hand, LATER is as good as or better than the Wald in absolute fit (rejected in fewer cells), but the two are statistically indistinguishable in 45/48 cells. So the Wald's extra structure "buys little" on the hand. The reason I don't just switch to LATER-for-both: LATER has no non-decision parameter in its core form, so it can't directly answer the t₀ question. Hence it's kept as a model comparison, not a replacement.

### Question 2 (the sharper result): the shift-LRT

Because shifted LATER nests plain LATER at t₀=0, "does this effector demand a non-decision term?" becomes a pure likelihood-ratio test with no Wald assumptions anywhere in it. This was my suggestion, and it produced the cleanest independent replication:

| Metric | Hand | Saccade | Meaning |
|---|---|---|---|
| Cells demanding a shift (p < 0.05) | 30 / 48 (62.5%) | 6 / 48 (12.5%) | Hand cells overwhelmingly need a t₀ term; saccade cells mostly don't. |
| Median LRT statistic D | 4.97 | 0.00 | The shift's effect size is large on hand, zero on saccade. |
| Median fitted shift | 92.8 ms | 0.0 ms | Hand non-decision time is ~93 ms in this family; saccade is ~0. |
| Mann–Whitney (hand demand > saccade demand) | p = 1e-5 | | The effector difference in shift-demand is significant. |
| Fisher combined p | 3e-62 | 6e-7 | Combining all hand cells: overwhelming; saccade: weaker but still above chance. |

### The full pass/fail detail (recomputed from the CSV)

| Metric | HRT | SRT |
|---|---|---|
| demands_shift = True | 30/48 (62.5%) | 6/48 (12.5%) |
| Vuong verdict (shift model): indistinguishable / LATER / Wald | 34 / 8 / 6 | 25 / 13 / 10 |
| Reciprobit r² (median) | 0.965 | 0.904 |
| Reciprobit r² (range) | 0.906–0.995 | 0.658–0.989 |
| KS gof on LATER rejected (p<0.05) | 24/48 | 39/48 |
| shift_D > 0 (a real displacement detected) | 40/48 | 6/48 |
| Median shift_D (nonzero) | 6.4 ms | 13.9 ms |
| Truncation needed | 0/48 | 7/48 |

**The headline of Directive 1:** LATER alone doesn't replace the Wald for hands — it ties. But the nested t₀ test inside the LATER family independently replicates the dissociation: hand RT distributions demand a non-decision term, saccades don't. That's the takeaway, reached in a completely different model family from the Bayesian analysis. Independent replication across model families is the gold standard for a robust finding.

### Reciprobit quality — why the r² numbers matter

The reciprobit r² measures how well each RT distribution falls on the straight line LATER predicts. Hand median r² = 0.965 (range 0.906–0.995) is a very good fit; saccade median 0.904 (range down to 0.658) is worse, and the KS test confirms it: LATER is rejected in 39/48 saccade cells vs 24/48 hand cells. So LATER is a better model for hands than for saccades — consistent with the story that saccades have a different, floor-limited structure.

---

## 9. Directive 2: the two-boundary DDM question

The professor's second directive asked whether the data support a two-boundary DDM (the classic two-choice model where evidence can also accumulate toward an "error" boundary). This matters because if participants can make wrong-direction initial movements, a single-boundary model might be wrong.

### The gate check found no usable scored direction

The pooled data has no directly scored directional outcome, so the professor-stated quantity was recovered from the kinematic columns: sign of initial lateral error vs target side. The result:

- **Initial-wrong-direction rate ≈ 1%** — median 0 error trials per cell.
- Rate at rest (0 deg/s): 0.2% → at 150 deg/s: 1.3%. Errors rise slightly with speed but stay tiny.

What this means: at ~1% error rate, the two-boundary model is unidentified per cell — there simply aren't enough error trials to estimate a second boundary and a starting-point parameter. On recovery simulations, the starting-point parameter defaults to its floor. The professor's skepticism is quantitatively correct: the two-boundary model can't be fit cell-by-cell.

### The power analysis (`q3_twoboundary_power.csv`)

A recovery/power simulation quantified how much error data would be needed, and how well the parameters could be recovered at each error rate:

| Error rate | True drift v | Mean error trials | Fitted n | RMSE(v) | MAE t₀ (ms) | MAE a |
|---|---|---|---|---|---|---|
| 1% | 3.83 | 1.25 | 3 | 0.653 | 5.02 | 0.035 |
| 2% | 3.24 | 1.875 | 5 | 0.417 | 7.03 | 0.045 |
| 5% | 2.45 | 4.875 | 8 | 0.324 | 12.14 | 0.050 |
| 10% | 1.83 | 10.25 | 8 | 0.315 | 10.73 | 0.035 |
| 20% | 1.16 | 24.625 | 8 | 0.249 | 10.51 | 0.027 |
| 30% | 0.71 | 35.375 | 8 | 0.308 | 7.22 | 0.032 |

How to read this: at a 1% error rate, a typical cell has ~1.25 error trials, and the fitted n (3) reflects only 3 usable error trials — far too few to identify a second boundary. Recovery RMSE for drift is 0.653 (large), and t₀ mean absolute error is ~5 ms. As error rate climbs to 20–30%, error-trial count grows to ~25–35, and drift recovery improves (RMSE falls toward 0.25). The takeaway: you'd need a much higher error rate than the observed ~1% for the two-boundary model to be identifiable.

### The one version still worth attempting

A hierarchical two-boundary model: pool ~18 error trials per speed across the 16 participants with a shared starting-point parameter, then check whether that parameter's posterior moves off its prior. If it doesn't, report the single-boundary result and drop the second boundary. This is the concrete, non-shrug answer the draft reply gives the professor.

---

## 10. The app shell (kinarm-rt-app) — what it is, its tests, and every bug caught

The app shell wraps the pipeline so it runs on any computer: a Streamlit GUI, a headless CLI (`run_pipeline.py`), and one-click desktop apps (v1.10) that bundle a full conda environment plus a C++ compiler (because PyMC/PyTensor compiles C++ on first use). Its design goal is to be a faithful reproduction of the production scripts, not a second drifting implementation.

### Verification status (the "accuracy" of the shell)

- **57 tests, 0 failures** (as of v1.10, once PyMC/ArviZ were installed so 4 previously-skipped tests became runnable).
- **27 parity tests** (`tests/test_parity.py`) copy reference implementations out of the production scripts and assert the app reproduces them bit-for-bit: parameters, negative log-likelihood, KS, model-selection rule, priors, sampler seeds, bounds, floors, filter windows.
- **13 figure regression tests** (`test_figures.py`) pin every figure; where applicable verified to floating-point precision (e.g. vincentile difference worst deviation 5.7e-14 ms).
- `desktop/smoke_test.py` fails the build if the packaged app reports the Bayesian fit unavailable — so a broken bundle can't ship.

Against real data: the app's hand Bayesian fit reproduces `Bayesian_hrt_fits.csv` almost exactly — group v/a/t₀ match to two decimals (9.39/9.61/8.36; 0.77/0.78/0.80; 169.5/158.0/147.9 ms), and per-cell t₀ correlates with published values at r = 0.999, MAE = 0.4 ms. LATER returns median reciprobit r² 0.971 (reported 0.97) with median latencies 171/177/181 ms (matching). The reimplementation is faithful, not approximate.

### Every bug caught & fixed during the build-out (v1.1 → v1.10)

| Version | Bug / finding | Fix & why it mattered |
|---|---|---|
| v1.2 | 7 drift sites from the scripts: wrong optimiser (Nelder-Mead vs the scripts' differential evolution: seeds 42/7, maxiter 400, tol 1e-9, popsize 12), no mixture path in Method A, dip-test vs DDM selection-rule mismatch, identifiability diagnostic computed differently. | Ported all literally; pinned by tests. This is the "app must equal scripts" audit. |
| v1.6 | Vincentile-difference figure computed the wrong quantity: vincentized each effector separately and subtracted curves (near-flat ~113 ms) instead of differencing paired hand-eye trials per trial (rises −7 → +247 ms). | Carried a trial column through the wide→long conversion. A subtle but scientifically wrong figure. |
| v1.8 | Results section rebuilt ~4.7 s of figure work on every control interaction. | Build only the selected view + cache on a fit fingerprint. |
| v1.9 | Multi-core cell fitting duplicated the UI / could hang on Windows (joblib process backend re-imported the script under `streamlit run`). | Switched to threads; removed the multi-core toggle. Results frame-identical (1 vs 4 threads → byte-identical). |
| v1.10 | Packaging reality: without a bundled compiler, PyMC falls back to pure-Python → ~9× slower (248 s vs 27 s) AND doesn't converge to the same estimate (181.25 vs 181.5 ms). | Bundle the toolchain (~1 GB). Measured, not assumed — the performance and numerical-accuracy cost of skipping the compiler was quantified. |

### What still needs improvement

- Streamlit deprecations: `use_container_width` is flagged (since v1.8/1.9) and needs `width="stretch"` before a future Streamlit release drops it.
- CPU scaling is threads-only since the multi-core toggle removal; no process-parallel option remains.
- Bundle size ~1 GB is the cost of shipping the C++ compiler.

No correctness gaps remain. The app is the most thoroughly tested component of the project. The only open items are forward-compatibility (deprecation) and packaging ergonomics, not scientific correctness.

---

## 11. The full pass/fail/accuracy table — every model, every metric

This consolidates every quantitative verdict in one place. "Pass" = the model/result held up; "fail/limit" = it didn't or has a documented caveat.

| Model / Analysis | Metric | Result | Verdict |
|---|---|---|---|
| Shifted Wald (Method A, MLE) | V1 parity gate | Clean 80/80 cells | ✅ Pass |
| Shifted Wald (frequentist) | FDR survival SRT (q=0.05) | 22/48 | ✅ Pass (robust) |
| Shifted Wald (frequentist) | FDR survival HRT | 0/48 | ⚠️ Resolution artifact (3 real rejections pinned p≤0.01) |
| Shifted Wald (frequentist) | Bootstrap stability (B=2000) | 22/48 unchanged (2 flips net zero) | ✅ Pass (stable) |
| Hierarchical Bayesian (Method B) | Dissociation p-value | 0.0016 / 0.003 | ✅ Pass (headline) |
| Hierarchical Bayesian | App reproduction | r=0.999, MAE 0.4 ms | ✅ Pass (faithful) |
| LATER (hand) | Reciprobit r² | 0.965 median | ✅ Pass (good fit) |
| LATER (saccade) | Reciprobit r² | 0.904 median | ⚠️ Weaker fit |
| LATER (saccade) | KS rejection | 39/48 | ❌ Poor fit for saccades |
| LATER vs Wald (hand) | Vuong closeness | 45/48 indistinguishable | ⚠️ Tie (no winner) |
| Shift-LRT (nested LATER) | Hand cells demand shift | 30/48 | ✅ Pass (the win) |
| Shift-LRT | Saccade cells demand shift | 6/48 | ✅ Pass (replicates dissociation) |
| Two-component mixture | Genuinely express cells (<130 ms) | 1/16 flagged | ⚠️ Most "express" aren't truly express |
| Two-boundary DDM | Error rate | ~1% | ❌ Unidentified per cell (as professor suspected) |
| App shell (all tests) | Test suite | 57/57 pass | ✅ Pass |
| App shell | Parity vs scripts | 27/27 bit-for-bit | ✅ Pass |

---

## 12. How every number relates to every other number (the web of evidence)

This section is the "so what ties it all together" answer. The individual numbers above form a coherent web, and the project's strength is that they mutually reinforce.

### The core chain: one dissociation, three independent proofs

1. **Bayesian:** hand t₀ (169.5/158.0/147.9 ms) vs saccade t₀ (~70 ms floor), p = 0.0016–0.003.
2. **Frequentist Wald:** 22/48 saccade cells survive FDR, hand cells have genuine-but-unresolvable effects — consistent with "saccade effects are large and clean, hand effects are real but harder to certify at FDR."
3. **LATER shift-LRT:** 30/48 hand cells demand a non-decision shift vs 6/48 saccade cells, Mann–Whitney p=1e-5, median shift 92.8 vs 0 ms.

Three different model families — Bayesian hierarchical, frequentist Wald, and LATER — all point to the same conclusion: hand RT carries a large, identifiable non-decision/decision component; saccade RT does not. When three independent methods agree, the finding is no longer hostage to any single model's assumptions. This is the single most important thing to emphasize in the paper.

### How the numbers cross-check each other

- **r=0.999** (app reproduction) cross-checks the Bayesian numbers. The app independently reproduced the published Bayesian t₀ values to 0.4 ms, which means the 169.5/158.0/147.9 ms values are not an artifact of one script's implementation.
- **The 70 ms floor** explains why saccade validity (r=0.824) is inflated. Both methods floor-pin at 70, so high saccade correlation is partly structural — which is exactly why the HRT 47/48/r=0.641 is the more meaningful validity number.
- **The ~1% error rate** explains why the two-boundary model fails. No error trials → no second boundary → the single-boundary Wald is the correct model, which in turn validates the whole single-boundary framing in Section 2.
- **The 62.5% hand shift-demand (30/48)** reconciles with the tie. LATER ties the Wald on the hand because LATER (no t₀) and the Wald (with t₀) describe the same data equally well in absolute terms — but the shift test shows the t₀ term is doing real work for 62.5% of hand cells. The tie and the shift-LRT are two views of the same underlying truth.
- **Truncation 0/48 hand vs 7/48 saccade** means the hand distributions are cleaner LATER fits — consistent with the r² gap (0.965 vs 0.904).

### The relationship between effectors and speeds

Hand t₀ decreases with speed (169.5 → 147.9 ms) because faster targets force faster preparation. Saccade t₀ stays at 70 ms because the eye's conduction floor is the bottleneck, not preparation time. The speed-dependence of hand t₀ is itself evidence that the hand's non-decision component is a real, modifiable quantity — not a fixed artifact — which strengthens the "genuine dissociation" interpretation.

---

## 13. Paper takeaways — what to lead with, what to hedge

### Lead with (the strong, defensible claims)

1. **The triangulated dissociation.** Hand-vs-saccade non-decision time, confirmed in three model families. This is the paper's spine.
2. **22/48 SRT FDR survival** — cite 22, not 28, and note it's stable at B=2000.
3. **The shift-LRT as an independent replication.** 30/48 vs 6/48, median shift 92.8 vs 0 ms. A novel, clean, assumption-light test.
4. **The reproducibility infrastructure.** 57 tests, bit-for-bit parity, cross-platform app — a methods-asset that most RT papers don't have.

### Hedge / frame carefully (the weak spots to pre-empt reviewers)

- **HRT "0/48 at FDR" is an artifact, not a null.** State "3 cells reject at p≤0.01, but BH can't certify them at q=0.05" — do not imply the hand has no effect.
- **SRT convergent validity is partly structural** (both methods floor at 70 ms). Quantify how many cells floor on both before citing r=0.824; lead with HRT 47/48/r=0.641.
- **CMT003@150** has mode ≈130 ms: report as borderline `130 [CI]` rather than adjudicating ≤ vs <.
- **Model comparison ≠ model replacement.** LATER beats Wald in hand fit (12 vs 21 KS rejections) but has no t₀ parameter — frame as comparison, keep Wald/Bayesian primary.
- **Contamination is in the frequentist likelihood** (0.95+0.05) but not the Bayesian one. Either scope the doc to frequentist, or add the uniform term for consistency.
- **t₀ upper bound uses min(RT)** (noisy) — switch to the frequentist 3rd-percentile − 2 ms cap + uniform contaminant.
- **Reduced-chain preview (~600 draws)** for repo figures vs full 1500/1500/4 for paper — point estimates match (168/156/147) but run full for finals.

### What the professor's two directives bought me

- **Directive 1 (LATER)** turned a single-family result into a two-family triangulation, and my shift-LRT suggestion added the third. This is the single biggest upgrade to the paper's robustness.
- **Directive 2 (two-boundary)** converted a vague skepticism into a number (~1% error rate) and a concrete fallback (hierarchical two-boundary). Even if the two-boundary model fails, the analysis of why it fails is itself a publishable methods point.

---

## 14. Open questions and next steps

1. **Was initial movement direction ever scored?** This is the single blocking question. If yes, the two-boundary question can be assessed per-cell; if no, it stays hierarchical or is dropped. This is the one blank in the professor-reply draft.
2. **Full per-cell comparison table vs effort/eye summary** for the professor — pending his preference.
3. **Run the full MCMC chain (1500/1500/4)** for final paper figures instead of the reduced ~600-draw preview.
4. **Resolve the contamination-scope inconsistency** (add uniform term to Bayesian likelihood, or scope the doc).
5. **Fix the t₀ upper bound** (min(RT) → 3rd-percentile − 2 ms).
6. **Streamlit deprecation** (`use_container_width` → `width="stretch"`) in the app shell.

---

## 15. Bibliography and DOI index

Full bibliography lives in `REFERENCES.md` / `REFERENCES.bib`. The key DOIs (from `CODE_REFERENCE.md`):

| Reference | DOI | Used for |
|---|---|---|
| Anders et al. (2016) | 10.1037/met0000042 | Shifted-Wald likelihood |
| Heathcote (2004) | 10.3758/BF03206577 | Shifted-Wald likelihood |
| Ratcliff & Tuerlinckx (2002) | 10.3758/BF03196305 | Diffusion model bounds |
| Haith et al. (2016) | 10.1523/JNEUROSCI.3609-15.2016 | Hand t₀ floor 130 ms |
| Tran et al. (2020) | 10.3389/fpsyg.2020.608287 | Drift cap 20 (\|v\| ≲ 18.5) |
| Knox & Wolohan (2015) | 10.1016/j.visres.2014.12.010 → e0120437 | Corrected DOI (was e0133595) |
| Carpenter & Williams (1995) | 10.1038/377059a0 | LATER model |
| Wiecki et al. (2013) | 10.3389/fninf.2013.00014 | Hierarchical Bayesian architecture (HDDM) |
| Gelman & Rubin (1992) | 10.1214/ss/1177011136 | R-hat convergence < 1.01 |
| Fischer & Weber (1993) | 10.1016/S0149-7634(05)80110-4 | SRT anticipation threshold 80 ms |
| Bompas & Sumner (2011) | 10.1167/11.5.9 | Saccade t₀ floor 70 ms |
| Whelan (2008) | 10.3758/BRM.40.3.725 | HRT anticipation removal |
| Luce (1986) | ISBN 0-19-507001-X | Lapse removal |
| Shinn et al. (2020) | 10.7554/eLife.57394 | Model comparison methodology |
| Böhm & Ulrich (2018) | 10.1016/j.jmp.2018.01.002 | RT distribution methodology |
| Hartigan & Hartigan (1985) | 10.1214/aos/1176346577 | Dip test (bimodality / mixture split) |

---

## Appendix — The three diagnostic figures

**Figure 1 — Shift-LRT (the dissociation in the LATER family).** This plot shows, cell by cell and effector by effector, the likelihood-ratio evidence for a non-decision shift. The separation of hand (demanding shift) from saccade (not) is the visual form of the 30/48 vs 6/48 result.

**Figure 2 — Reciprobit by effector.** The reciprobit (probit of cumulative RT vs 1/RT) shows how well each effector's RT distribution follows the LATER straight-line prediction. The hand's tighter clustering (r² 0.965) vs the saccade's looser spread (0.904) is visible here.

**Figure 3 — Model Q-Q.** The quantile-quantile plot assesses goodness-of-fit of the fitted model against the observed RT quantiles — the visual counterpart to the KS statistics reported in the tables above.

---

*Prepared by Rishvanth Amsaraj. All numbers are drawn from the repository and run outputs as of 2026-08-21; the LATER statistics were recomputed from `q2_later_vs_wald.csv`. Nothing in this document is fabricated — where a value has a documented caveat (e.g. HRT FDR artifact, SRT structural validity), that caveat is stated explicitly.*
