# SNL RT Research — Paradigm 2 (CIR) Technical Breakdown

**Rishvanth Amsaraj · 2026-09-24 · data: 16 × `CIR*_TRIAL_Summary_v0_1_36.csv`**

Every Paradigm 2 result below is computed from the result tables in `Code/` by `make_breakdown.py` (re-run it after any refit). Paradigm 1 reference values come from the committed `Current Pipeline/` tables; data-structure counts come from `pooled_data_P2_audit.txt`.

---

## 1. Bottom line

1. **Same pipeline, unchanged.** The single-boundary shifted-Wald pipeline — Method A (MLE with 5% contamination) and Method B (hierarchical Bayesian, NUTS 1500/1500/4) — was run on Paradigm 2: 16 participants × 4 speeds (75/100/125/150 deg/s), 10,158 hand and 10,017 saccade trials after the Paradigm 1 RT windows. Only configuration changed (speeds, block type, input file, figure layout).
2. **The code and environment are verified.** Run on the Paradigm 1 data, the original scripts in this environment reproduce the published fits exactly for Method A (48/48 hand and 48/48 saccade cells, identical model choices) and to sampling error for Method B (hand t₀ r = 0.9995, largest cell difference 1 ms; group 169.4/158.0/148.0 vs published 169.5/158.0/147.9 ms).
3. **Hand t₀ is identified, as in Paradigm 1.** 0 divergences, max R-hat 1.003, 0/64 Bayesian cells at the 130 ms floor; the floor sweep gives a hand median slope of 0.00 (Paradigm 1: 0.00); no Bayesian hand interval reaches the fastest-RT ceiling.
4. **Hand t₀ does not change reliably between moving speeds — in either paradigm.** Hand t₀ = 146.7 / 147.0 / 151.7 / 150.6 ms. Friedman p = 0.026 (permutation p = 0.022), but the change is small (+3.9 ms from 75 to 150 deg/s [0.8, 6.9]), the per-participant trend is not reliable (p = 0.074), Method A shows nothing (Friedman p = 0.398), and raw median HRT barely moves (-2.8 ms, p = 0.059). Paradigm 1's -10.1 ms over the same 75 → 150 range was not reliable either (p = 0.066; raw HRT +3.2 ms), and in both paradigms the t₀ shift is offset by an opposite shift in decision time (r = -0.80 in Paradigm 1, -0.80 here). Paradigm 1's robust effect is the stationary → moving step (t₀ -11.5 ms, p = 0.005; raw HRT -10.2 ms, p < 0.001), which Paradigm 2 has no condition to test (see `P2_Results_and_P1_Comparison.md` §7).
5. **Saccadic t₀:** still not identifiable, but for a new reason — 10/15 participants' 95% intervals reach the 70 ms floor and 6/15 reach the model's ceiling, the participant's fastest saccade − 1 ms (2 reach both); only CIR014 (94 ms [76, 107]) is clear of both. In Paradigm 1 all 14 reached the floor and none the ceiling. Reporting saccadic t₀ as fixed at 70 ms remains the defensible choice. Per cell (Method A) the same two bounds show up: 25/52 single cells sit on the 70 ms floor, 10 against the cell's fastest saccade, and 17 in between.
6. **Left vs Right changes where the hand goes, not when it starts.** Right − Left median HRT +4.8 ms (p = 0.453), no consistent t₀, v or a difference, but signed error is larger for Right targets by +13.2° (+10.0° at the HRT50 measurement point, close to the ~10° you described).
7. **The extraction's QA flags don't matter.** Excluding the 48 flagged saccades changes saccadic t₀ by at most 1.6 ms and flips the model in 1/25 affected cells (a borderline one).

---

## 2. What was run

The full Paradigm 1 run order, plus five supplementary scripts (marked ✚) and one validation script.

| Step | Script | Output |
|---|---|---|
| input | `build_pooled_data_P2.py` | `pooled_data_P2.csv` (10,239 rows, all 121 source columns + `Participant`), `pooled_data_P2_audit.txt` |
| Method A | `DDM_fit.py` | `DDM_hrt_fits.csv`, `DDM_srt_fits.csv` |
| Method B | `Bayesian_HRT_fit.py`, `Bayesian_SRT_fit.py`, `Bayesian_SRT_ndt.py` | `Bayesian_hrt_fits.csv`, `Bayesian_hrt_ndt.csv`, `Bayesian_srt_fits.csv`, `Bayesian_srt_ndt.csv`, `Bayesian_srt_ndt_cells.csv` |
| figures | `DDM_figures.py`, `DDM_conceptual.py`, `NDT_barchart.py`, `Bayesian_figures.py`, `Bayesian_conceptual.py`, `NDT_barchart_bayesian.py`, `vincentile_figures.py` | summaries, 16 schematics, NDT charts, 4 vincentile figures |
| SRT diagnostics | `SRT_identifiability_check.py`, `SRT_fixed_t0_analysis.py`, `why_saccadic_t0_floors.py` | floor sweep, fixed-t₀ sensitivity, shape mechanism |
| ✚ | `dissociation_tests.py` | the Paradigm 1 app's speed battery (Friedman, participant bootstrap, permutation) + a per-participant slope test for 4 ordered speeds |
| ✚ | `HRT_floor_control.py` | hand negative control for the floor sweep (Paradigm 1's v3 control) |
| ✚ | `SRT_QA_flag_sensitivity.py` | refits every cell touched by the extraction's eye QA flag under both rules |
| ✚ | `direction_check.py` | Left vs Right: RT, Method A t₀/v/a, lead share, signed error |
| ✚ | `parameter_recovery_P2.py` | simulate-and-refit at Paradigm 2's cell size |
| validation | `P1_parity_check.py` | environment reproduces the published Paradigm 1 fits |

`run_all_P2.py` runs everything in order (see `RUN_GUIDE_P2.md`).

**What changed in the ported scripts** (full unified diff: `Code/Validation/PORT_DIFF_P1_to_P2.diff`):

- **Configuration only in the model code of every fitting script** — `SPEEDS = [75, 100, 125, 150]`, `BlockType == "P2"` (Paradigm 1: `"I"`), input `pooled_data_P2.csv`. Likelihood, priors, bounds, floors (hand 130 / saccade 70 ms), RT windows (150–800 / 80–600 ms), optimiser seeds and settings, the saccade mixture rule, and the sampler settings and seeds are byte-identical. The only other change inside a fitting script is plotting: the forest plot in `Bayesian_SRT_ndt.py` moved into a function, is judged against both bounds, and has a `--replot` mode that redraws it from the saved table without refitting.
- **One diagnostic extended.** `SRT_identifiability_check.py` now separates cells whose t₀ is stuck at their fastest saccade (orange) from genuinely identified ones — the Paradigm 1 version counted them as identified — and `HRT_floor_control.py` excludes them from the slope comparison.
- **Four-speed figure layout** — tick ranges, panel counts, label positions; 75 and 150 deg/s keep their Paradigm 1 colours, 100 is amber and 125 is purple.
- **Text that was typed in is now computed.** Paradigm 1 figures hard-coded their own results (e.g. "0 / 48 cells floored", "170→158→148 ms", "0% Poor") or claims that the data had to confirm ("pinned at the 70 ms floor", "conclusions are robust"). Those are now calculated, so a figure cannot state something its data do not show.
- **Stale labels fixed — worth back-porting to Paradigm 1:** `DDM_figures.py` and `Bayesian_figures.py` still draw the retired 100 ms hand floor (the floor is 130 ms); the `Bayesian_srt_ndt` forest-plot title says "ESTIMATED, not fixed", contradicting the reported fixed-at-70 conclusion; a legend in `SRT_fixedt0_sensitivity` hid a data point; a comment in `Bayesian_HRT_fit.py` said 100 ms.
- **Two tables are now saved that Paradigm 1 only plotted:** `SRT_identifiability.csv`, `SRT_fixedt0_sensitivity.csv`.

---

## 3. Data

- 16 participants (CIR001–CIR017, no CIR004), 640 trials each (CIR011: 639 — one 150 deg/s Right trial missing). 16 blocks of 40 trials; **all four speeds interleaved within every block**, 80 trials per speed × direction. One block type (`P2`).
- Paradigm 1 filter rule applied unchanged: block type, then hand 150–800 ms / saccade 80–600 ms. Hand: 10,158 of 10,238 trials with an RT kept (151–160 per cell). Saccade: 10,017 of 10,046 kept (148–160 per cell). Paradigm 1 had ~120 per cell.
- `GazeSRT_ms` equals the timestamp-based SRT exactly (the frame-based variant differs by up to 4 ms).
- **QA flags.** Every hand exclusion in the extraction (HRT ≤ 100 ms, one manual QA failure) is already outside the 150 ms window. For the eye, the extraction marks the whole trial excluded when HRT ≤ 100 ms, which removes 48 in-window saccades; the Paradigm 1 rule keeps them (Paradigm 1 kept 33 such saccades), so the production fits keep them. Sensitivity (`SRT_QA_flag_sensitivity.csv`): 25 affected cells, model choice unchanged in 24; single-cell t₀ changes by ≤ 1.6 ms (mean 0.1). The one flip is CIR001@75, whose single-Wald KS (0.110) sits just above the 0.10 mixture trigger.

---

## 4. Hand results

| Speed | Method A t₀ (ms, mean ± SD) | A at floor | A mean KS | Method B t₀ (ms, mean [mean 95% CI]) | B at floor | B v | B a |
|---|---|---|---|---|---|---|---|
| 75 | 148.1 ± 21.6 | 8/16 | 0.058 | 146.7 [138.8, 154.8] | 0/16 | 10.18 | 0.91 |
| 100 | 146.4 ± 14.4 | 4/16 | 0.058 | 147.0 [138.7, 155.4] | 0/16 | 10.50 | 0.92 |
| 125 | 156.5 ± 17.7 | 2/16 | 0.054 | 151.7 [142.1, 160.7] | 0/16 | 10.75 | 0.89 |
| 150 | 146.5 ± 15.9 | 5/16 | 0.057 | 150.6 [141.0, 159.9] | 0/16 | 10.89 | 0.90 |

Method B: one hierarchical model over all 64 participant × speed units, 0 divergences, max R-hat 1.003. Paradigm 1 Method B for comparison: 169.5 / 158.0 / 147.9 ms at 0 / 75 / 150 deg/s.

**Why Method A floors 19/64 cells while Method B floors none.** The recovery study (§7) simulates cells from the Method B estimates at n = 158: Method A recovers hand t₀ without meaningful bias (-5.4 to +2.2 ms) but with a per-cell RMSE of 12–14 ms, and because the true values sit only ~17–22 ms above the 130 ms floor, 8–32% of simulated cells land on the floor anyway. Method A flooring is estimation noise, not evidence that t₀ is at the floor; partial pooling removes it.

**Speed effect** (`dissociation_tests_P2.csv`; bootstrap, permutation and slope use 3000 resamples, seed 0, as in the Paradigm 1 app):

| Test | Hand, Method B | Hand, Method A | Saccade, Method A |
|---|---|---|---|
| Friedman χ², p | 9.27, 0.026 | 2.96, 0.398 | 3.59, 0.309 |
| Permutation p | 0.022 | 0.407 | 0.309 |
| 150 − 75 (ms), 95% bootstrap CI | +3.9 [0.8, 6.9] | -1.6 [-15.1, 10.9] | +5.1 [-6.3, 17.0] |
| Slope (ms per 25 deg/s), CI; negative slopes; Wilcoxon p | +1.63 [0.40, 2.79]; 4/16; 0.074 | +0.52 [-3.11, 3.96]; 5/16; 0.464 | +0.96 [-2.58, 4.61]; 10/16; 0.959 |

Reading: the Friedman and permutation tests (the Paradigm 1 tests) are significant for Method B, but what they detect is a ~4–5 ms step between 100 and 125 deg/s, not the Paradigm 1 pattern. The trend test built for four ordered speeds is not significant, and Method A — the noisier estimator — shows nothing. Decision time moves the other way (-6.3 ms from 75 to 150, r = -0.80 with the t₀ change across participants), so raw median HRT barely changes (-2.8 ms). The defensible statement is: **hand non-decision time is essentially flat across 75–150 deg/s in Paradigm 2 (within ~5 ms) — and Paradigm 1 is not reliably different over the same range** (-10.1 ms [-20.1, +1.0], p = 0.066, offset by decision time). Paradigm 1's speed effect rests on its stationary → moving step, which has no counterpart here.

**Checked and ruled out: blocked vs interleaved design.** Both paradigms interleave speeds within every block (Paradigm 1: all 3 speeds in each of 192 blocks; Paradigm 2: all 4 in each of 256). The within-Paradigm 1 analysis (`Comparison/P1_vs_P2_summary.csv`) locates Paradigm 1's effect: the stationary → moving step is robust in the model and in raw RTs (median and 10th percentile both drop ~10 ms, 15/16 and 14/16 participants), while the 75 → 150 step is not. The cohorts also differ (CMT vs CIR are different people), so a within-subject test with a stationary condition is what would settle it.

**Floor-sweep control** (`HRT_floor_control.csv`): 8/60 hand fits follow the floor, median slope 0.00 (Paradigm 1 with the same code: 5/43, 0.00). Hand t₀ is set by the data, not the floor.

---

## 5. Saccade results

| Speed | Method A single / mixture cells | A single t₀ (ms) | A single at floor | A single KS | Method B unimodal t₀ (ms, mean [mean 95% CI]) | B at floor | B div / R-hat |
|---|---|---|---|---|---|---|---|
| 75 | 12 / 4 | 86.8 | 6/12 | 0.069 | 77.6 [70.8, 86.2] | 5/12 | 0 / 1.003 |
| 100 | 14 / 2 | 88.1 | 7/14 | 0.074 | 76.4 [70.2, 86.9] | 5/14 | 0 / 1.007 |
| 125 | 13 / 3 | 85.1 | 6/13 | 0.064 | 72.1 [70.0, 78.2] | 7/13 | 5 / 1.002 |
| 150 | 13 / 3 | 91.9 | 6/13 | 0.077 | 80.1 [70.5, 93.7] | 1/13 | 0 / 1.003 |

**Two-component cells** (Method B, `Bayesian_srt_fits.csv`). The column Paradigm 1 calls `express_mode` holds each component's mean, not its mode (Paradigm 1 audit item A6, left unchanged for parity):

| Participant | Speed | fast-component weight π [95% CI] | fast mean (ms) | slow mean (ms) | fast < 130 ms | div / R-hat |
|---|---|---|---|---|---|---|
| CIR003 | 75 | 0.48 [0.22, 0.75] | 138 | 171 | no | 0 / 1.000 |
| CIR003 | 125 | 0.41 [0.08, 0.68] | 152 | 167 | no | 0 / 1.001 |
| CIR003 | 150 | 0.51 [0.18, 0.78] | 151 | 171 | no | 0 / 1.003 |
| CIR007 | 75 | 0.33 [0.23, 0.43] | 125 | 210 | yes | 0 / 1.000 |
| CIR007 | 125 | 0.34 [0.25, 0.44] | 126 | 218 | yes | 0 / 1.000 |
| CIR007 | 150 | 0.39 [0.27, 0.55] | 133 | 234 | no | 0 / 1.004 |
| CIR008 | 75 | 0.57 [0.37, 0.73] | 106 | 156 | yes | 0 / 1.001 |
| CIR008 | 100 | 0.58 [0.41, 0.68] | 105 | 163 | yes | 0 / 1.005 |
| CIR008 | 125 | 0.39 [0.21, 0.62] | 106 | 156 | yes | 0 / 1.000 |
| CIR008 | 150 | 0.56 [0.46, 0.64] | 105 | 176 | yes | 0 / 1.000 |
| CIR015 | 75 | 0.45 [0.27, 0.65] | 181 | 208 | no | 0 / 1.001 |
| CIR015 | 100 | 0.33 [0.21, 0.47] | 181 | 208 | no | 0 / 1.001 |

6/12 two-component cells have a genuinely express-range fast component (< 130 ms): CIR008 at every speed (~105 ms; median SRT 113 ms) and CIR007 at two speeds (~125 ms). CIR015's "fast" component is ~181 ms and CIR003's components are only 15–33 ms apart with wide π intervals — two-component fits, but not express saccades. CIR001's single-Wald fit is poor at every speed (KS 0.110–0.158) and no mixture qualifies; it is the least well-described participant.

**Participant-level saccadic t₀** (`Bayesian_srt_ndt.csv`, 15 participants with single cells; CIR008 has none): divergences = 0, max R-hat = 1.004. Posterior means 71–110 ms (mean of means 81 ms); the parent-distribution mean μ is 58 ms, below the floor — the data still favour sub-70 values overall. The higher individual values are ceiling-bound, not located: CIR005 80, CIR006 81, CIR016 82, CIR010 83, CIR013 91, CIR002 110 ms all run into their own fastest saccade − 1 ms (grey ticks in the figures). Conclusion: still not identifiable, but for a new reason — 10/15 participants' 95% intervals reach the 70 ms floor and 6/15 reach the model's ceiling, the participant's fastest saccade − 1 ms (2 reach both); only CIR014 (94 ms [76, 107]) is clear of both. In Paradigm 1 all 14 reached the floor and none the ceiling. Reporting saccadic t₀ as fixed at 70 ms remains the defensible choice.

**Identifiability.** The floor sweep (`SRT_identifiability.csv`) finds 23/52 single cells tracking the floor. A flat line is not proof of identification, though: 7 cells are flat because their t₀ is stuck against their fastest saccade at every floor (orange in the figure — the diagnostic inherited from Paradigm 1 counted these as identified; rerun with the same code, Paradigm 1 has 1/32 such cells). Excluding them and restricting to cells whose fastest saccade sits above every imposed floor, 19/38 (50%) follow the floor (slope > 0.7); Paradigm 1 with the same code: 9/16 (56%) — the same share (Fisher p = 0.77). The slopes split into two clusters, near 0 and near 1, so the median is not a useful summary here: it lands in the gap when the split is close to even (0.70 here, 0.99 in Paradigm 1). The earlier Paradigm 1 reference (median 0.975, 9/17) came from a different fitter without the ceiling exclusion. At the production 70 ms floor: 25 cells on the floor, 10 on the fastest saccade, 17 in between. The recovery study shows why: with saccade-shaped distributions at n = 158, a true t₀ of 90 ms is recovered on average (-5.1 to +3.5 ms bias) but with ~20 ms per-cell RMSE and 30–42% of cells on the floor; a true t₀ of 44 ms (below the floor) returns ~70–72 ms in 80–90% of cells. So per-cell saccadic t₀ in Paradigm 2 is weakly identified at best: about a third of cells sit between the bounds, and the recovery error there (~20 ms) is as large as the effect one would want to measure.

**Shape mechanism** (`why_saccadic_t0_floors`): pooled hand skew/CV 12.9 (Paradigm 1: 12.9) → shape-implied t₀ 181 ms, above the 130 ms floor; pooled saccade skew/CV 3.9 (Paradigm 1: 3.4) → implied 44 ms, below the 70 ms floor. The Paradigm 1 argument holds; the saccade distribution is somewhat more skewed than in Paradigm 1 (implied t₀ 44 vs 20 ms), but the share of cells on the floor is not significantly different (25/52 vs 21/32, Fisher p = 0.18).

**Fixed t₀ = 70 ms** (`SRT_fixedt0_fits.csv`): 52 single cells, mean KS 0.074, 88% below 0.10. Fit quality is insensitive to the fixed value (mean KS 0.077 / 0.074 / 0.074 at 50 / 70 / 90 ms). Unlike Paradigm 1, the drift-by-speed profile is not stable across the fixed value (minimum profile correlation 0.30) — but drift barely differs between speeds at any fixed t₀ (≤ 0.7 units), so there is no speed pattern in saccadic drift to be robust or fragile.

---

## 6. Left vs Right and aiming (supplementary)

Right − Left, mean across 16 participants (Wilcoxon p in brackets); "all" = each participant's mean over speeds:

| Measure | 75 | 100 | 125 | 150 | all |
|---|---|---|---|---|---|
| median HRT (ms) | +6.25 (p 0.277) | +6.22 (p 0.289) | +3.06 (p 0.532) | +3.62 (p 0.679) | +4.79 (p 0.453) |
| median SRT (ms) | +3.13 (p 0.528) | +4.53 (p 0.464) | +3.44 (p 0.532) | +2.50 (p 0.706) | +3.40 (p 0.860) |
| Method A hand t₀ (ms) | -8.24 (p 0.433) | +12.33 (p 0.211) | -0.79 (p 1.000) | -2.65 (p 0.940) | +0.16 (p 0.597) |
| Method A hand v | +0.10 (p 1.000) | -2.76 (p 0.004) | -0.77 (p 0.632) | +0.27 (p 0.980) | -0.79 (p 0.093) |
| Method A hand a | +0.20 (p 0.495) | -0.26 (p 0.083) | -0.00 (p 0.980) | +0.07 (p 0.860) | +0.00 (p 0.464) |
| signed error, SignedError_deg (°) | +11.85 (p 0.006) | +13.48 (p 0.006) | +13.60 (p 0.002) | +13.84 (p 0.001) | +13.19 (p 0.004) |
| signed error, SignedError_deg_HRT50 (°) | +9.50 (p <0.001) | +10.31 (p <0.001) | +9.92 (p <0.001) | +10.25 (p <0.001) | +10.00 (p <0.001) |

- **Movement start does not differ by direction.** HRT, SRT, and Method A t₀ and a show no reliable difference. The one significant cell (drift at 100 deg/s, p = 0.004) is 1 of 20 speed-level tests and does not recur at any other speed; treat it as chance. Pooling directions within a cell — what every production fit does — is therefore harmless.
- **Aim differs by direction.** The hand leads the target on 91–95% of Left trials and 98–99% of Right trials, and Right targets draw a larger lead by ~12–14° (SignedError_deg) or ~10° (SignedError_deg_HRT50).
- **Implication for your question (motor plan vs biomechanics vs cognition):** whatever produces the Left/Right launch-angle asymmetry does not change the time to initiate or any decision-stage parameter. It acts on the direction of the movement — its planned content or its execution — not on when it starts. The RT data cannot separate plan content from biomechanics; the submovement/kinematic analysis you are planning is the right tool.
- **Why Paradigm 1's lead-vs-lag RT comparison cannot be repeated:** lag trials are 1–9% of Paradigm 2 trials (Paradigm 1: ~35%). The continuous stand-in, the within-cell Spearman correlation of HRT with signed error, is weakly negative (-0.09, -0.10, -0.08, -0.06 at 75–150 deg/s; p 0.029, 0.029, 0.039, 0.159). **Do not interpret it yet:** the target direction in `SignedError_deg` is measured at a time that depends on when the hand moved, so an earlier start mechanically changes the signed error. It needs the exact definition of the measurement point in the v0_1_36 extraction first — and lead percentages should not be compared across paradigms until that is confirmed for both extraction versions.

---

## 7. Validation

**Paradigm 1 parity** (`P1_parity_results.csv`) — original Paradigm 1 scripts, unmodified, rerun here on Paradigm 1's `pooled_data.csv`:

| Table | Cells matched | Largest difference | Other |
|---|---|---|---|
| `DDM_hrt_fits.csv` | 48/48 | v 0.0, a 0.0, t₀ 0.0 ms, KS 0.0 | exact |
| `DDM_srt_fits.csv` | 48/48 | all stored columns 0.0 | model choice 48/48 identical |
| `Bayesian_hrt_fits.csv` | 48/48 | t₀ 1 ms (mean 0.27), CI bounds ≤ 2 ms | r = 0.9995; group 169.4/158.0/148.0 vs 169.5/158.0/147.9 |

Environment: Python 3.12, PyMC 6.3.2 / PyTensor 3.3.2 with a C compiler, ArviZ 1.3, SciPy 1.17. Method A is deterministic and matches exactly; Method B differs only by Monte Carlo error. Within Paradigm 2, a second full run of `Bayesian_SRT_ndt.py` reproduced `Bayesian_srt_ndt.csv` and `Bayesian_srt_ndt_cells.csv` byte-for-byte (fixed seeds), so the Bayesian tables are reproducible in this environment, not just close.

**Convergence of every Bayesian model in Paradigm 2:** hand (1 model): 0 divergences, max R-hat 1.003. Saccade unimodal (4 models): divergences 0, 0, 5, 0 at 75/100/125/150 (5 of 6,000 draws at 125 deg/s — few, but noted in §9), max R-hat 1.007. Two-component cells (12 models): 0 divergences, max R-hat 1.005 after relabeling. Participant-level saccadic t₀: divergences = 0, max R-hat = 1.004.

**Bayesian recovery** (`Validation/bayesian_recovery_P2.py`): one simulated Paradigm 2 dataset — the real 16 participants × 4 speeds and cell sizes, each participant's Method B estimates as truth, hand t₀ falling 10 ms from 75 to 150 deg/s with v and a fixed — fitted with the unchanged `Bayesian_HRT_fit.py`. The model **detects** the change (15/16 participants, Wilcoxon p = 0.003) but **shrinks** it (-5.9 ms, CI [-8.2, -3.4]) and moves the rest into decision time (-4.3 ms, p = 0.013, truth 0). With no true variation in either change, the two estimated changes still correlate at r = -0.75. Absolute hand t₀ comes out +6.0 ms high (RMSE 7.2 ms) and per-cell 95% intervals contain the truth in only 70% of cells. The raw simulated data move by the full amount (10th-percentile HRT -9.9 ms), which is what makes the fast-end check in §4 a valid test. Decision time from the ratio of posterior means differs from the per-draw mean of a/v by at most 0.16 ms.

**Parameter recovery, Method A** (`parameter_recovery_P2_summary.csv`; 40 simulated cells per row, n = 158, production fitter):

| Scenario | Speed | True t₀ | Mean fitted t₀ | Bias | RMSE | At floor |
|---|---|---|---|---|---|---|
| EYE-A_t0_90 | 75 | 90.0 | 93.5 | +3.5 | 20.4 | 30% |
| EYE-A_t0_90 | 100 | 90.0 | 84.9 | -5.1 | 18.0 | 40% |
| EYE-A_t0_90 | 125 | 90.0 | 89.9 | -0.1 | 20.7 | 38% |
| EYE-A_t0_90 | 150 | 90.0 | 88.0 | -2.0 | 19.6 | 42% |
| EYE-B_t0_44 | 75 | 44.0 | 70.4 | +26.4 | 26.4 | 90% |
| EYE-B_t0_44 | 100 | 44.0 | 70.8 | +26.8 | 27.0 | 90% |
| EYE-B_t0_44 | 125 | 44.0 | 71.2 | +27.2 | 27.4 | 88% |
| EYE-B_t0_44 | 150 | 44.0 | 71.7 | +27.7 | 27.9 | 80% |
| HAND | 75 | 146.7 | 146.2 | -0.5 | 14.2 | 28% |
| HAND | 100 | 147.0 | 141.6 | -5.4 | 12.4 | 32% |
| HAND | 125 | 151.7 | 153.9 | +2.2 | 13.2 | 8% |
| HAND | 150 | 150.6 | 148.0 | -2.6 | 12.2 | 18% |

---

## 8. Paradigm 1 vs Paradigm 2 at the shared speeds

| | 75 deg/s | 150 deg/s |
|---|---|---|
| Hand t₀, Method B — Paradigm 1 | 158.0 | 147.9 |
| Hand t₀, Method B — Paradigm 2 | 146.7 | 150.6 |

Different participants, so this is a between-cohort comparison. What replicates: hand t₀ identified, hand skew/CV 12.9, no reliable hand-t₀ change between moving speeds (both paradigms, once the decision-time trade-off is counted), saccadic RT rising with target speed, saccadic t₀ not identifiable at the participant level, express-dominant individuals needing two components. What differs: Paradigm 1's stationary → moving drop has no counterpart condition in Paradigm 2, and at the participant level 6/15 saccadic estimates press against the fastest-saccade ceiling (Paradigm 1: 0/14, Fisher p = 0.017). Per-cell saccade flooring and the share of fits that follow the floor are statistically the same in both. Figures: `Figures/Comparison/`.

---

## 9. Caveats and open items

1. **Measurement definitions in v0_1_36.** Needed before any aiming result is interpreted: the timing point of `HandDir_deg` / `TargetDir_deg` and of the `_HRT50` variants (now 99.5% populated, versus 25% in Paradigm 1).
2. **Inherited from Paradigm 1, deliberately unchanged for parity:** the KS < 0.10 mixture trigger is uncalibrated (the true 5% critical value at n ≈ 158 is lower); the hand and per-speed saccade Bayesian likelihoods have no contamination term (Method A and the participant-level saccade model do); `express_mode` holds the component mean.
3. **5 divergences** in the 125 deg/s saccade unimodal model (0.08% of draws); refitting with a higher `target_accept` is the standard check if a reviewer asks.
4. **The Streamlit app** hard-codes speeds 0/75/150 (`kinarm_rt/_speeds.py`), so it will not run Paradigm 2 without a small change.
5. **Posterior `.nc` files** are not included (large); the scripts regenerate them when `netCDF4` or `h5netcdf` is installed (without either, the save is skipped silently).
6. **The participant-level saccadic model's ceiling now binds.** Its upper bound is each participant's single fastest saccade − 1 ms (the Paradigm 1 audit flagged min(RT) as a noisy bound); in Paradigm 1 it never bound, here it binds for 6/15 participants. Both methods require t₀ to sit below every RT it explains (Method A rejects any t₀ above the fastest trial too), so the estimate is tied to a single trial. Letting the 5% contamination term account for trials faster than t₀ would remove that dependence, but it changes the model — worth deciding with the professors before anyone changes it.
7. **Read Bayesian hand t₀ as conservative and slightly high.** In the recovery test the model halves part of a real change, moves the rest into decision time, places absolute t₀ about 6 ms high, and its per-cell 95% intervals under-cover (70%). Group-level comparisons and the raw-RT checks are the robust evidence; per-cell intervals are not calibrated.

---

## 10. Files

`Code/` — `pooled_data_P2.csv`, `pooled_data_P2_audit.txt`, `build_pooled_data_P2.py`, `run_all_P2.py`; sub-folders `DDM/`, `Bayesian/`, `NDT/`, `SRT Analysis/`, `Vincentile/`, `Supplementary/`, `Validation/` hold each script with its output tables (`Validation/` also has the unified diff and the run logs). `Figures/` mirrors the sub-folders (PDF + PNG; vincentile figures PDF only, as in Paradigm 1). `Documents/` — this breakdown and `RUN_GUIDE_P2.md`.
