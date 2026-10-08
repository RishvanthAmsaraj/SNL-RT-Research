# Figure Guide — Paradigm 2 and the Paradigm 1 comparison

**SNL RT Research · 2026-10-06**

This guide explains every figure the Paradigm 2 pipeline and the cross-paradigm comparison produce: what each one shows, how to read it, the numbers on it, and where it fits in the overall argument. Each figure is tagged:

- **Main** — carries a conclusion; a candidate for publication.
- **Supporting** — context or a check behind a main figure.
- **Method A / diagnostic** — kept for completeness and parity with Paradigm 1. The lab's results use Method B (Bayesian).

Every number below is taken from the result tables (`Code/…`). The checks behind them are in `VERIFICATION_REPORT_P2.md`, and `Code/Validation/verify_claims.py` re-derives the key numbers and confirms they match the documents.

---

## 0. The big picture (what all the figures add up to)

1. **Every reaction time is split into two parts by the model:** non-decision time t₀ (seeing the target and getting the movement going) and decision time (deciding where and when to go). The model is a single-boundary shifted Wald, fitted the same way in both paradigms.
2. **Hand t₀ can be measured; eye t₀ cannot.** The hand's estimates sit well inside their allowed range. The eye's estimates sit against a bound — the 70 ms floor, or (in Paradigm 2, for some people) the ceiling set by their fastest saccade. This holds in both paradigms.
3. **Paradigm 1's hand effect is stationary vs moving.** Hand t₀ is about 11 ms longer for a stationary target, and the raw reaction times shift by the same amount. Between moving speeds, hand t₀ does not change reliably in either paradigm. Paradigm 2 has no stationary condition, so it cannot test Paradigm 1's effect — but it shows the effect does not extend to differences between moving speeds.
4. **Saccades get slower as targets get faster**, in both paradigms, so the gap between the eye and the hand shrinks with speed.
5. **The Bayesian model is conservative.** In a recovery test it detects a 10 ms change but shrinks it, moves part of it into decision time, and reads absolute hand t₀ about 6 ms high. That is why every model-based claim above is paired with a raw-data (model-free) check.

---

## 1. Conventions used in every figure

- **"Cell"** = one participant at one speed. Each cell gets its own fit.
- **Speed colours:** 0 deg/s green (Paradigm 1 only), 75 red, 100 amber, 125 purple, 150 blue. 75 and 150 are the same colour in both paradigms.
- **Paradigm colours (comparison figures):** Paradigm 1 teal squares, Paradigm 2 orange circles.
- **Bounds.** Floor = the lowest t₀ the model allows: 130 ms hand, 70 ms eye (physiological minimums). Ceiling = t₀ must be shorter than the fastest response it explains. The model therefore cannot place t₀ above a cell's fastest RT — or, in the participant-level eye model, above that person's fastest saccade minus 1 ms.
- **Two ways of being "at the floor".** Some charts count a cell as at the floor when its *estimate* is within 2 ms of the floor. Others count it when its *95% interval reaches* the floor (the floor is limiting the estimate). The second is stricter, so the counts differ — e.g. Paradigm 1 eye: 18/32 estimates on the floor, 28/32 intervals reaching it.
- **Error bars differ by figure — always check the legend.**
  - NDT charts and schematics: ±1 SD across participants (how spread out people are).
  - Comparison line plots and the step chart: 95% bootstrap interval of the group mean (how sure we are about the average).
- **Statistics.**
  - Friedman: do values differ across speeds (all speeds at once)?
  - Wilcoxon signed-rank: do participants change consistently in one direction between two speeds?
  - Bootstrap interval: the plausible range of the average change.
  - Fisher exact: do two proportions differ?
  - Unless stated, p < 0.05 is called significant; n = 16 participants per paradigm.

---

## 2. Comparison figures (`Figures/Comparison/`)

Made by `Code/Comparison/P1_vs_P2_comparison.py` (hand, saccade, distributions), `P1_vs_P2_basic_charts.py` (NDT 2×2 charts, eye–hand lag) and `P1_vs_P2_NDT_bayesian_overlay.py`. Numbers are in `Code/Comparison/P1_vs_P2_summary.csv`. The paradigms are different people, so every comparison is between groups, not a within-person replication.

### 2.1 `P1_vs_P2_hand` — where the hand speed effect lives · **Main**

**Shows.** Six panels that separate what the model says about the hand from what the raw data say.

- **A. Hand t₀ (Bayesian) by speed.**
  - Paradigm 1: 169.5 → 158.0 → 147.9 ms (0/75/150).
  - Paradigm 2: 146.7 / 147.0 / 151.7 / 150.6 ms.
  - Thin lines = participants; band = 95% interval of the group mean; grey shading = speeds both paradigms share.
- **B. Decision time (a/v) by speed.**
  - Paradigm 1: 84.0 / 85.1 / 102.1 ms — jumps at 150.
  - Paradigm 2: 90.5 / 89.4 / 84.1 / 84.2 ms — dips slightly.
- **C. One point per participant: change in t₀ (x) vs change in decision time (y) from 75 to 150.**
  - Points fall along the "full compensation" line (r = −0.80 in both paradigms): when the model lowers t₀ it raises decision time by about as much.
  - The recovery test shows this coupling appears from estimation noise alone (r = −0.75 with no true variation), so it is not a finding in itself. Its meaning is negative: the model's split between the two boxes is not trustworthy for small differences.
- **D. Raw median hand RT (no model).**
  - Paradigm 1: 246.1 → 235.9 → 239.1 ms.
  - Paradigm 2: 232.5 / 231.1 / 230.5 / 229.7 ms.
- **E. Raw 10th-percentile hand RT — the fast end of each person's distribution.**
  - Paradigm 1: 220.4 → 210.8 → 208.3 ms.
  - Paradigm 2: 205.5 / 205.6 / 206.5 / 206.4 ms.
- **F. "Which steps are real?"** Each row is one step; each colour is one measurement of the per-participant change (mean ± 95% interval), with the Wilcoxon p.

**How to read E and F.** t₀ is the dead time before *any* response. A real change in t₀ slides the whole distribution, fast end included. Changes in decision time mostly stretch or shrink the slow end. The recovery test confirms it: a true 10 ms t₀ change moved the simulated fast end by 9.9 ms. So:

| Step | t₀ (model) | decision time (model) | median RT (raw) | fast end (raw) | Reading |
|---|---|---|---|---|---|
| P1: 0 → 75 | −11.5 ms, p = 0.005 | +1.1, p = 0.60 | −10.2, p < 0.001 | −9.6, p < 0.001 | model and raw data agree: a real ~10 ms shift |
| P1: 75 → 150 | −10.1, p = 0.066 | +17.0, p = 0.013 | +3.2, p = 0.29 | −2.5 [−6.5, +2.5], p = 0.13 | raw data flat; a 10 ms t₀ drop would have moved the fast end ~10 ms, outside its interval |
| P2: 75 → 150 | +3.9, p = 0.062 | −6.3, p = 0.013 | −2.8, p = 0.059 | +0.8 [−1.2, +3.1], p = 0.47 | raw data flat; a 10 ms drop is ruled out |

**Bigger picture.** This is the central result. Paradigm 1's speed effect is a stationary-vs-moving effect. Between moving speeds, neither paradigm shows a hand t₀ change. Paradigm 2's design was sensitive enough to see one (the recovery test detected a 10 ms change in 15/16 participants, p = 0.003).

**Caveats.**
- The decision-time changes are significant in both moving steps, but the recovery test produces a significant spurious one (−4.3 ms, p = 0.013) from a pure t₀ change. Don't present them as effects.
- Decision time is computed from the posterior means of a and v. In the recovery run this differs from the per-draw average by at most 0.16 ms.

**Caption.** *Hand non-decision time and raw reaction times by target speed in Paradigm 1 (0/75/150 deg/s) and Paradigm 2 (75–150 deg/s). Only Paradigm 1's stationary-to-moving step shifts the raw distribution, including its fast end; between moving speeds, model t₀ changes are offset by decision time and the raw RTs are flat.*

### 2.2 `P1_vs_P2_saccade` — the eye and identifiability · **Main**

- **A. Participant-level eye t₀ (Bayesian), one row per person.**
  - Red line = 70 ms floor; grey tick = that person's ceiling (fastest saccade − 1 ms).
  - Paradigm 1: all 14 at the floor, none at the ceiling.
  - Paradigm 2: 10 of 15 at the floor and 6 at the ceiling (2 both); only CIR014 (94 ms [76, 107]) is clear of both.
  - The ceiling-bind difference is significant (0/14 vs 6/15, Fisher p = 0.017).
  - Why Paradigm 2 hits the ceiling: one t₀ per person must sit below their fastest saccade across all four speeds — a single, very fast trial.
- **B. Raw median eye RT (no model).**
  - Paradigm 1: 173.8 → 182.5 → 185.7 ms.
  - Paradigm 2: 176.8 → 178.9 → 181.1 → 183.6 ms.
  - From 75 to 150: +3.2 ms (P1, p = 0.011) and +6.8 ms (P2, p < 0.001), with 14 of 16 people slower in each.
- **C. Bayesian cells: does each 95% interval for t₀ run into a bound?**
  - Hand: 4/48 (P1) and 5/64 (P2) intervals reach the floor; the rest are clear.
  - Eye (single-component cells): 28/32 and 48/52 reach the floor; none reach the fastest RT. Per cell the ceiling never binds, because each cell's fastest saccade is a higher ceiling than a person's fastest across all speeds.
- **D. Floor sweep, same code on both paradigms.**
  - Each fit is redone with the floor forced to six values; a slope near 1 means t₀ just copies the floor.
  - Shown is the share of fits whose t₀ follows the floor. Only fits whose fastest response sits above every floor tested are used, and fits stuck at their fastest RT are left out.
  - Hand: 5/43 (12%) vs 8/60 (13%). Eye: 9/16 (56%) vs 19/38 (50%). The same in both paradigms (Fisher p = 1.00 and 0.77).
  - The slopes split into two clusters near 0 and near 1, which is why the share, not the median, is the right summary.

**Bigger picture.** Hand t₀ is a measurement; eye t₀ is set by a bound, so the lab reports it as fixed at 70 ms. The eye's real speed effect is in its raw RTs (panel B), not in t₀.

**Caption.** *Saccadic non-decision time is not identifiable in either paradigm: participant-level estimates sit on the 70 ms floor (Paradigm 1) or on the floor or the fastest-saccade ceiling (Paradigm 2), while raw saccadic RT rises with target speed in both.*

### 2.3 `P1_vs_P2_NDT_bayesian` (2×2) and `P1_vs_P2_NDT_bayesian_overlay` (shared axes) · **Main**

**Shows.** The lab's NDT chart for both paradigms, drawn the same way for hand and eye. Small dots = participants; big marker = group mean; bars = ±1 SD.

- The **2×2** puts Paradigm 1 left and Paradigm 2 right, with a shared scale per row.
- The **overlay** puts both paradigms on one speed axis, so 75 and 150 sit side by side.

**Numbers.**

| | Paradigm 1 | Paradigm 2 |
|---|---|---|
| Hand t₀ (ms) | 170 / 158 / 148 (0/75/150), Friedman p = 0.003 | 147 / 147 / 152 / 151, Friedman p = 0.026 |
| Hand estimates on the floor | 0 | 0 |
| Eye t₀ (ms), single-component cells | 79 / 79 / 71 | 78 / 76 / 72 / 80 |
| Eye estimates on the floor | 18/32 | 18/52 |

**How to read the eye row.**
- The eye values bunch just above 70 ms because the eye data point below the floor. The Bayesian model removes flooring caused by noise (the hand), not flooring caused by the data.
- The eye p-values (Paradigm 2: 0.006) only reflect how many cells hit the floor at each speed. Each speed is also a separate eye model with its own population. The panels say so; don't present them.
- Two-component (express) cells are excluded from the eye row because they have no single t₀ (P1 16, P2 12).

**Bigger picture.** This is the cleanest one-glance version of points 2 and 3 above. Paradigm 2's Friedman p = 0.026 reflects a 4–5 ms bump between 100 and 125 deg/s, not a trend (per-participant slope test p = 0.074).

**Caption.** *Bayesian non-decision time by target speed for hand and saccade, Paradigm 1 vs Paradigm 2 (mean ± 1 SD across participants).*

### 2.4 `P1_vs_P2_eye_hand_lag` — how much later the hand starts than the eyes · **Main**

**Shows.** Hand RT minus eye RT on the same trial, for trials with both a valid hand and eye RT.

- **A, B. At 75 and 150 deg/s:** each person's trials are sorted into 20 equal bins by lag, then the bins are averaged across people (band ±1 SD).
  - Bin 1 is negative: on those trials the hand started first.
  - At 75 the paradigms overlap. At 150, Paradigm 1's longest lags are longer.
- **C. Mean lag by speed:**
  - Paradigm 1: 74.2 (stationary) → 58.7 → 60.2 ms.
  - Paradigm 2: 54.6 → 52.2 → 49.5 → 45.5 ms.

**Bigger picture.** The eyes lead the hand by roughly 45–75 ms. In Paradigm 1 the gap closes from stationary to moving, because the hand speeds up and the eyes slow down. In Paradigm 2 it closes steadily with speed: the eyes slow (+6.8 ms from 75 to 150) while the hand barely changes (−2.8 ms). It is the model-free summary of points 3 and 4.

**Caveat.** The trial-wise difference mixes eye and hand variability, so the bands are wide.

### 2.5 `P1_vs_P2_distributions` — raw RT distributions at the shared speeds · **Supporting**

**Shows.** Pooled trial distributions (smoothed) for hand and eye at 75 and 150 deg/s; dashed lines = medians.

| Pooled median (ms) | Paradigm 1 | Paradigm 2 |
|---|---|---|
| Hand, 75 deg/s | 234 (n = 1,899) | 232 (n = 2,533) |
| Hand, 150 deg/s | 237 (n = 1,883) | 229 (n = 2,548) |
| Eye, 75 deg/s | 182 (n = 1,717) | 178 (n = 2,507) |
| Eye, 150 deg/s | 186 (n = 1,727) | 185 (n = 2,501) |

**Bigger picture.** The raw data of the two cohorts are very similar, so the differences in the model results do not come from wildly different data. The visible difference is Paradigm 1's heavier slow tail for the hand at 150 deg/s — the same feature that pushes Paradigm 1's decision time up at 150 (2.1 B).

### 2.6 `P1_vs_P2_NDT_methodA` — the same NDT chart with Method A · **Method A / diagnostic**

Same layout as 2.3, but each cell is fitted on its own.

| | Paradigm 1 | Paradigm 2 |
|---|---|---|
| Hand t₀ (ms) | 165 / 156 / 150, Friedman p = 0.105 | 148 / 146 / 156 / 146, Friedman p = 0.398 |
| Hand cells on the floor | 11/48 | 19/64 |
| Eye t₀ (ms) | 91 / 78 / 84 | 86 / 90 / 84 / 91 |
| Eye cells on the floor | 24/48 | 29/64 |

Kept to show why the lab uses Method B. Note that Paradigm 1's speed effect is only significant with the Bayesian method.

---

## 3. Paradigm 2 pipeline figures (`Figures/<folder>/`)

These mirror Paradigm 1's figure set one-for-one, with one per speed wherever Paradigm 1 had one per speed.

### 3.1 `NDT/NDT_barchart_bayesian` · **Main**

- **Left: hand t₀ by speed (Bayesian).** 147 / 147 / 152 / 151 ms, Friedman p = 0.026, 0/64 cells floored.
- **Right: participant-level eye t₀ against both bounds.**
  - 10 of 15 intervals reach the floor, 6 reach the fastest-saccade ceiling (grey ticks), 1 is clear of both.
  - The dashed line is the across-participant mean (81 ms).
- The footnote states the reporting decision: eye t₀ is fixed at 70 ms.
- **Connects to** 2.2 A and 2.3. This is the Paradigm 2 version of the chart Paradigm 1 reported.

### 3.2 `Bayesian/Bayesian_srt_ndt` — participant-level eye t₀ forest · **Main / supporting**

- The detailed version of the eye panel in 3.1. Posterior mean ± 95% interval per participant, with grey ceiling ticks.
- The dashed line is the population mean μ = 58 ms. μ sits *below* the floor because it is the centre of the model's population distribution before the 70 ms floor is applied: the data favour values under 70 ms.
- CIR008 is absent: all four of their cells are two-component.
- `python Bayesian_SRT_ndt.py --replot` redraws it from the saved table without refitting.

### 3.3 `Bayesian/Bayesian_summary` · **Supporting**

- **A. Hand t₀, Method A vs Method B per cell.** Method A floors 19/64 cells (the red line); Method B floors none. Points pulled up off the floor show partial pooling at work.
- **B. The 12 two-component eye cells: weight of the fast component (95% interval).**
  - CIR008 (fast component ~105 ms at every speed) and CIR007 (~125 ms at 75 and 125) are genuine express saccades.
  - CIR003 and CIR015 are two-component fits whose "fast" parts are 138–181 ms, i.e. not express.
  - Wide bars mean the split is not firmly identified.
- **C. Participant-level eye t₀** (as in 3.2).

### 3.4 `Bayesian/bayes_hrt_<speed>_degs`, `bayes_srt_<speed>_degs` (8 schematics) · **Supporting (conceptual)**

- **What they are:** drawings of the model at each speed's group-mean Bayesian parameters (title line). For the hand: v 10.2–10.9, a 0.89–0.92, t₀ 147–152 ms.
- **What they show:**
  - The grey block is t₀.
  - Paths start at the starting point and drift toward the threshold.
  - The curve above the threshold is the predicted RT distribution.
- They illustrate the model; they are not data plots.
- **Caveat:** the eye schematics show a t₀ that the data cannot actually pin down (72–80 ms, held near the floor).

### 3.5 `SRT Analysis/why_saccadic_t0_floors` · **Main**

**Shows.** The pooled hand and eye RT distributions with the t₀ implied by their shape.

- For a shifted Wald, implied t₀ = mean RT − 3·SD / skewness. This is the method-of-moments value, and it is exact for this distribution.
- **Hand:** skew/CV = 12.9, implied t₀ = 181 ms — above the 130 ms floor, so it is identifiable.
- **Eye:** skew/CV = 3.9, implied t₀ = 44 ms — below the 70 ms floor, so the fit is pushed onto the floor. (A pure Wald has skew/CV = 3.)
- **Bigger picture:** this is the reason eye t₀ cannot be measured. It is set by the shape of the eye distribution, so more trials would not fix it. The same mechanism and numbers (12.9; 3.4 in Paradigm 1) hold in both paradigms.
- **Caveat:** pooling across participants inflates the spread. This is a heuristic for the whole group; the per-cell fits are the formal test.

### 3.6 `SRT Analysis/SRT_identifiability` · **Supporting**

**Shows.** Each eye single-component cell (Method A fitter) refitted at floors of 40–90 ms.

- **Left:** fitted t₀ against the floor imposed.
  - Red lines track the floor (t₀ set by the floor).
  - Orange lines are stuck at the cell's fastest saccade (set by the ceiling).
  - Green lines are genuinely identified.
- **Right:** the slopes. 23/52 cells track the floor, 7 are stuck at the fastest saccade, 22 are identified.
- **Connects to** 2.2 D and 3.7, which add the eligibility rule and the hand comparison. It uses the Method A fitter because the test needs many refits; the conclusion is about the data, not the method.

### 3.7 `Supplementary/HRT_floor_control` · **Main (identifiability evidence)**

**Shows.** The floor sweep for the hand (floors 90–140 ms) next to the eye.

- **Left:** hand fits stay flat (green) as the floor moves — t₀ comes from the data.
- **Right:** histogram of slopes. 8/60 hand fits follow the floor vs 19/38 eye fits (eligible cells only, stuck-at-fastest-RT cells excluded).
- **Bigger picture:** the direct test behind "hand t₀ is measurable, eye t₀ is not" — the identifiability dissociation. Paradigm 1 gives the same result with the same code (5/43 and 9/16; see 2.2 D).

### 3.8 `SRT Analysis/SRT_fixedt0_sensitivity` · **Diagnostic**

**Shows.** Eye drift and fit quality when t₀ is fixed at 50, 70 or 90 ms instead of estimated.

- Fit quality is the same at all three (mean KS 0.077 / 0.074 / 0.074; 88% of cells below 0.10 at 70 ms), so the data cannot choose between them.
- Drift shifts with the assumed t₀, and its small speed pattern (≤ 0.7 units) is not stable across the three values (minimum profile correlation 0.30).
- **Bigger picture:** supports reporting eye t₀ as a fixed value. It also warns against reading speed differences in eye drift.

### 3.9 Vincentile figures (`Vincentile/`, PDF) · **Supporting**

- **`fig1_kde_overlay`:** eye (blue) and hand (orange) RT distributions per speed, with medians.
- **`fig2_histograms`:** the same as histograms — eye on top, hand below.
- **`fig3_vincentile_by_speed`:** HRT − SRT on the same trial, sorted into 20 bins per person, at each speed (mean ± SD). It rises from about −25 ms (hand first) to about +120–135 ms (hand much later).
- **`fig4_combined_vincentile`:** the same for all speeds together.
- **Connects to** 2.4, which compares this measure across paradigms.

### 3.10 `DDM/DDM_summary`, `DDM/ddm_<hrt|srt>_<speed>_degs`, `NDT/NDT_barchart` · **Method A / diagnostic**

- **`DDM_summary`:**
  - A. Hand fit quality: KS 0.05–0.06, 0% poor fits.
  - B. Eye fit quality, with the 12 cells where a two-component fit was selected.
  - C. Method A hand t₀ by speed: 148 / 146 / 156 / 146 ms (±1 SD).
- **`ddm_*` schematics:** as 3.4 at the Method A group means.
- **`NDT_barchart`:** Method A NDT by speed. Hand 148 / 146 / 156 / 146 ms (p = 0.398); eye 86 / 90 / 84 / 91 ms (p = 0.309), with many cells on the floors.
- Kept for parity with Paradigm 1 and as the record of fit quality. The lab's results use Method B.

---

## 4. How the figures build the argument

1. **The raw data look alike across cohorts** — 2.5, then 3.9 for the eye–hand structure within Paradigm 2.
2. **The distribution shapes predict which t₀ can be measured** — 3.5.
3. **The floor tests confirm it** — 3.7 and 2.2 D: the hand is set by data, the eye by bounds; the Bayesian per-cell and participant views are 2.2 A and C, and 3.2.
4. **Hand t₀ by speed** — 2.3 and 3.1 give the numbers; 2.1 tests which steps are real against the raw RTs.
5. **What does change with speed is in the eye** — 2.2 B and 2.4.
6. **How much to trust the model** — the recovery test in `VERIFICATION_REPORT_P2.md`. It detects a 10 ms hand change, under-estimates its size, and does not separate t₀ from decision time reliably. This is why 2.1 E–F carry the central claim.

---

## 5. Before these figures go into a paper or poster

1. **Replace the long figure titles with captions.** Suggested captions are given for the main figures above. The titles in the files are written to be self-explanatory on screen.
2. **Quote group-level comparisons.** Per-cell Bayesian intervals under-cover in simulation (70% for a nominal 95%), and absolute hand t₀ reads about 6 ms high. Differences between conditions are conservative.
3. **Do not present eye per-speed p-values or eye drift-by-speed patterns as findings** (2.3, 3.8).
4. **Say that the cohorts differ.** Paradigm 1 vs Paradigm 2 is a between-group comparison.
5. **Aiming results are not in these figures.** The Left/Right signed-error numbers in the breakdown depend on the v0_1_36 measurement definitions, which are still to be confirmed.
6. **PDFs embed fonts as TrueType** (`pdf.fonttype 42`), so text stays editable in Illustrator or Inkscape. PNGs are for slides.
