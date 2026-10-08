# Figure explanations

The explanations the KINARM RT app shows next to each figure, collected in one place. In the app, each figure page also lists the numbers behind it for the results being shown ("In these results"), computed from the tables.

## Main

### Non-decision time by speed

*Group: Main. File: `NDT_barchart_bayesian`.*

**What it shows.** The project's headline chart. Left: hand non-decision time (t₀) at each target speed from the hierarchical Bayesian model; each small dot is one participant, the large dot is the group mean, and the bars are ±1 SD. Right: one row per participant for saccadic t₀, each with its 95% interval, the 70 ms physiological floor (dotted line) and that person's ceiling (grey tick, their fastest saccade minus 1 ms).

**Why it is here.** Non-decision time is the part of a reaction time spent seeing the target and getting the movement going, before the decision itself. The project asks whether it changes with target speed, and whether hand and eye behave alike; this chart answers both at a glance.

**How to read it.** Look at the group means on the left first. If they step up or down by more than the spread of the dots, hand t₀ changes with speed. The footnote states how many participant × speed cells sit on the 130 ms hand floor; when that is none, every estimate was set by the data. On the right, check where each interval ends: an interval that reaches the dotted floor or the grey tick is held there by that bound and was not located by the data.

**What it means for the project.** Hand t₀ is measured, so its pattern across speeds is a real result: in Experiment 1 it is about 11 ms longer for the stationary target, and between moving speeds it is flat in both experiments. Saccadic t₀ runs into a bound for almost every participant, so the data cannot place it and it is reported as fixed at 70 ms. Measurable hand, unmeasurable eye is one of the project's main findings.

**Keep in mind.** The Bayesian model is conservative. In a recovery test it detected a 10 ms change but shrank it to about 6 ms and read absolute t₀ about 6 ms high, so compare groups rather than quoting single cells, and check a change against the raw reaction times (the comparison figure "Where the hand speed effect is" does this).

### Saccadic t₀ per participant

*Group: Main. File: `Bayesian_srt_ndt`.*

**What it shows.** Each participant's saccadic non-decision time from the participant-level Bayesian model, which fits one t₀ per person shared across all their speeds, with its 95% interval. The red line is the 70 ms floor, the grey ticks are each person's ceiling (fastest saccade minus 1 ms), and the dashed line is the population mean.

**Why it is here.** It is the formal test of whether saccadic t₀ can be estimated person by person. If it could, the intervals would sit in the open between the bounds and differ from person to person.

**How to read it.** For each row, see what the interval touches. Touching the red floor means the data push t₀ below 70 ms, which is physiologically impossible, so it stops at the floor. Touching the grey tick means t₀ is as high as the person's single fastest saccade allows. Touching neither means the data located it. The dashed population mean is estimated before the floor is applied; when it lies below 70 ms, the data as a whole favour values under the floor.

**What it means for the project.** When nearly every interval is held by a bound, saccadic t₀ cannot be estimated per person, which is why it is reported as fixed at 70 ms and why the project's conclusions about non-decision time and speed come from the hand.

**Keep in mind.** The ceiling is set by one trial, the fastest saccade, so a single unusually fast saccade moves it. Participants whose every cell needed the express/regular two-component model have no single t₀ and are not in this model.

### Bayesian summary

*Group: Main. File: `Bayesian_summary`.*

**What it shows.** Three panels. A: hand t₀ from Method A (each cell fitted on its own) against the Bayesian fit, for every participant × speed. B: in the saccade cells fitted with two components, the share of fast (express) saccades with its 95% interval. C: saccadic t₀ per participant, as in its own figure.

**Why it is here.** It shows what the Bayesian model changes compared with fitting each cell alone, and where express saccades are.

**How to read it.** In A, points on the red vertical line are Method A fits stuck on the 130 ms floor; their height shows where the Bayesian fit places them instead. Points near the diagonal agree between the two methods. In B, a bar well above zero with a narrow interval is a clear express component; a wide bar means the express/regular split is uncertain.

**What it means for the project.** Partial pooling lifts the floored Method A cells to plausible values without disturbing cells that were already well estimated, which is why the lab reports the Bayesian fits. Panel B shows which participants make express saccades, the very fast saccades that need a second component.

### Model schematic, hand (Bayesian) (one per target speed)

*Group: Main. File: `bayes_hrt_<speed>_degs`.*

**What it shows.** The single-boundary shifted Wald drawn at this speed's group-mean Bayesian parameters for the hand (drift v, boundary a and non-decision time t₀ in the title), with simulated evidence paths and the reaction-time distribution the model predicts.

**Why it is here.** It makes the model's three parameters concrete: how long the dead time is, how fast evidence builds, and how much evidence a response needs.

**How to read it.** The grey block on the left is t₀: nothing can happen before it ends. After it, evidence starts at the starting point and drifts towards the threshold; the slope of that drift is the drift rate v and the height of the threshold is the boundary a. Each path that reaches the threshold is one response, and the curve drawn above the threshold is the distribution of reaction times the model predicts. Step through the speeds to compare them: a wider grey block means a longer t₀, a steeper drift means faster decisions, a higher threshold means more caution.

**What it means for the project.** An illustration of the fitted model at the group level, not a plot of individual data. The numbers in its title are the group means that the other figures summarise.

### Model schematic, saccade (Bayesian) (one per target speed)

*Group: Main. File: `bayes_srt_<speed>_degs`.*

**What it shows.** The same model drawn for saccades at this speed, at the group-mean Bayesian parameters of the cells fitted with a single component.

**Why it is here.** Seen next to the hand schematic, it shows how differently the two effectors' decisions unfold.

**How to read it.** The grey block on the left is t₀: nothing can happen before it ends. After it, evidence starts at the starting point and drifts towards the threshold; the slope of that drift is the drift rate v and the height of the threshold is the boundary a. Each path that reaches the threshold is one response, and the curve drawn above the threshold is the distribution of reaction times the model predicts. Step through the speeds to compare them: a wider grey block means a longer t₀, a steeper drift means faster decisions, a higher threshold means more caution.

**What it means for the project.** Illustrates the saccade model. The saccadic t₀ in the drawing is held near the 70 ms floor and is not a measured value; compare drift and boundary between hand and eye rather than t₀.

### Reaction-time distributions

*Group: Main. File: `vincentile_results_fig1_kde_overlay`.*

**What it shows.** Smoothed distributions of every eye (blue) and hand (orange) reaction time at each target speed, pooled across participants, with their medians marked.

**Why it is here.** Models can mislead; the raw distributions cannot. This is the check underneath everything else.

**How to read it.** The horizontal gap between the two curves is how much later the hand starts than the eyes. The width of a curve is the trial-to-trial spread, and a long right tail means occasional slow responses. Compare the panels to see how target speed shifts each curve.

**What it means for the project.** The raw data every model result has to agree with: saccades start well before reaching movements, and both distributions shift with speed before any model is fitted.

### Reaction-time histograms

*Group: Main. File: `vincentile_results_fig2_histograms`.*

**What it shows.** The same reaction times as histograms, one column per speed: eye on top, hand below.

**Why it is here.** Smoothing can blur a second peak; histograms show it.

**How to read it.** Bars show how many trials fell in each 10 ms-wide bin. A second, early bump in the eye histograms is the signature of express saccades.

**What it means for the project.** Shows the raw shape of each distribution, including features a smooth curve can hide.

### Eye-to-hand lag by speed

*Group: Main. File: `vincentile_results_fig3_vincentile_by_speed`.*

**What it shows.** Hand reaction time minus eye reaction time on the same trial, with each person's trials sorted into 20 equal bins (vincentiles) and averaged across people, at each speed (mean ± SD).

**Why it is here.** It measures eye-hand coordination directly, trial by trial, without a model.

**How to read it.** Bin 1 holds the trials where the hand was earliest relative to the eyes; negative values mean the hand moved first. A curve that rises from left to right means the lag grows on the slower trials. Compare speeds by the height of each curve.

**What it means for the project.** The eyes lead the hand on almost every trial, and by more on slow trials. How much they lead changes with speed, because the eyes slow down as targets get faster while the hand does not (see the comparison figures).

### Eye-to-hand lag, all speeds

*Group: Main. File: `vincentile_results_fig4_combined_vincentile`.*

**What it shows.** The eye-to-hand lag by vincentile with all speeds together.

**Why it is here.** A single summary of eye-hand timing, independent of speed.

**How to read it.** As for the by-speed version: bin 1 is the earliest hand relative to the eyes, negative means the hand went first.

**What it means for the project.** The overall eye-hand timing relationship for the experiment.

## Diagnostic

### Why saccadic t₀ floors

*Group: Diagnostic. File: `why_saccadic_t0_floors`.*

**What it shows.** The pooled hand and eye reaction-time distributions with the non-decision time their shape implies, computed as mean − 3·SD / skewness (exact for a shifted Wald), drawn against each effector's physiological floor.

**Why it is here.** It explains the mechanism behind the main identifiability result: why the eye floors and the hand does not.

**How to read it.** Compare the red dashed line (the shape-implied t₀) with the dotted floor. To the right of the floor, t₀ can be estimated; to the left, the model needs a t₀ below what is physiologically possible and is pushed onto the floor. The box gives skewness and skew/CV; a pure Wald distribution has skew/CV = 3.

**What it means for the project.** The eye's distribution is too symmetric for its spread, which drives the model's t₀ below 70 ms. That is a property of the data's shape, so collecting more trials would not make saccadic t₀ estimable. The hand's distribution implies a t₀ well above its floor.

**Keep in mind.** Pooling across participants widens the distributions; this is a heuristic for the whole group, and the per-cell floor tests are the formal check.

### Floor test: hand vs saccade

*Group: Diagnostic. File: `HRT_floor_control`.*

**What it shows.** Every fit redone with the floor forced to several values: hand at 90 to 140 ms, eye at 40 to 90 ms. Left: each line is one hand fit, showing its t₀ at every imposed floor. Right: the distribution of floor-tracking slopes for hand and eye.

**Why it is here.** The cleanest test of identifiability: move the floor and see whether the answer moves with it.

**How to read it.** A flat line on the left means t₀ ignores the floor, so the data set it. A line that follows the dashed diagonal means t₀ copies whatever floor is imposed, so the data say nothing. On the right, a slope near 0 is a data-determined t₀ and a slope near 1 is a floor-determined t₀; the legend counts the fits above 0.7. Only fits whose fastest response sits above every floor tested are used, and fits stuck at their fastest response are left out.

**What it means for the project.** Hand fits rarely follow the floor, saccade fits often do, in both experiments. This is the direct evidence that hand t₀ is a measurement and saccadic t₀ is not.

**Keep in mind.** It uses the per-cell Method A fitter, because the test needs many refits; the conclusion concerns the data, not the method.

### Saccade floor sweep

*Group: Diagnostic. File: `SRT_identifiability`.*

**What it shows.** Every single-component saccade cell refitted at floors from 40 to 90 ms. Left: one line per cell. Right: the histogram of floor-tracking slopes.

**Why it is here.** It is the saccade half of the floor test, with every cell shown.

**How to read it.** Red lines follow the floor (t₀ set by the floor), orange lines are stuck at the cell's fastest saccade (t₀ set by the ceiling), green lines are genuinely identified. The title counts each kind.

**What it means for the project.** Shows, cell by cell, how much of saccadic t₀ comes from the bounds rather than from the data.

### Fixed saccadic t₀ sensitivity

*Group: Diagnostic. File: `SRT_fixedt0_sensitivity`.*

**What it shows.** Saccade fits with non-decision time fixed at 50, 70 or 90 ms instead of estimated: A, mean drift rate by speed under each value; B, mean fit quality (KS statistic).

**Why it is here.** If t₀ cannot be estimated, does the value we assume matter? This answers that.

**How to read it.** In B, equal bars mean the data fit equally well whichever t₀ is assumed, so the data cannot choose between them. In A, lines that move with the assumed t₀ show that drift trades off against t₀; the profile correlation in the title says whether the pattern across speeds survives the choice.

**What it means for the project.** Supports reporting saccadic t₀ as a fixed value, and warns against reading speed differences in saccade drift when they depend on the assumed t₀.

## Method A

### Non-decision time by speed (Method A)

*Group: Method A. File: `NDT_barchart`.*

**What it shows.** Hand and saccadic t₀ by speed from Method A, which fits each participant × speed on its own by maximum likelihood (group mean ± 1 SD, participants as dots).

**Why it is here.** It shows what the analysis looks like without partial pooling, and why the lab moved to the Bayesian fits.

**How to read it.** Dots on the dotted floor line are fits stuck at the floor (130 ms hand, 70 ms eye). Compare with the Bayesian NDT chart.

**What it means for the project.** Kept for comparison: many hand fits sit on the floor here, which the Bayesian fit corrects. Its speed pattern is noisier and its test less sensitive than the Bayesian one.

### Method A summary

*Group: Method A. File: `DDM_summary`.*

**What it shows.** A: fit quality (KS statistic) for every hand cell; B: fit quality for every saccade cell, sorted, with the cells where a two-component (express/regular) model was selected marked red; C: Method A hand t₀ by speed.

**Why it is here.** A model's conclusions are only as good as its fit; this is the fit check.

**How to read it.** KS measures the largest gap between the fitted and observed distributions; below 0.10 is a good fit and the dotted lines mark the thresholds. In B, each red point drops below the grey line where the two-component model fixes a bimodal cell.

**What it means for the project.** The record of how well the shifted Wald describes each cell, and of which saccade cells need the express component.

### Model schematic, hand (Method A) (one per target speed)

*Group: Method A. File: `ddm_hrt_<speed>_degs`.*

**What it shows.** The model drawn at this speed's Method A group means for the hand.

**Why it is here.** The Method A version of the schematic, for comparison with the Bayesian one.

**How to read it.** The grey block on the left is t₀: nothing can happen before it ends. After it, evidence starts at the starting point and drifts towards the threshold; the slope of that drift is the drift rate v and the height of the threshold is the boundary a. Each path that reaches the threshold is one response, and the curve drawn above the threshold is the distribution of reaction times the model predicts. Step through the speeds to compare them: a wider grey block means a longer t₀, a steeper drift means faster decisions, a higher threshold means more caution.

**What it means for the project.** Illustration only; compare with the Bayesian schematic of the same speed to see the effect of partial pooling.

### Model schematic, saccade (Method A) (one per target speed)

*Group: Method A. File: `ddm_srt_<speed>_degs`.*

**What it shows.** The model drawn at this speed's Method A group means for saccades.

**Why it is here.** The Method A version of the saccade schematic.

**How to read it.** The grey block on the left is t₀: nothing can happen before it ends. After it, evidence starts at the starting point and drifts towards the threshold; the slope of that drift is the drift rate v and the height of the threshold is the boundary a. Each path that reaches the threshold is one response, and the curve drawn above the threshold is the distribution of reaction times the model predicts. Step through the speeds to compare them: a wider grey block means a longer t₀, a steeper drift means faster decisions, a higher threshold means more caution.

**What it means for the project.** Illustration only; the saccadic t₀ shown is held near the floor.

## Deprecated

### LATER reciprobit plots

*Group: Deprecated. File: `LATER_reciprobit`.*

**What it shows.** Saccade latencies on a reciprobit plot (reciprocal latency against probit cumulative probability) for a participant with regular saccades and one with many express saccades, and the LATER model's median latency by speed.

**Why it is here.** LATER is the classic saccade model. This shows what it says about these saccades, and why the lab set it aside.

**How to read it.** LATER predicts a straight line on a reciprobit plot: the closer the points lie to the fitted line, the better the model describes that person. A second, steeper line at short latencies marks express saccades. The right panel shows how median latency changes with speed.

**What it means for the project.** LATER describes saccade latencies well, but its parameters (rate mean and spread) cannot be compared with the hand model's drift, boundary and t₀, so it cannot answer the project's hand-versus-eye questions.

**Keep in mind.** Deprecated: kept so its output can be inspected, not used for conclusions.

### Two-boundary vs single-boundary

*Group: Deprecated. File: `two_boundary_verdict`.*

**What it shows.** The committed Experiment 1 comparison of the two-boundary drift-diffusion model, which also models whether the hand led or lagged the target, against the single-boundary Wald for the hand.

**Why it is here.** It documents that the single-boundary model was tested against the main alternative, not assumed.

**How to read it.** The panels compare how well each model fits the reaction-time distributions and how t₀ is recovered when the true answer is known.

**What it means for the project.** The two-boundary model fits the reaction times worse and inflates t₀ when the data lack real two-boundary structure, which is why the single-boundary Wald was kept.

**Keep in mind.** Deprecated: the evidence for the model choice, not a current analysis.

### Two-boundary readiness (hand)

*Group: Deprecated. File: `two_boundary_readiness_hand`.*

**What it shows.** Whether the hand data have the structure a two-boundary model needs: in each participant × speed cell, the share of the less common response (lagging rather than leading the target) and how reaction time differs between the two.

**Why it is here.** Before testing a two-boundary model, check that the data could support one.

**How to read it.** A two-boundary model needs enough trials of both kinds; cells with a very small minority share cannot inform the second boundary.

**What it means for the project.** Experiment 1 passes this check, which is why the two-boundary model could be tested there; Experiment 2, with only 1–9% lagging trials, would not.

**Keep in mind.** Deprecated model: kept as part of the evidence.

## Experiment comparison

### Non-decision time, both experiments

*Group: Experiment comparison. File: `P1_vs_P2_NDT_bayesian`.*

**What it shows.** The Bayesian NDT chart for both experiments, drawn the same way for hand and saccade on a shared scale: Experiment 1 left, Experiment 2 right, hand on top, saccade below.

**Why it is here.** The cross-experiment version of the headline chart.

**How to read it.** Compare rows: hand t₀ by speed in each experiment (top) and saccadic t₀ by speed (bottom). The shared speeds are 75 and 150 deg/s.

**What it means for the project.** Hand t₀ is measured in both experiments; saccadic t₀ is held by the floor in both. Experiment 1's higher hand t₀ at 0 deg/s is the stationary target; between moving speeds both experiments are flat.

**Keep in mind.** The saccade p-values only reflect how many cells hit the floor at each speed and should not be read as an effect.

### Non-decision time on one axis

*Group: Experiment comparison. File: `P1_vs_P2_NDT_bayesian_overlay`.*

**What it shows.** Both experiments on a single speed axis, so the shared speeds (75 and 150 deg/s) sit side by side: teal squares for Experiment 1, orange circles for Experiment 2, big markers for group means ± 1 SD.

**Why it is here.** One axis makes the shared speeds directly comparable.

**How to read it.** Read across the speed axis; where both experiments have a speed, compare the two markers directly.

**What it means for the project.** The quickest view of how the two cohorts' non-decision times line up.

### Where the hand speed effect is

*Group: Experiment comparison. File: `P1_vs_P2_hand`.*

**What it shows.** Six panels separating what the model says about the hand from what the raw data say: model hand t₀ (A) and decision time (B) by speed; each participant's change in t₀ against their change in decision time (C); raw median (D) and fastest-10% (E) hand RT; and a test of each speed step (F).

**Why it is here.** It is the test that decides which speed effects are real.

**How to read it.** A real change in t₀ moves the whole RT distribution, including its fastest responses, so look for agreement between the model panels (A, B) and the raw-data panels (D, E). Panel F lists each step with its change and p-value.

**What it means for the project.** Only Experiment 1's stationary-to-moving step moves the raw data (about 10 ms in both the model and the fastest responses). Between moving speeds, model changes in t₀ are offset by decision time and the raw RTs stay flat in both experiments.

**Keep in mind.** The t₀/decision-time trade-off in panel C appears from estimation noise alone (recovery test), so it is not a finding by itself.

### Saccades and identifiability

*Group: Experiment comparison. File: `P1_vs_P2_saccade`.*

**What it shows.** Saccadic t₀ per participant against both bounds (A), raw median saccade RT by speed (B), where per-cell estimates land (C), and the floor test for both experiments (D).

**Why it is here.** The eye side of the comparison, including the identifiability evidence.

**How to read it.** In A, intervals touching the red floor or a grey tick are held by a bound. Panel D shows the share of fits whose t₀ follows the floor (slope above 0.7).

**What it means for the project.** Saccadic t₀ is not identifiable in either experiment, while raw saccade RT rises with target speed in both — the eye's real speed effect is in its reaction times, not in t₀.

### Eye-to-hand lag, both experiments

*Group: Experiment comparison. File: `P1_vs_P2_eye_hand_lag`.*

**What it shows.** Hand RT minus eye RT on the same trial, by vincentile at 75 and 150 deg/s, and the mean lag by speed for both experiments.

**Why it is here.** Eye-hand coordination compared across the two cohorts.

**How to read it.** Higher means the hand starts later relative to the eyes; bin 1 is the earliest hand relative to the eyes.

**What it means for the project.** The eyes lead the hand by roughly 45–75 ms. The gap shrinks with speed in Experiment 2 because saccades slow down while the hand does not.

**Keep in mind.** The trial-wise difference mixes eye and hand variability, so the bands are wide.

### Raw RT distributions at shared speeds

*Group: Experiment comparison. File: `P1_vs_P2_distributions`.*

**What it shows.** Pooled hand and eye reaction-time distributions at 75 and 150 deg/s for both experiments; dashed lines are medians.

**Why it is here.** Checks that the comparison compares like with like.

**How to read it.** Overlapping curves mean the cohorts behave alike at that speed.

**What it means for the project.** The two cohorts' raw data are very similar, so differences in model results do not come from very different data.

### Non-decision time, Method A

*Group: Experiment comparison. File: `P1_vs_P2_NDT_methodA`.*

**What it shows.** The same chart as the Bayesian comparison using the per-cell Method A fits.

**Why it is here.** The Method A version, for completeness.

**How to read it.** More cells sit on the floors.

**What it means for the project.** Shows why the lab uses the Bayesian fits.

## Earlier pipeline versions

Figures from pipeline versions 2, 2.5 and 3 and the working iterations appear under Deprecated with the version in their title. Only genuinely older figures are shown: copies identical to a current figure are recorded in pipelines/MANIFEST.json ("duplicates") rather than repeated, and the pre-correction versions of the figures fixed in October 2026 are kept because they show the mistake that was corrected.
