"""
explain.py -- the words shown next to every figure, table and model. Plain language, written for someone who has not
read the methods. Each figure has: what it shows, why it is shown (the question it answers), how to read it, what it
means for the project, and caveats. Numbers that depend on the data are computed from the tables (insights.py), so the
text never goes stale when the results change.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Fig:
    stem: str
    title: str
    category: str            # main | diagnostic | method_a | deprecated | compare | other
    analysis: str
    what: str
    read: str
    meaning: str
    caveat: str = ""
    per_speed: bool = False  # stem contains {s}; one file per target speed
    experiments: tuple = ("E1", "E2")
    why: str = ""


CATEGORY_LABEL = {"main": "Main", "diagnostic": "Diagnostic", "method_a": "Method A", "deprecated": "Deprecated",
                  "compare": "Experiment comparison", "other": "Other"}
ARCHIVE_LABEL = {"v1": "pipeline version 1", "v2": "pipeline version 2", "v2_5": "pipeline version 2.5", "v3": "pipeline version 3",
                 "wi": "working iterations"}

SIDE_BY_SIDE_PRESETS = [("NDT chart", "NDT_barchart_bayesian"), ("Saccadic t₀ per participant", "Bayesian_srt_ndt"),
                        ("Floor test", "HRT_floor_control"), ("Why saccadic t₀ floors", "why_saccadic_t0_floors"),
                        ("Bayesian summary", "Bayesian_summary"), ("Eye-to-hand lag", "vincentile_results_fig3_vincentile_by_speed"),
                        ("LATER", "LATER_reciprobit")]

_SCHEMATIC_READ = ("The grey block on the left is t₀: nothing can happen before it ends. After it, evidence starts at the starting "
                   "point and drifts towards the threshold; the slope of that drift is the drift rate v and the height of the "
                   "threshold is the boundary a. Each path that reaches the threshold is one response, and the curve drawn above "
                   "the threshold is the distribution of reaction times the model predicts. Step through the speeds to compare "
                   "them: a wider grey block means a longer t₀, a steeper drift means faster decisions, a higher threshold means "
                   "more caution.")

FIGURES = [
    Fig("NDT_barchart_bayesian", "Non-decision time by speed", "main", "bayesian",
        "The project's headline chart. Left: hand non-decision time (t₀) at each target speed from the hierarchical Bayesian "
        "model; each small dot is one participant, the large dot is the group mean, and the bars are ±1 SD. Right: one row per "
        "participant for saccadic t₀, each with its 95% interval, the 70 ms physiological floor (dotted line) and that "
        "person's ceiling (grey tick, their fastest saccade minus 1 ms).",
        "Look at the group means on the left first. If they step up or down by more than the spread of the dots, hand t₀ "
        "changes with speed. The footnote states how many participant × speed cells sit on the 130 ms hand floor; when that is "
        "none, every estimate was set by the data. On the right, check where each interval ends: an interval that reaches the "
        "dotted floor or the grey tick is held there by that bound and was not located by the data.",
        "Hand t₀ is measured, so its pattern across speeds is a real result: in Experiment 1 it is about 11 ms longer for the "
        "stationary target, and between moving speeds it is flat in both experiments. Saccadic t₀ runs into a bound for almost "
        "every participant, so the data cannot place it and it is reported as fixed at 70 ms. Measurable hand, unmeasurable "
        "eye is one of the project's main findings.",
        "The Bayesian model is conservative. In a recovery test it detected a 10 ms change but shrank it to about 6 ms and read "
        "absolute t₀ about 6 ms high, so compare groups rather than quoting single cells, and check a change against the raw "
        "reaction times (the comparison figure \"Where the hand speed effect is\" does this).",
        why="Non-decision time is the part of a reaction time spent seeing the target and getting the movement going, before the "
            "decision itself. The project asks whether it changes with target speed, and whether hand and eye behave alike; "
            "this chart answers both at a glance."),
    Fig("Bayesian_srt_ndt", "Saccadic t₀ per participant", "main", "bayesian",
        "Each participant's saccadic non-decision time from the participant-level Bayesian model, which fits one t₀ per person "
        "shared across all their speeds, with its 95% interval. The red line is the 70 ms floor, the grey ticks are each "
        "person's ceiling (fastest saccade minus 1 ms), and the dashed line is the population mean.",
        "For each row, see what the interval touches. Touching the red floor means the data push t₀ below 70 ms, which is "
        "physiologically impossible, so it stops at the floor. Touching the grey tick means t₀ is as high as the person's single "
        "fastest saccade allows. Touching neither means the data located it. The dashed population mean is estimated before the "
        "floor is applied; when it lies below 70 ms, the data as a whole favour values under the floor.",
        "When nearly every interval is held by a bound, saccadic t₀ cannot be estimated per person, which is why it is reported "
        "as fixed at 70 ms and why the project's conclusions about non-decision time and speed come from the hand.",
        "The ceiling is set by one trial, the fastest saccade, so a single unusually fast saccade moves it. Participants whose "
        "every cell needed the express/regular two-component model have no single t₀ and are not in this model.",
        why="It is the formal test of whether saccadic t₀ can be estimated person by person. If it could, the intervals would sit "
            "in the open between the bounds and differ from person to person."),
    Fig("Bayesian_summary", "Bayesian summary", "main", "bayesian",
        "Three panels. A: hand t₀ from Method A (each cell fitted on its own) against the Bayesian fit, for every participant × "
        "speed. B: in the saccade cells fitted with two components, the share of fast (express) saccades with its 95% interval. "
        "C: saccadic t₀ per participant, as in its own figure.",
        "In A, points on the red vertical line are Method A fits stuck on the 130 ms floor; their height shows where the "
        "Bayesian fit places them instead. Points near the diagonal agree between the two methods. In B, a bar well above zero "
        "with a narrow interval is a clear express component; a wide bar means the express/regular split is uncertain.",
        "Partial pooling lifts the floored Method A cells to plausible values without disturbing cells that were already well "
        "estimated, which is why the lab reports the Bayesian fits. Panel B shows which participants make express saccades, the "
        "very fast saccades that need a second component.",
        why="It shows what the Bayesian model changes compared with fitting each cell alone, and where express saccades are."),
    Fig("bayes_hrt_{s}_degs", "Model schematic, hand (Bayesian)", "main", "bayesian",
        "The single-boundary shifted Wald drawn at this speed's group-mean Bayesian parameters for the hand (drift v, boundary a "
        "and non-decision time t₀ in the title), with simulated evidence paths and the reaction-time distribution the model "
        "predicts.",
        _SCHEMATIC_READ,
        "An illustration of the fitted model at the group level, not a plot of individual data. The numbers in its title are the "
        "group means that the other figures summarise.", "", per_speed=True,
        why="It makes the model's three parameters concrete: how long the dead time is, how fast evidence builds, and how much "
            "evidence a response needs."),
    Fig("bayes_srt_{s}_degs", "Model schematic, saccade (Bayesian)", "main", "bayesian",
        "The same model drawn for saccades at this speed, at the group-mean Bayesian parameters of the cells fitted with a single "
        "component.",
        _SCHEMATIC_READ,
        "Illustrates the saccade model. The saccadic t₀ in the drawing is held near the 70 ms floor and is not a measured value; "
        "compare drift and boundary between hand and eye rather than t₀.", "", per_speed=True,
        why="Seen next to the hand schematic, it shows how differently the two effectors' decisions unfold."),
    Fig("vincentile_results_fig1_kde_overlay", "Reaction-time distributions", "main", "model_free",
        "Smoothed distributions of every eye (blue) and hand (orange) reaction time at each target speed, pooled across "
        "participants, with their medians marked.",
        "The horizontal gap between the two curves is how much later the hand starts than the eyes. The width of a curve is the "
        "trial-to-trial spread, and a long right tail means occasional slow responses. Compare the panels to see how target "
        "speed shifts each curve.",
        "The raw data every model result has to agree with: saccades start well before reaching movements, and both "
        "distributions shift with speed before any model is fitted.", "",
        why="Models can mislead; the raw distributions cannot. This is the check underneath everything else."),
    Fig("vincentile_results_fig2_histograms", "Reaction-time histograms", "main", "model_free",
        "The same reaction times as histograms, one column per speed: eye on top, hand below.",
        "Bars show how many trials fell in each 10 ms-wide bin. A second, early bump in the eye histograms is the signature of "
        "express saccades.",
        "Shows the raw shape of each distribution, including features a smooth curve can hide.", "",
        why="Smoothing can blur a second peak; histograms show it."),
    Fig("vincentile_results_fig3_vincentile_by_speed", "Eye-to-hand lag by speed", "main", "model_free",
        "Hand reaction time minus eye reaction time on the same trial, with each person's trials sorted into 20 equal bins "
        "(vincentiles) and averaged across people, at each speed (mean ± SD).",
        "Bin 1 holds the trials where the hand was earliest relative to the eyes; negative values mean the hand moved first. A "
        "curve that rises from left to right means the lag grows on the slower trials. Compare speeds by the height of each curve.",
        "The eyes lead the hand on almost every trial, and by more on slow trials. How much they lead changes with speed, "
        "because the eyes slow down as targets get faster while the hand does not (see the comparison figures).", "",
        why="It measures eye-hand coordination directly, trial by trial, without a model."),
    Fig("vincentile_results_fig4_combined_vincentile", "Eye-to-hand lag, all speeds", "main", "model_free",
        "The eye-to-hand lag by vincentile with all speeds together.", "As for the by-speed version: bin 1 is the earliest hand "
        "relative to the eyes, negative means the hand went first.", "The overall eye-hand timing relationship for the experiment.", "",
        why="A single summary of eye-hand timing, independent of speed."),
    Fig("why_saccadic_t0_floors", "Why saccadic t₀ floors", "diagnostic", "identifiability",
        "The pooled hand and eye reaction-time distributions with the non-decision time their shape implies, computed as "
        "mean − 3·SD / skewness (exact for a shifted Wald), drawn against each effector's physiological floor.",
        "Compare the red dashed line (the shape-implied t₀) with the dotted floor. To the right of the floor, t₀ can be "
        "estimated; to the left, the model needs a t₀ below what is physiologically possible and is pushed onto the floor. The "
        "box gives skewness and skew/CV; a pure Wald distribution has skew/CV = 3.",
        "The eye's distribution is too symmetric for its spread, which drives the model's t₀ below 70 ms. That is a property of "
        "the data's shape, so collecting more trials would not make saccadic t₀ estimable. The hand's distribution implies a t₀ "
        "well above its floor.",
        "Pooling across participants widens the distributions; this is a heuristic for the whole group, and the per-cell floor "
        "tests are the formal check.",
        why="It explains the mechanism behind the main identifiability result: why the eye floors and the hand does not."),
    Fig("HRT_floor_control", "Floor test: hand vs saccade", "diagnostic", "identifiability",
        "Every fit redone with the floor forced to several values: hand at 90 to 140 ms, eye at 40 to 90 ms. Left: each line is "
        "one hand fit, showing its t₀ at every imposed floor. Right: the distribution of floor-tracking slopes for hand and eye.",
        "A flat line on the left means t₀ ignores the floor, so the data set it. A line that follows the dashed diagonal means t₀ "
        "copies whatever floor is imposed, so the data say nothing. On the right, a slope near 0 is a data-determined t₀ and a "
        "slope near 1 is a floor-determined t₀; the legend counts the fits above 0.7. Only fits whose fastest response sits above "
        "every floor tested are used, and fits stuck at their fastest response are left out.",
        "Hand fits rarely follow the floor, saccade fits often do, in both experiments. This is the direct evidence that hand t₀ "
        "is a measurement and saccadic t₀ is not.",
        "It uses the per-cell Method A fitter, because the test needs many refits; the conclusion concerns the data, not the "
        "method.",
        why="The cleanest test of identifiability: move the floor and see whether the answer moves with it."),
    Fig("SRT_identifiability", "Saccade floor sweep", "diagnostic", "identifiability",
        "Every single-component saccade cell refitted at floors from 40 to 90 ms. Left: one line per cell. Right: the histogram of "
        "floor-tracking slopes.",
        "Red lines follow the floor (t₀ set by the floor), orange lines are stuck at the cell's fastest saccade (t₀ set by the "
        "ceiling), green lines are genuinely identified. The title counts each kind.",
        "Shows, cell by cell, how much of saccadic t₀ comes from the bounds rather than from the data.", "",
        why="It is the saccade half of the floor test, with every cell shown."),
    Fig("SRT_fixedt0_sensitivity", "Fixed saccadic t₀ sensitivity", "diagnostic", "identifiability",
        "Saccade fits with non-decision time fixed at 50, 70 or 90 ms instead of estimated: A, mean drift rate by speed under each "
        "value; B, mean fit quality (KS statistic).",
        "In B, equal bars mean the data fit equally well whichever t₀ is assumed, so the data cannot choose between them. In A, "
        "lines that move with the assumed t₀ show that drift trades off against t₀; the profile correlation in the title says "
        "whether the pattern across speeds survives the choice.",
        "Supports reporting saccadic t₀ as a fixed value, and warns against reading speed differences in saccade drift when they "
        "depend on the assumed t₀.", "",
        why="If t₀ cannot be estimated, does the value we assume matter? This answers that."),
    Fig("NDT_barchart", "Non-decision time by speed (Method A)", "method_a", "method_a",
        "Hand and saccadic t₀ by speed from Method A, which fits each participant × speed on its own by maximum likelihood "
        "(group mean ± 1 SD, participants as dots).",
        "Dots on the dotted floor line are fits stuck at the floor (130 ms hand, 70 ms eye). Compare with the Bayesian NDT chart.",
        "Kept for comparison: many hand fits sit on the floor here, which the Bayesian fit corrects. Its speed pattern is noisier "
        "and its test less sensitive than the Bayesian one.", "",
        why="It shows what the analysis looks like without partial pooling, and why the lab moved to the Bayesian fits."),
    Fig("DDM_summary", "Method A summary", "method_a", "method_a",
        "A: fit quality (KS statistic) for every hand cell; B: fit quality for every saccade cell, sorted, with the cells where a "
        "two-component (express/regular) model was selected marked red; C: Method A hand t₀ by speed.",
        "KS measures the largest gap between the fitted and observed distributions; below 0.10 is a good fit and the dotted lines "
        "mark the thresholds. In B, each red point drops below the grey line where the two-component model fixes a bimodal cell.",
        "The record of how well the shifted Wald describes each cell, and of which saccade cells need the express component.", "",
        why="A model's conclusions are only as good as its fit; this is the fit check."),
    Fig("ddm_hrt_{s}_degs", "Model schematic, hand (Method A)", "method_a", "method_a",
        "The model drawn at this speed's Method A group means for the hand.", _SCHEMATIC_READ,
        "Illustration only; compare with the Bayesian schematic of the same speed to see the effect of partial pooling.", "",
        per_speed=True, why="The Method A version of the schematic, for comparison with the Bayesian one."),
    Fig("ddm_srt_{s}_degs", "Model schematic, saccade (Method A)", "method_a", "method_a",
        "The model drawn at this speed's Method A group means for saccades.", _SCHEMATIC_READ,
        "Illustration only; the saccadic t₀ shown is held near the floor.", "", per_speed=True,
        why="The Method A version of the saccade schematic."),
    Fig("LATER_reciprobit", "LATER reciprobit plots", "deprecated", "later",
        "Saccade latencies on a reciprobit plot (reciprocal latency against probit cumulative probability) for a participant with "
        "regular saccades and one with many express saccades, and the LATER model's median latency by speed.",
        "LATER predicts a straight line on a reciprobit plot: the closer the points lie to the fitted line, the better the model "
        "describes that person. A second, steeper line at short latencies marks express saccades. The right panel shows how "
        "median latency changes with speed.",
        "LATER describes saccade latencies well, but its parameters (rate mean and spread) cannot be compared with the hand "
        "model's drift, boundary and t₀, so it cannot answer the project's hand-versus-eye questions.",
        "Deprecated: kept so its output can be inspected, not used for conclusions.",
        why="LATER is the classic saccade model. This shows what it says about these saccades, and why the lab set it aside."),
    Fig("two_boundary_verdict", "Two-boundary vs single-boundary", "deprecated", "two_boundary", experiments=("E1",),
        what="The committed Experiment 1 comparison of the two-boundary drift-diffusion model, which also models whether the hand "
             "led or lagged the target, against the single-boundary Wald for the hand.",
        read="The panels compare how well each model fits the reaction-time distributions and how t₀ is recovered when the "
             "true answer is known.",
        meaning="The two-boundary model fits the reaction times worse and inflates t₀ when the data lack real two-boundary "
                "structure, which is why the single-boundary Wald was kept.",
        caveat="Deprecated: the evidence for the model choice, not a current analysis.",
        why="It documents that the single-boundary model was tested against the main alternative, not assumed."),
    Fig("two_boundary_readiness_hand", "Two-boundary readiness (hand)", "deprecated", "two_boundary", experiments=("E1",),
        what="Whether the hand data have the structure a two-boundary model needs: in each participant × speed cell, the share of "
             "the less common response (lagging rather than leading the target) and how reaction time differs between the two.",
        read="A two-boundary model needs enough trials of both kinds; cells with a very small minority share cannot inform the "
             "second boundary.",
        meaning="Experiment 1 passes this check, which is why the two-boundary model could be tested there; Experiment 2, with "
                "only 1–9% lagging trials, would not.",
        caveat="Deprecated model: kept as part of the evidence.",
        why="Before testing a two-boundary model, check that the data could support one."),
]

COMPARE_FIGURES = [
    Fig("P1_vs_P2_NDT_bayesian", "Non-decision time, both experiments", "compare", "compare",
        "The Bayesian NDT chart for both experiments, drawn the same way for hand and saccade on a shared scale: Experiment 1 "
        "left, Experiment 2 right, hand on top, saccade below.",
        "Compare rows: hand t₀ by speed in each experiment (top) and saccadic t₀ by speed (bottom). The shared speeds are 75 and "
        "150 deg/s.",
        "Hand t₀ is measured in both experiments; saccadic t₀ is held by the floor in both. Experiment 1's higher hand t₀ at 0 "
        "deg/s is the stationary target; between moving speeds both experiments are flat.",
        "The saccade p-values only reflect how many cells hit the floor at each speed and should not be read as an effect.",
        why="The cross-experiment version of the headline chart."),
    Fig("P1_vs_P2_NDT_bayesian_overlay", "Non-decision time on one axis", "compare", "compare",
        "Both experiments on a single speed axis, so the shared speeds (75 and 150 deg/s) sit side by side: teal squares for "
        "Experiment 1, orange circles for Experiment 2, big markers for group means ± 1 SD.",
        "Read across the speed axis; where both experiments have a speed, compare the two markers directly.",
        "The quickest view of how the two cohorts' non-decision times line up.", "", why="One axis makes the shared speeds directly comparable."),
    Fig("P1_vs_P2_hand", "Where the hand speed effect is", "compare", "compare",
        "Six panels separating what the model says about the hand from what the raw data say: model hand t₀ (A) and decision "
        "time (B) by speed; each participant's change in t₀ against their change in decision time (C); raw median (D) and "
        "fastest-10% (E) hand RT; and a test of each speed step (F).",
        "A real change in t₀ moves the whole RT distribution, including its fastest responses, so look for agreement between "
        "the model panels (A, B) and the raw-data panels (D, E). Panel F lists each step with its change and p-value.",
        "Only Experiment 1's stationary-to-moving step moves the raw data (about 10 ms in both the model and the fastest "
        "responses). Between moving speeds, model changes in t₀ are offset by decision time and the raw RTs stay flat in both "
        "experiments.",
        "The t₀/decision-time trade-off in panel C appears from estimation noise alone (recovery test), so it is not a finding by itself.",
        why="It is the test that decides which speed effects are real."),
    Fig("P1_vs_P2_saccade", "Saccades and identifiability", "compare", "compare",
        "Saccadic t₀ per participant against both bounds (A), raw median saccade RT by speed (B), where per-cell estimates land "
        "(C), and the floor test for both experiments (D).",
        "In A, intervals touching the red floor or a grey tick are held by a bound. Panel D shows the share of fits whose t₀ "
        "follows the floor (slope above 0.7).",
        "Saccadic t₀ is not identifiable in either experiment, while raw saccade RT rises with target speed in both — the eye's "
        "real speed effect is in its reaction times, not in t₀.", "",
        why="The eye side of the comparison, including the identifiability evidence."),
    Fig("P1_vs_P2_eye_hand_lag", "Eye-to-hand lag, both experiments", "compare", "compare",
        "Hand RT minus eye RT on the same trial, by vincentile at 75 and 150 deg/s, and the mean lag by speed for both experiments.",
        "Higher means the hand starts later relative to the eyes; bin 1 is the earliest hand relative to the eyes.",
        "The eyes lead the hand by roughly 45–75 ms. The gap shrinks with speed in Experiment 2 because saccades slow down while "
        "the hand does not.", "The trial-wise difference mixes eye and hand variability, so the bands are wide.",
        why="Eye-hand coordination compared across the two cohorts."),
    Fig("P1_vs_P2_distributions", "Raw RT distributions at shared speeds", "compare", "compare",
        "Pooled hand and eye reaction-time distributions at 75 and 150 deg/s for both experiments; dashed lines are medians.",
        "Overlapping curves mean the cohorts behave alike at that speed.",
        "The two cohorts' raw data are very similar, so differences in model results do not come from very different data.", "",
        why="Checks that the comparison compares like with like."),
    Fig("P1_vs_P2_NDT_methodA", "Non-decision time, Method A", "compare", "compare",
        "The same chart as the Bayesian comparison using the per-cell Method A fits.", "More cells sit on the floors.",
        "Shows why the lab uses the Bayesian fits.", "", why="The Method A version, for completeness."),
]


@dataclass(frozen=True)
class ModelNote:
    what: str
    why: str
    evidence: tuple = ()
    still_useful: str = ""


MODEL_NOTES = {
    "bayesian": ModelNote(
        "A single-boundary shifted Wald (drift v, boundary a, non-decision time t₀) for each participant × speed, fitted with PyMC "
        "(NUTS, 1,500 tuning and 1,500 kept draws, 4 chains). Participants share population distributions, so each estimate borrows "
        "strength from the group. Bimodal saccade cells get an express/regular two-component version.",
        "This is the model the lab reports. It keeps hand t₀ off the 130 ms floor where single-cell fits land on it by chance.",
        ("In a recovery test with a known 10 ms hand t₀ change it detected the change in 15 of 16 participants, but shrank it to about "
         "6 ms and read absolute t₀ about 6 ms high — so group comparisons are conservative.",)),
    "method_a": ModelNote(
        "The same shifted Wald, fitted by maximum likelihood to each participant × speed on its own (differential evolution, with a 5% "
        "contamination term).",
        "Supporting only. Without pooling, many hand fits land exactly on the 130 ms floor, so the lab no longer uses these estimates "
        "for conclusions. They are still needed: they decide which saccade cells get the two-component model.", ()),
    "identifiability": ModelNote(
        "Tests that ask whether a t₀ estimate comes from the data or from the bounds placed on it.",
        "They are why hand t₀ is reported as measured and saccadic t₀ as fixed at 70 ms.", ()),
    "model_free": ModelNote("Plots of the raw reaction times. No model is fitted.", "The check every model result has to agree with.", ()),
    "model_free_e2": ModelNote("Two data checks specific to Experiment 2's richer files.", "Rule out data-handling explanations.", ()),
    "robustness": ModelNote("The speed-effect test battery on hand t₀ and a simulate-and-refit recovery study.",
                            "Quantify how strong the speed result is and how well t₀ can be recovered at this design.", ()),
    "later": ModelNote(
        "LATER treats a saccade as a decision signal rising at a rate that varies normally from trial to trial. On a reciprobit plot its "
        "prediction is a straight line. It has no separate non-decision time, so nothing can floor.",
        "Deprecated. LATER's rate and threshold live in a different parameter space from the Wald's drift, boundary and t₀, so it cannot "
        "be compared with the hand model — and comparing hand and eye is the point of the project. Its fit to saccades is about the "
        "same as the Wald's, so it offers no empirical advantage. The lab moved to one model family, the Bayesian shifted Wald, for "
        "both effectors.",
        ("Experiment 1: median reciprobit r² ≈ 0.97 — saccade latencies are close to reciprocal-normal.",
         "Raw goodness of fit is about equal to the Wald's for saccades (KS ≈ 0.12 for each)."),
        "Checking whether saccade latencies are reciprocal-normal, and spotting express-saccade participants (the second, early line)."),
    "two_boundary": ModelNote(
        "A two-boundary drift-diffusion model of reaction time and which of two responses was made — here, whether the hand led or "
        "lagged the target.",
        "Deprecated. When the data lack genuine two-boundary structure it inflates t₀, it fits the reaction-time distribution worse "
        "than the single-boundary Wald, and a drift-variability extension did not help. The lab keeps the single-boundary Wald and "
        "reports the two-boundary test as a robustness check.",
        ("Recovery: when the truth is single-boundary, the two-boundary model inflates t₀ by about 30 ms.",
         "Fit to the RT distribution: KS 0.148 against 0.071 for the Wald, worse in 39 of 40 cells.",
         "Hierarchical Bayesian comparison (PSIS-LOO): it lost by 12.6 standard errors.",
         "Experiment 2 cannot take it at all: only 1–9% of trials lag, so the second boundary has almost no data."),
        "Showing that the single-boundary choice was tested, not assumed."),
}

TABLE_NOTES = {
    "Bayesian_hrt_fits.csv": "Hand fits, Bayesian: v, a, t₀ (posterior means) and t₀'s 95% interval per participant × speed.",
    "Bayesian_hrt_ndt.csv": "Hand non-decision time per cell from the Bayesian fit, with a flag for cells at the floor.",
    "Bayesian_srt_fits.csv": "Saccade fits, Bayesian, per speed: single-component cells (v, a, t₀) and two-component cells (fast weight and component means).",
    "Bayesian_srt_ndt.csv": "Saccadic t₀ per participant (shared across speeds), its 95% interval and the participant's fastest saccade.",
    "Bayesian_srt_ndt_cells.csv": "Per-cell drift and boundary from the participant-level saccade model.",
    "DDM_hrt_fits.csv": "Hand fits, Method A, per participant × speed (v, a, t₀, KS).",
    "DDM_srt_fits.csv": "Saccade fits, Method A, with the single/two-component model choice per cell.",
    "SRT_fixedt0_fits.csv": "Saccade fits with t₀ fixed at 50, 70 and 90 ms.",
    "SRT_fixedt0_sensitivity.csv": "Mean drift by speed and mean KS for each fixed t₀.",
    "SRT_identifiability.csv": "Saccade floor sweep: fitted t₀ at each imposed floor and the slope, per cell.",
    "HRT_floor_control.csv": "Hand and saccade floor sweep: slope, eligibility and whether the fit is stuck at its fastest RT.",
    "LATER_fits.csv": "LATER per cell: mean and SD of the reciprocal rate, median latency, reciprobit r², KS, express fraction.",
    "dissociation_tests_P2.csv": "Speed-effect tests on t₀: Friedman, permutation, bootstrap 150 − 75 ms, per-participant slope.",
    "parameter_recovery_P2_summary.csv": "Recovery study summary: true vs fitted t₀, bias, RMSE and share at the floor.",
    "direction_check_summary.csv": "Right − left differences in RT, t₀, v, a and signed error, with Wilcoxon p.",
    "SRT_QA_flag_sensitivity.csv": "Saccade fits with and without the extraction's eye QA exclusions.",
}
