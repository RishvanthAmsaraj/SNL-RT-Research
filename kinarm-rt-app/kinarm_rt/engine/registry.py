"""
registry.py -- what the app knows about: experiments, the script steps of each pipeline, and the analyses a
person can choose. Nothing here computes anything; every step is one of the repository's scripts, run as is.
"""
from __future__ import annotations
from dataclasses import dataclass, field

STATUS_ORDER = ("recommended", "supporting", "diagnostic", "deprecated")
STATUS_LABEL = {"recommended": "Recommended", "supporting": "Supporting", "diagnostic": "Diagnostic", "deprecated": "Deprecated"}


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    cohort: str
    speeds: tuple
    block: str
    data_file: str
    pipeline: str
    later_pipeline: str
    accent: str
    tagline: str
    raw_builder: str | None = None          # script that turns raw per-participant files into the pooled file


EXPERIMENTS = {
    "E1": Experiment("E1", "Experiment 1", "CMT", (0, 75, 150), "I", "pooled_data.csv", "experiment1", "later_e1",
                     "#1b7f79", "A stationary target and two moving speeds"),
    "E2": Experiment("E2", "Experiment 2", "CIR", (75, 100, 125, 150), "P2", "pooled_data_P2.csv", "experiment2", "later_e2",
                     "#d95f02", "Four moving speeds, no stationary target", raw_builder="build_pooled_data_P2.py"),
}

SPEED_COLOURS = {0: "#5a9e52", 75: "#d98c8c", 100: "#e0ad5c", 125: "#a88ccc", 150: "#809ed1"}   # the figures' palette


def _per_speed(stem: str, speeds) -> tuple:
    return tuple(f"{stem}_{s}_degs.{ext}" for s in speeds for ext in ("png", "pdf"))


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    script: str
    outputs: tuple
    needs: tuple = ()
    minutes: float = 0.5
    bayes: bool = False
    pipeline: str = "main"                  # "main" = the experiment's pipeline, "later" = its LATER pipeline
    columns: tuple = ()                     # extra input columns the script reads
    args: tuple = ()


def steps_for(exp: Experiment) -> dict[str, Step]:
    sp = exp.speeds
    s = [
        Step("ddm_fit", "Method A fits (hand and saccade)", "DDM_fit.py", ("DDM_hrt_fits.csv", "DDM_srt_fits.csv"), minutes=1.5),
        Step("ddm_figures", "Method A summary figure", "DDM_figures.py", ("DDM_summary.png", "DDM_summary.pdf"), ("ddm_fit",)),
        Step("ddm_conceptual", "Method A schematics", "DDM_conceptual.py", _per_speed("ddm_hrt", sp) + _per_speed("ddm_srt", sp), ("ddm_fit",)),
        Step("ndt_barchart", "Method A NDT chart", "NDT_barchart.py", ("NDT_barchart.png", "NDT_barchart.pdf"), ("ddm_fit",)),
        Step("vincentile", "Vincentile figures", "vincentile_figures.py",
             tuple(f"vincentile_results_fig{i}.pdf" for i in ("1_kde_overlay", "2_histograms", "3_vincentile_by_speed", "4_combined_vincentile"))),
        Step("why_floors", "Why saccadic t₀ floors", "why_saccadic_t0_floors.py", ("why_saccadic_t0_floors.png", "why_saccadic_t0_floors.pdf")),
        Step("srt_identifiability", "Saccade floor sweep", "SRT_identifiability_check.py",
             ("SRT_identifiability.png", "SRT_identifiability.pdf", "SRT_identifiability.csv"), ("ddm_fit",), minutes=2),
        Step("srt_fixed_t0", "Fixed-t₀ sensitivity", "SRT_fixed_t0_analysis.py",
             ("SRT_fixedt0_fits.csv", "SRT_fixedt0_sensitivity.png", "SRT_fixedt0_sensitivity.pdf"), ("ddm_fit",)),
        Step("hrt_floor_control", "Hand floor control", "HRT_floor_control.py",
             ("HRT_floor_control.csv", "HRT_floor_control.png", "HRT_floor_control.pdf"), ("ddm_fit", "srt_identifiability"), minutes=3),
        Step("bayes_hrt", "Bayesian hand fit", "Bayesian_HRT_fit.py", ("Bayesian_hrt_fits.csv", "Bayesian_hrt_ndt.csv"), minutes=6, bayes=True),
        Step("bayes_srt", "Bayesian saccade fits", "Bayesian_SRT_fit.py", ("Bayesian_srt_fits.csv",), ("ddm_fit",), minutes=15, bayes=True),
        Step("bayes_srt_ndt", "Bayesian saccade t₀ per participant", "Bayesian_SRT_ndt.py",
             ("Bayesian_srt_ndt.csv", "Bayesian_srt_ndt_cells.csv", "Bayesian_srt_ndt.png", "Bayesian_srt_ndt.pdf"), ("ddm_fit",), minutes=12, bayes=True),
        Step("bayes_figures", "Bayesian summary figure", "Bayesian_figures.py", ("Bayesian_summary.png", "Bayesian_summary.pdf"),
             ("ddm_fit", "bayes_hrt", "bayes_srt")),          # uses Bayesian_srt_ndt.csv for panel C when it exists
        Step("bayes_conceptual", "Bayesian schematics", "Bayesian_conceptual.py", _per_speed("bayes_hrt", sp) + _per_speed("bayes_srt", sp),
             ("bayes_hrt", "bayes_srt")),
        Step("ndt_bayesian", "Bayesian NDT chart", "NDT_barchart_bayesian.py", ("NDT_barchart_bayesian.png", "NDT_barchart_bayesian.pdf"),
             ("bayes_hrt", "bayes_srt_ndt")),
        Step("later", "LATER reciprobit fits", "LATER_analysis.py", ("LATER_fits.csv", "LATER_reciprobit.png", "LATER_reciprobit.pdf"),
             pipeline="later", columns=("SpeedCode",)),
    ]
    if exp.id == "E2":
        s += [
            Step("qa_sensitivity", "QA-flag sensitivity", "SRT_QA_flag_sensitivity.py", ("SRT_QA_flag_sensitivity.csv",), columns=("IncludeInEyeAnalysis",)),
            Step("direction", "Left vs right check", "direction_check.py", ("direction_check_cells.csv", "direction_check_summary.csv"),
                 columns=("Direction", "SignedError_deg", "SignedError_deg_HRT50"), minutes=1),
            Step("dissociation", "Speed-effect tests", "dissociation_tests.py", ("dissociation_tests_P2.csv",), ("ddm_fit", "bayes_hrt")),
            Step("recovery", "Parameter recovery (Method A)", "parameter_recovery_P2.py",
                 ("parameter_recovery_P2.csv", "parameter_recovery_P2_summary.csv"), ("bayes_hrt", "srt_fixed_t0"), minutes=3),
        ]
    return {x.id: x for x in s}


@dataclass(frozen=True)
class Analysis:
    id: str
    name: str
    status: str
    steps: tuple
    summary: str
    minutes: str
    experiments: tuple = ("E1", "E2")
    reference_only: bool = False


ANALYSES = [
    Analysis("bayesian", "Hierarchical Bayesian shifted Wald", "recommended",
             ("ddm_fit", "bayes_hrt", "bayes_srt", "bayes_srt_ndt", "bayes_figures", "bayes_conceptual", "ndt_bayesian"),
             "The lab's model. Hand and saccade reaction times are fitted with partial pooling across participants, so a "
             "noisy cell cannot drag hand non-decision time onto the floor. Produces the NDT chart, the summary figure "
             "and the per-speed schematics.", "15–40 min"),
    Analysis("identifiability", "Identifiability checks", "diagnostic",
             ("ddm_fit", "why_floors", "srt_identifiability", "hrt_floor_control", "srt_fixed_t0"),
             "Which non-decision times can the data actually pin down? The distribution-shape argument, the floor sweep "
             "for hand and saccade, and the fixed-t₀ sensitivity.", "3–6 min"),
    Analysis("model_free", "Model-free views", "supporting", ("vincentile",),
             "Raw reaction-time distributions and the eye-to-hand lag, sorted from fastest to slowest trials. No model involved.",
             "under a minute"),
    Analysis("model_free_e2", "Data checks", "supporting", ("qa_sensitivity", "direction"),
             "Whether the extraction's quality flags change anything, and how left and right targets differ in timing and aim.",
             "1–2 min", experiments=("E2",)),
    Analysis("robustness", "Speed-effect tests and recovery", "diagnostic", ("ddm_fit", "bayes_hrt", "srt_fixed_t0", "dissociation", "recovery"),
             "The Friedman, bootstrap, permutation and trend tests on hand t₀, and a simulate-and-refit recovery study.",
             "8–12 min", experiments=("E2",)),
    Analysis("method_a", "Method A — per-cell maximum likelihood", "supporting",
             ("ddm_fit", "ddm_figures", "ddm_conceptual", "ndt_barchart"),
             "Each participant × speed fitted on its own. Fast, but many hand fits land on the 130 ms floor, which is why "
             "the lab reports the Bayesian fits. Its saccade fits also decide which cells get a two-component model.", "1–2 min"),
    Analysis("later", "LATER model (saccades)", "deprecated", ("later",),
             "Saccade latencies as a rise to threshold with a normally distributed rate. Kept so its output can be inspected.",
             "under a minute"),
    Analysis("two_boundary", "Two-boundary DDM (hand)", "deprecated", (),
             "Reaction time and whether the hand led or lagged the target, modelled jointly. The committed Experiment 1 "
             "results are shown; it cannot be fitted to Experiment 2 (almost no lag trials).", "results only",
             experiments=("E1",), reference_only=True),
]


def analyses_for(exp_id: str) -> list[Analysis]:
    return [a for a in ANALYSES if exp_id in a.experiments]


def plan(exp: Experiment, analysis_ids) -> list[Step]:
    """All steps the chosen analyses need, dependencies first, each once."""
    steps = steps_for(exp); order, seen = [], set()
    def visit(sid):
        if sid in seen or sid not in steps: return
        for d in steps[sid].needs: visit(d)
        seen.add(sid); order.append(steps[sid])
    for a in analyses_for(exp.id):
        if a.id in analysis_ids:
            for sid in a.steps: visit(sid)
    return order
