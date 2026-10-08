# KINARM RT — reaction-time models, point and click

A desktop app for the SNL-RT-Research reaction-time analyses. Pick an experiment, pick the analyses, load your data,
press run, and read the results with an explanation next to every figure and table. No code editor, no terminal.

## The one design rule: the app runs the repository's scripts, unchanged

Version 1 re-implemented the pipeline inside the app. It was validated against the scripts (r = 0.999), but it was not
the same code, and its figures were simplified copies. Version 2 does not re-implement anything. It ships the
repository's own scripts in `pipelines/` and runs each one exactly as the lab does by hand — `python <script>` in a
folder holding the data — then shows the tables and figures the scripts write.

- `tools/sync_pipelines.py` copies the scripts from the repository and records a SHA-256 for each in
  `pipelines/MANIFEST.json`; `--check` fails if any byte differs. The app's footer reports the same check.
- The tests run the vendored scripts through the app's runner and require the committed tables to come back
  **byte-identical** (Experiment 2 Method A fits and fixed-t₀ fits, Experiment 1 LATER fits).
- The figures in the app are therefore the publication figures, not look-alikes.

## What you can do

**Switch experiments.** Experiment 1 (CMT cohort; 0, 75, 150 deg/s — includes a stationary target) and
Experiment 2 (CIR cohort; 75, 100, 125, 150 deg/s). Each has its own copy of the code. A third view, *Compare*,
puts the two side by side.

**Choose analyses** — each labelled with its status and an explanation:

| Analysis | Status | Scripts |
|---|---|---|
| Hierarchical Bayesian shifted Wald | Recommended — the lab's model | `Bayesian_HRT_fit`, `Bayesian_SRT_fit`, `Bayesian_SRT_ndt`, `Bayesian_figures`, `Bayesian_conceptual`, `NDT_barchart_bayesian` |
| Identifiability checks | Diagnostic | `why_saccadic_t0_floors`, `SRT_identifiability_check`, `HRT_floor_control`, `SRT_fixed_t0_analysis` |
| Model-free views | Supporting | `vincentile_figures` |
| Data checks (Experiment 2) | Supporting | `SRT_QA_flag_sensitivity`, `direction_check` |
| Speed-effect tests and recovery (Experiment 2) | Diagnostic | `dissociation_tests`, `parameter_recovery_P2` |
| Method A — per-cell maximum likelihood | Supporting | `DDM_fit`, `DDM_figures`, `DDM_conceptual`, `NDT_barchart` |
| LATER model (saccades) | **Deprecated** — runnable, with the reason it was set aside | `LATER_analysis` (Experiment 1 original; Experiment 2 configuration copy) |
| Two-boundary DDM (hand) | **Deprecated** — committed Experiment 1 results, with the evidence against it | results only |

Dependencies are resolved automatically (the Bayesian saccade fits need Method A's mixture selection, for example).

**Everything the repository holds is here.** Every figure file, table and document in the repository is in the app — the current results, the deprecated models, the earlier pipeline versions and the working iterations — and a test fails if anything is missing. Each figure is one card offering both of its files (PNG and PDF).

**Read the documents.** Reports, guides and records render in the app, PDFs with a preview, all downloadable.

**Browse figures.** A gallery of every figure, filtered by group (Main, Diagnostic, Method A, Deprecated) or by search. Opening a figure shows what it is, why it is there, how to read it, the numbers behind it in the results being shown, what it means for the project and what to keep in mind, with PNG and PDF downloads and Previous / Next to step through.

**Put figures side by side.** Any two figures next to each other — the same figure for both experiments, a figure from your run against the lab's, or two different figures — with one-click pairs for the main comparisons.

**Read tables.** Every CSV the scripts wrote, with a one-line description, its size, and a download.

**Link to anything.** The address bar follows the page, so `?exp=E2&view=Figures&fig=Bayesian_srt_ndt` opens that figure directly.

**See the lab's results instantly.** The committed results for both experiments ship in `reference_results/`, so
everything is browsable before anything runs. Your own runs appear alongside, under *Your run*.

## Running analyses

Load a pooled trial file in the pipeline's format (`pooled_data.csv` for Experiment 1, `pooled_data_P2.csv` for
Experiment 2). For Experiment 2 you can instead load all the per-participant `CIR…_TRIAL_Summary` files at once; the
app builds the pooled file with the repository's `build_pooled_data_P2.py`. The data are checked first (required
columns, block type, speeds, trials per cell, which optional columns are missing and which steps that skips).

Runs happen in the background. You can keep browsing, refresh the page, or cancel; the progress list shows each step,
its time, and its log if it fails. Each experiment + dataset gets its own folder (shown in the footer), and a step that
already ran on the same data with the same script is reused, so a second run is instant and an interrupted run
continues where it stopped.

Bayesian fits need PyMC **and a C++ compiler** (without one PyMC silently runs about nine times slower and does not land
on the same estimates). The packaged desktop app ships both; the app checks before it starts a Bayesian step.

## Install and start

- **Desktop app** (Windows, macOS): built by `.github/workflows/build-desktop.yml`; see `desktop/README.md`.
- **Conda**: `conda env create -f environment.yml && conda activate kinarm-rt && streamlit run app.py`
- **Docker**: `docker build -t kinarm-rt . && docker run -p 8501:8501 kinarm-rt`

## When the pipelines change

Edit the scripts in the repository as usual, then from this folder run `python tools/sync_pipelines.py` (copies the
scripts and the committed results, rewrites the manifest) and `python -m pytest -q`.

To check the interface in a real browser, start the app and run `python desktop/ui_check.py http://localhost:8501 shots/` (needs `pip install playwright`). It clicks through every section and a sample run in light and dark mode,
fails on any error, and saves a screenshot of each step.

## Layout

```
app.py                     interface
kinarm_rt/engine/          registry (experiments, steps, analyses), runner, data intake, results reader
kinarm_rt/explain.py       the text shown beside every figure, table and model
kinarm_rt/theme.py         the glass theme and the hero orbit (targets moving at their real speeds)
pipelines/                 the repository's scripts, vendored unchanged, + MANIFEST.json
reference_results/         the committed tables and figures for both experiments, LATER, two-boundary, comparison
tools/sync_pipelines.py    vendoring and the --check guard
tests/                     sync, registry, engine (byte-identical reproduction) and interface tests
desktop/                   the double-clickable app (launcher, stub, smoke test)
```

See `V2_PLAN.md` for the design decisions and what comes next.
