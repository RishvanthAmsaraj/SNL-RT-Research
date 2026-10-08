# Changelog — SNL RT Research Pipeline

All notable changes are documented here, mapped to the corresponding repository folder and the
deprecated version they belong to. The format follows [Keep a Changelog](https://keepachangelog.com/).

> **Version-to-folder mapping:**
> - `[0.0.1]–[0.0.x]` → [`Deprecated Pipelines/Deprecated Ver 1`](/Deprecated%20Pipelines/Deprecated%20Ver%201/) — PyDDM prototypes
> - `[0.1.0]` → [`Deprecated Pipelines/Deprecated Ver 2`](/Deprecated%20Pipelines/Deprecated%20Ver%202/) — Native MLE pipeline
> - `[0.2.0]` → [`Deprecated Pipelines/Deprecated Ver 2.5`](/Deprecated%20Pipelines/Deprecated%20Ver%202.5/) — Early Bayesian refinement
> - `[1.0.0]` → [`Deprecated Pipelines/Deprecated Ver 3`](/Deprecated%20Pipelines/Deprecated%20Ver%203/) — Hierarchical Bayesian + dissociation
> - `[2.0.0]` → [`Current Pipeline`](/Current%20Pipeline/) — Literature-anchored bounds, diagnostics

For the full narrative behind each decision, detailed diagnostic findings, and principles that guided
the entire project, see [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md).

---

## [2.5.5] — 2026-10-08 — Pre-computed insights and Windows-safe line endings

- `kinarm-rt-app` 2.3.2: figure explanations and comparison notes read their specific numbers
  from `pipelines/insights_summary.json`, pre-computed by `tools/precompute_insights.py` from
  the per-participant tables offline (aggregate only, no participant rows). A `.gitattributes`
  pins LF line endings and the repository was renormalized, so the Windows desktop build's
  hash checks match the manifest.

## [2.5.4] — 2026-10-08 — Participant data removed from the repository

- Removed all participant data (de-identified but unpublished human-subject data): raw trial tables, per-participant
  fit and NDT tables, and their notebooks and logs. Only aggregate results, figures, documentation and code remain.
  `.gitignore` rules keep pooled data and per-participant tables out for good.
- `kinarm-rt-app` 2.3.1: the app ships figures, documents and aggregate tables only. `sync_pipelines.py` skips any CSV
  carrying participant identifiers, the MANIFEST reflects the shipped set, and the build's test suite passes on the
  participant-data-free repository.

## [2.5.3] — 2026-10-08 — App 2.3: clearer figure groups, fuller explanations, comparison fixes

- `kinarm-rt-app` 2.3.0: figure groups Main / Diagnostic / Method A / Deprecated; Deprecated holds only genuinely older figures (25 earlier copies identical to current figures recorded instead of shown); fuller per-figure explanations with the numbers behind each figure; comparison notes in side by side; side-by-side figure size fixed on Streamlit 1.50; two-ring Compare hero; button rows. See `kinarm-rt-app/CHANGELOG.md`.

## [2.5.2] — 2026-10-08 — App 2.2: everything in the repository is in the app

- `kinarm-rt-app` 2.2.0: every figure file, table and document in the repository is in the app (checked by content), including the earlier pipeline versions and working iterations; a Documents section; gallery thumbnails; calmer transitions; run hardening and edge-case tests. See `kinarm-rt-app/CHANGELOG.md`.
- Desktop build: the Intel Mac build moved from the retired `macos-13` runner to `macos-15-intel`; the Mac bundle version comes from the package; release notes name the right download per platform.

## [2.5.1] — 2026-10-08 — App 2.1

- `kinarm-rt-app` 2.1.0: navigation crash fixed, every figure in the gallery, side-by-side view, deep links, error boundaries, WebKit-safe hero, panel spacing; checked in a real browser on Python 3.9 / Streamlit 1.50 and Python 3.12 / Streamlit 1.65. See `kinarm-rt-app/CHANGELOG.md`.
- `Current Pipeline/Code`: added the Experiment 1 tables the scripts write but were never committed — `SRT Analysis/SRT_identifiability.csv`, `SRT Analysis/SRT_fixedt0_sensitivity.csv`, `Supplementary/HRT_floor_control.csv`.

## [2.5.0] — 2026-10-07 — App 2.0: runs the repository's scripts; experiments, model choice, deprecated models

- `kinarm-rt-app` 2.0.0: vendors and runs the pipelines unchanged (byte-identical, checked), switches between Experiment 1 and 2, lets users choose analyses including the deprecated LATER and two-boundary models (with the reasons), and has a new glass interface. See `kinarm-rt-app/CHANGELOG.md` and `kinarm-rt-app/V2_PLAN.md`.
- `Deprecated Pipelines/LATER/`: the LATER script and outputs for Experiment 1, and an Experiment 2 configuration copy with its outputs.
- `Deprecated Pipelines/Two-boundary DDM/`: the two-boundary code and committed Experiment 1 results.
- The desktop build workflow packages the app's new `pipelines/`, `reference_results/` and `tools/` folders.

## [2.4.0] — 2026-10-07 — Corrected Paradigm 1 figures + cross-paradigm NDT comparison data

### Changed

- **Corrected every Paradigm 1 figure** (`Current Pipeline/Figures/`), redrawn from the committed Paradigm 1 tables using the Paradigm 2 figure scripts set to Paradigm 1 settings. No model was refitted, so no number changed. Fixes: the retired 100 ms hand floor in `DDM_summary` / `Bayesian_summary`; the "ESTIMATED, not fixed" title and legend overlap in `Bayesian_srt_ndt`; the legend covering data in `SRT_fixedt0_sensitivity`; typed-in text replaced by computed values; the fastest-saccade cell now shown as ceiling-bound in `SRT_identifiability`; a tick-label collision in `ddm_srt_0_degs`. Added `Figures/Supplementary/HRT_floor_control` (hand vs saccade floor test) to Paradigm 1, mirroring Paradigm 2. See `Current Pipeline/Documents/Figure_Corrections.md`.
- **Replaced the Paradigm 1 figure scripts** (`Current Pipeline/Code/`) with the corrected versions, adding `Code/Supplementary/HRT_floor_control.py`.

### Added

- **Cross-paradigm NDT comparison tables** (`Paradigm 2 Pipeline/Code/Comparison/`): per-participant hand NDT (`Experiment1_hand_NDT_Bayesian.csv`, `Experiment2_hand_NDT_Bayesian.csv`, `Hand_NDT_Exp1_vs_Exp2_side_by_side.csv`) and saccade NDT (`Saccade_NDT_Exp1_vs_Exp2_side_by_side.csv`, `Saccade_NDT_participant_level_Exp1_vs_Exp2.csv`).

## [2.3.2] — 2026-10-06 — Pre-publication verification of the Paradigm 2 figures and analysis

### Added

- `Paradigm 2 Pipeline/Documents/FIGURE_GUIDE.md` — every figure: what it shows, how to read it, its numbers, where it fits, caveats, suggested captions.
- `Code/Validation/verify_math.py`, `verify_claims.py` (72 quoted numbers re-derived and matched), `bayesian_recovery_P2.py` (known 10 ms hand t₀ change, production Bayesian model).

### Changed

- Interpretation, from the Bayesian recovery test: the model detects a 10 ms hand t₀ change at Paradigm 2's design but shrinks it (−5.9 ms) and moves part into decision time; absolute hand t₀ reads ~6 ms high; per-cell 95% intervals under-cover (70%); the t₀/decision-time anticorrelation (r ≈ −0.75) arises from noise alone. Documents now say so.
- `P1_vs_P2_saccade`: panel C uses the Bayesian fits; panel D reports the share of fits that follow the floor (the median misled — the eye slopes form two clusters). `HRT_floor_control` legend likewise.
- Two values corrected to the exact rounding used everywhere else (Paradigm 2 fast-end HRT at 100 deg/s 205.6 ms; its 75 → 150 change +0.8 ms).

## [2.3.1] — 2026-10-03 — Paradigm 2 verification: speed-effect finding reframed, cross-paradigm comparison added

### Changed

- **Key finding restated.** [2.3.0] said the Paradigm 1 speed effect "does not replicate". Within Paradigm 1, the 75 → 150 deg/s step in hand t₀ (−10.1 ms) is itself not reliable (p = 0.066) and is not visible in the raw RTs (median HRT +3.2 ms), while the stationary → moving step is (t₀ −11.5 ms, p = 0.005; median HRT −10.2 ms, p < 0.001). Between moving speeds hand t₀ is flat in both paradigms, and in both the small t₀ shift is cancelled by decision time (r = −0.80). Paradigm 1's effect is therefore stationary-vs-moving, which Paradigm 2 has no condition to test.
- `P2_Results_and_P1_Comparison.md` revised (participant IDs, trial counts, parity wording, saccadic-ceiling wording); `P2_Technical_Breakdown.md` regenerated (the claim that Paradigm 1 saccades never reached their fastest-saccade bound was wrong — 4/32 cells do; the floor-sweep comparison now uses the same code on both paradigms).
- `run_all_P2.py --skip-bayes-fits` now carries the Bayesian run logs, so the regenerated breakdown keeps its convergence numbers.

### Added

- `Paradigm 2 Pipeline/Code/Comparison/P1_vs_P2_comparison.py` → `Figures/Comparison/` (hand, saccade/identifiability, distributions) and `P1_vs_P2_summary.csv`; Paradigm 1 floor sweep rerun with the Paradigm 2 code.
- `Paradigm 2 Pipeline/Documents/VERIFICATION_REPORT_P2.md` — what was checked, what agreed, what was corrected.

## [2.3.0] — 2026-09-26 — Paradigm 2 (CIR): same pipeline, new cohort, speed effect does not replicate
**New folder:** [`Paradigm 2 Pipeline/`](Paradigm%202%20Pipeline/)

### Added

- **`Paradigm 2 Pipeline/`** — the single-boundary shifted-Wald pipeline run on Paradigm 2: 16 CIR participants, 4 speeds (75/100/125/150 deg/s), 10,158 hand + 10,017 saccade trials. Configuration-only port (speeds, `BlockType == "P2"`, input `pooled_data_P2.csv`); likelihood, priors, bounds, floors (hand 130 / saccade 70 ms), RT windows, and sampler settings are byte-identical to Paradigm 1.
- **Five supplementary analyses** — `dissociation_tests.py`, `HRT_floor_control.py`, `SRT_QA_flag_sensitivity.py`, `direction_check.py`, `parameter_recovery_P2.py` — plus **`P1_parity_check.py`**, which validates that this environment reproduces the published Paradigm 1 fits exactly (Method A, 48/48 cells) and to sampling error (Method B, r = 0.9995).
- **`Documents/P2_Results_and_P1_Comparison.md`** — synthesis of results, takeaways, open questions, and the Paradigm 1 comparison. `Documents/P2_Technical_Breakdown.md` carries the full computed tables.

### Key finding

**The Paradigm 1 speed effect does not replicate.** Paradigm 1's hand t₀ decreased with speed (158 → 148 ms across 75–150); Paradigm 2's is flat (≈147–152 ms, Friedman p = 0.026 but +3.9 ms in the opposite direction). Hand t₀ remains identified (0/64 Bayesian cells floored), saccadic t₀ remains reported as fixed at 70 ms (now also ceiling-bound at the participant level — 6/15 vs 0/14 in Paradigm 1).

## [2.2.0] — 2026-09-17 — Single-boundary consolidation: archive LATER + two-boundary

### Decision

The **single-boundary shifted Wald** is now the sole production model. The two alternatives explored as professor-directed next steps were archived:

- **LATER** (reciprobit) — a tie against the Wald on hands, and its rate/threshold parameters are not comparable to the Wald's drift/boundary/non-decision, so it offered no compelling benefit. Moved `Current Pipeline/Code/LATER Model/` and `Current Pipeline/Figures/LATER Model/` to `Working Iterations/LATER Model/`.
- **Two-boundary DDM** — the gate check found no usable scored direction. Moved to `Working Iterations/Two Boundary Model/`.

LATER was removed from the active run order (`CODE_REFERENCE.md`), the figure reference (`RUN_GUIDE.md`), and the repository structure (`README.md`).

## [2.1.0] — 2026-07-18 — kinarm-rt-app: Streamlit GUI + headless CLI pipeline
**New folder:** [`kinarm-rt-app/`](kinarm-rt-app/)

### Added

- **`kinarm-rt-app/`** — A point-and-click Streamlit app and headless CLI (`run_pipeline.py`) that reproduces the full SNL RT pipeline. Fits the same models (shifted-Wald Bayesian, MLE contamination, express/regular mixtures, LATER reciprobit) with a GUI, config-driven CLI, and Docker container.
- **Dockerfile** (`kinarm-rt-app/Dockerfile`) — fully reproducible environment on conda-forge PyMC, eliminating the conda/pip split documented in earlier issues.
- **Cross-validation** — PSIS-LOO comparison (estimated vs fixed t₀) using `arviz.compare`, addressing limitation #5.
- **Dissociation test battery** — participant-resampling bootstrap + within-participant permutation test supplementing the Friedman, implemented in `kinarm_rt/stats_tests.py`.
- **Parameter-recovery study** — simulates from known parameters and refits, demonstrating hand t₀ is recovered while saccadic t₀ is not.
- **Sensitivity sweeps** — dip-test mixture-threshold sweep and fixed-t₀ sensitivity (`Advanced analyses` tab).
- **Frequentist Method A fit** with contamination — available alongside the Bayesian in the `Model comparison` tab.
- **`run_pipeline.py`** — headless CLI for batch/cluster use, configured via `config.example.yaml`.
- **Repo-format CSV export** — Writes `Bayesian_hrt_fits.csv` / `Bayesian_srt_fits.csv` compatible with the existing pipeline's downstream scripts.
- **Graceful degradation** — if PyMC is missing, only the Bayesian fit is disabled; preview, LATER, figures, and export still work.

### Changed

- Updated top-level `README.md` with `kinarm-rt-app/` section and repository layout.
- Updated `Current Pipeline/ISSUES_AND_IMPROVEMENTS.md` — Docker, LOO-CV, bootstrap, and sensitivity items marked resolved.
- Updated `.gitignore` to cover app-generated outputs (`*.html`, `output/`, `*.zip`).

### Files added

```
kinarm-rt-app/
├── app.py
├── run_pipeline.py
├── Dockerfile
├── environment.yml
├── requirements.txt
├── config.example.yaml
├── run_app.sh / run_app.bat
├── kinarm_rt/
│   ├── __init__.py, _speeds.py
│   ├── data.py, filters.py
│   ├── models/wald.py, later.py
│   ├── analysis.py, compare.py
│   ├── figures.py, diagnostics.py
│   ├── exports.py, report.py
│   └── frequentist.py, stats_tests.py
├── sample_data/example_pooled_data.csv
├── tests/test_smoke.py, test_features.py
├── README.md
└── RESEARCH_AND_ROADMAP.md
```

---

## [2.0.0] — 2026-06-24 — Literature-anchored bounds, flooring diagnosis, LATER alternative
**Repo folder:** [`Current Pipeline/`](/Current%20Pipeline/)

### Changed

- **Drift cap `V_MAX` 40 → 20**, anchored to the Tran et al. (2020) systematic-review envelope at
  s = 1 (`|v| ≲ 18.5`). Never bound at 40 (fitted v ≈ 4.7–13.8); no change to results.
- **Hand t₀ floor 100 → 130 ms**, anchored to Haith et al. (2016) reach-preparation minimum. Barely
  binds (fitted min 129 ms); results survive (HRT t₀ 170 → 158 → 148 ms, p = 0.003).
- **SRT per-participant non-decision floor 35 → 70 ms** (harmonized with the per-cell fit and the
  saccadic dead-time literature). With the floor enforced, the per-participant model collapses to 70
  ms for all participants — confirming saccadic t₀ is **not identifiable above the physiological
  floor**. Saccadic t₀ is now **reported as fixed at 70 ms** rather than estimated per participant.
- **Re-attributed `v`/`a` bounds** from the informal Ratcliff & Tuerlinckx range to the Tran et al.
  (2020) systematic envelopes. Ratcliff & Tuerlinckx (2002) retained for the contamination model only.
- **NDT bar charts:** zoomed y-axes to populated range; switched t₀-by-speed panels from bars to
  mean-markers-with-dots (truncated bars exaggerate differences; point-with-CI does not).

### Added

- `why_saccadic_t0_floors.py` — diagnostic figure showing the **skew/spread mechanism**: a Wald ties
  implied t₀ to `mean − 3·SD/skewness`; near-symmetric saccadic distributions (skew/CV ≈ 3.4) force
  t₀ below the floor while right-skewed hand distributions (skew/CV ≈ 12.9) do not.
- `LATER_analysis.py` — the saccade-native LATER model (Carpenter & Williams, 1995). Saccadic
  latencies fall on the predicted reciprobit line (median r² = 0.98). Included as a **complementary**
  saccade analysis (its rate/threshold parameters do not map to the Wald's).
- Validation of implementation against field tools: hierarchical-Bayesian-PyMC architecture matches
  HDDM / HSSM; single-boundary shifted Wald vs the two-choice DDM (appropriate for go-type task);
  PyDDM (Shinn et al. 2020) retained for historical comparison.

### Fixed

- **Knox & Wolohan DOI** corrected in all docstrings: `e0133595` (unrelated HIV-vaccine paper) →
  `e0120437`.
- DDM NDT chart "physiological min" line moved 100 → 130 ms; Bayesian NDT chart floor-line label
  corrected from "100 ms" to 130 ms.

### Files affected

- `Current Pipeline/DDM/DDM_fit.py` — V_MAX, floor values, DOI, citation re-attribution
- `Current Pipeline/Bayesian/Bayesian_HRT_fit.py` — V_MAX, hand t₀ floor, citation updates
- `Current Pipeline/Bayesian/Bayesian_SRT_fit.py` — V_MAX, SRT floors, citation updates
- `Current Pipeline/Bayesian/Bayesian_SRT_ndt.py` — participant-level SRT floor 35→70, collapse reporting
- `Current Pipeline/Bayesian/why_saccadic_t0_floors.py` — new diagnostic
- `Current Pipeline/Bayesian/LATER_analysis.py` — new complementary analysis
- `Current Pipeline/DDM/DDM_figures.py` — NDT chart floor lines
- `Current Pipeline/NDT/NDT_barchart.py` — y-axis zoom, bar→point switch
- `Current Pipeline/NDT/NDT_barchart_bayesian.py` — y-axis zoom, bar→point switch, floor label fix

---

## [1.0.0] — 2026-06-23 — Hierarchical Bayesian pipeline and the dissociation result
**Repo folder:** [`Deprecated Pipelines/Deprecated Ver 3/`](/Deprecated%20Pipelines/Deprecated%20Ver%203/)

### Added

- **Hierarchical Bayesian estimation** (PyMC / NUTS, partial pooling) as Method B — the reported
  results: `Bayesian_HRT_fit.py`, `Bayesian_SRT_fit.py`, `Bayesian_SRT_ndt.py`. Architecture follows
  Wiecki et al. (2013) / Vandekerckhove et al. (2011); convergence via R-hat (Gelman & Rubin, 1992).
- **Ordered pipeline** with defined run order: fits → figures → diagnostics (figure/diagnostic scripts
  consume the CSV fit tables).
- **Three-category figure suite:** Bayesian (results), DDM (comparison/diagnostic — exposes flooring),
  vincentile (model-free raw RTs), with paired DDM/Bayesian versions.
- `SRT_identifiability_check.py` and `SRT_fixed_t0_analysis.py` — diagnostics establishing that
  saccadic t₀ is not identifiable and that drift-by-speed is robust to the fixed t₀ value.
### Changed

- **Headline HRT result now from Bayesian fits, not DDM.** The DDM hand speed effect (p = 0.047)
  rested entirely on three floored cells (`CMT001`, `CMT002`, `CMT010` at 150 deg/s); dropping them
  gave p = 0.199. The Bayesian model floors 0 cells and **strengthens** the effect to p = 0.0016.
- NDT bar chart HRT panel switched from DDM t₀ / p = 0.047 to Bayesian t₀ / p = 0.0016; SRT panel
  uses a per-participant forest plot rather than by-speed bars.

### Identified

- The non-decision-time floor-piling is a **per-cell identifiability** phenomenon, not a
  DDM-vs-Bayesian one: even the Bayesian per-cell saccadic fit floors 19/33 cells; only pooling t₀
  to the participant level removes it.

### Files (Ver 3, migrated to Current Pipeline later)

- `Bayesian/Bayesian_HRT_fit.py`, `Bayesian_SRT_fit.py`, `Bayesian_SRT_ndt.py`
- `Bayesian/Bayesian_figures.py`, `Bayesian_conceptual.py`
- `Bayesian/SRT_identifiability_check.py`, `SRT_fixed_t0_analysis.py`
- `DDM/DDM_fit.py`, `DDM_figures.py`, `DDM_conceptual.py`
- `NDT/NDT_barchart.py`, `NDT_barchart_bayesian.py`
- `Vincentile/vincentile_figures.py`
- `RUN_GUIDE.md` — installation and run order

---

## [0.2.0] — 2025-08 — Express saccades + first Bayesian models
**Repo folder:** [`Deprecated Pipelines/Deprecated Ver 2.5/`](/Deprecated%20Pipelines/Deprecated%20Ver%202.5/)

### Added

- **First Bayesian models** (per-cell, participant-level) as early exploration of the floor-piling
  problem. These preceded the full hierarchical build-out and lacked participant-level t₀ pooling.
- **Express/regular saccadic mixture detection** using a fit-driven + structural validation approach
  (single Wald fit attempted first; mixture adopted only when single fails with KS > 0.10, components
  are substantial `0.10 ≤ π ≤ 0.90`, and modes are separated `≥ 30 ms`).
- `SRT_ndt` analysis — early per-participant saccadic non-decision model (precursor to the Phase 1
  version, with a lower 35 ms floor that produced overscattered estimates).

### Changed

- **Bimodal detection replaced:** BIC (over-detection) and dip-test (under-detection) replaced by
  the fit-driven + structural approach described above.
- Portability improvements: hard-coded absolute Windows paths replaced with relative paths using
  `SCRIPT_DIR`.

### Identified

- SRT t₀ floor-piling as the central technical problem requiring the Bayesian solution.
- The need for a participant-level hierarchical approach (per-cell pooling insufficient).

### Key problems that drove migration to Ver 3

| Problem | Impact | Ver 3 Fix |
|---|---|---|
| Per-cell SRT t₀ floors 19/33 cells even with Bayesian | ~50% of estimates are bounds, not measurements | Participant-level t₀ pooling |
| No full credible intervals | Cannot distinguish well-identified from poorly identified cells | Full posterior CIs |
| No convergence diagnostics | Cannot detect model misfit | R-hat + divergence tracking |

### Files

- `Bayesian/Bayesian_HRT_fit.py` (early version) — first pass at hand Bayesian
- `Bayesian/Bayesian_SRT_fit.py` (early version) — per-cell saccadic Bayesian with mixture
- `Bayesian/Bayesian_SRT_ndt.py` (early version) — per-participant SRT NDT (35 ms floor)
- `DDM/DDM_fit.py` — native scipy MLE (mature at this point)
- `DDM/DDM_figures.py` — publication figures (mature)
- `Vincentile/vincentile_figures.py` — model-free figures (mature)
- `NDT/NDT_barchart.py`, `NDT_barchart_bayesian.py` — NDT visualization
- `RUN_GUIDE.md` — structured run order

---

## [0.1.0] — 2025-08 — Native MLE pipeline with real data
**Repo folder:** [`Deprecated Pipelines/Deprecated Ver 2/`](/Deprecated%20Pipelines/Deprecated%20Ver%202/)

### Added

- Initial drift-diffusion fitting on the **real KINARM dataset**: `pooled_data.csv` (7,676 trials,
  16 participants, three target speeds), with `DDM_fit.py` (frequentist MLE, `scipy.optimize`).
- First figure families: vincentile plots, RT histograms / KDE overlays, DDM conceptual schematics,
  NDT bar charts, and a 9-page diagnostic suite. House style: pale green/red/blue per speed, Arial
  font, 300 DPI PDFs with `pdf.fonttype=42` (editable text).
- `MIGRATION_NOTES.md` (now in `Deprecated Pipelines/Deprecated Ver 1/`) documenting the transition.
- Reproducible cached outputs (`.npz`) and a conda environment workaround for PyMC on Windows.

### Changed (from Ver 1 prototypes)

- **Model class:** two-choice DDM (synthetic prototype) → **single-boundary shifted Wald**. The
  interception task is go-type (no binary choice); the Wald is the correct first-passage density.
- **Saccadic RT filter:** ≥150 ms → **80–600 ms**. The 150 ms cutoff removed genuine fast (express)
  saccades; 80 ms is the human anticipation threshold.
- **Likelihood:** pure Wald MLE → **95% Wald + 5% uniform contamination mixture** (Ratcliff &
  Tuerlinckx, 2002). Down-weights outliers without excluding data.
- **Visualization:** basic matplotlib → vector PDFs, condition-specific color scheme, proper
  typography, conceptual schematics.
- **Validation:** none → KS goodness-of-fit statistics, diagnostic PDFs, mixture validation.
- **Portability:** PyDDM dependency → native scipy (no external DDM library).

### Identified

- `CMT0012` and `CMT002` (and express cases `CMT003`, `CMT004`) as **express-saccade-dominant**.
  Decision: model bimodality with mixtures; **never exclude participants.**
- The saccadic t₀ floor-piling artifact (per-cell t₀ estimates pile at the imposed floor).
- The first hierarchical-Bayesian approach as a potential fix (seed of Method B).

### Files

- `DDM Model/DDM_fit.py` — core fitting (MLE)
- `DDM Model/DDM_figures.py` — publication figures
- `DDM Model/DDM_conceptual.py` — process schematics
- `NDT Code/` — early NDT calculations
- `Vincentile Code/` — early vincentile methods
- `Verification Code/ddm_diagnostics.py` — comprehensive 9-page diagnostic
- `Bayesian Model/` — early per-cell Bayesian implementations
- `Deprecated/` — even older code preserved for reference

---

## [0.0.1]–[0.0.x] — 2025-08 — PyDDM prototypes
**Repo folder:** [`Deprecated Pipelines/Deprecated Ver 1/`](/Deprecated%20Pipelines/Deprecated%20Ver%201/)

### Added

- Proof-of-concept using the **PyDDM library** (Shinn et al. 2020).
- Synthetic data generators for parameter recovery validation.
- Single-choice and dual-choice task implementations (both PyDDM and native).
- Basic visualization with matplotlib.

### Identified problems (why Ver 1 was deprecated)

| Problem | Impact | Fixed In |
|---|---|---|
| **PyDDM dependency** | Limited customization; version conflicts; slower on large datasets | Ver 2 (native scipy) |
| **Synthetic data only** | No connection to real KINARM data; no empirical validation | Ver 2 (real data loading) |
| **Two-boundary DDM** | Designed for binary choice tasks, not interception | Ver 2 (single-boundary Wald) |
| **No statistical validation** | No KS, no parameter recovery, no goodness-of-fit | Ver 2 (diagnostic suite) |
| **Basic visualization** | Simple histograms; no publication quality | Ver 2 (vector PDFs, proper fonts) |
| **No hierarchical structure** | Each participant fitted in isolation | Ver 3 (partial pooling) |

### Files

- `DualChoice*.py` — Dual choice task models (PyDDM and native)
- `SingleChoice*.py` — Single choice task models (PyDDM and native)
- `*DataGen.py` — Synthetic data generators

---

## Migration Guide

### From Ver 1 → Ver 2
1. Replace PyDDM with native scipy optimization
2. Switch from two-boundary DDM to single-boundary shifted Wald
3. Load real data from `pooled_data.csv` with proper filters
4. Add KS goodness-of-fit and contamination mixture
5. Move from basic to publication-quality figures

### From Ver 2 → Ver 2.5
1. Replace BIC/dip-test mixture detection with fit-driven + structural validation
2. Introduce first Bayesian per-cell models
3. Replace hard-coded absolute paths with relative paths
4. Document express-saccade participants; build mixture models

### From Ver 2.5 → Ver 3
1. Replace per-cell t₀ with participant-level hierarchical estimation
2. Replace MLE with full Bayesian posterior (credible intervals)
3. Add convergence diagnostics (R-hat, divergences)
4. Build structured pipeline (fits → figures → diagnostics)
5. Create three-category figure system

### From Ver 3 → Current (v3.0 Final)
1. Anchor all bounds to systematic literature review (Tran 2020; Haith 2016)
2. Add LATER model as complementary saccade analysis
3. Diagnose *why* saccadic t₀ floors (skew/spread mechanism figure)
4. Tighten V_MAX (40→20); raise hand t₀ floor (100→130 ms)
5. Fix Knox & Wolohan DOI; re-attribution of v/a bounds
6. Report saccadic t₀ as fixed at 70 ms (not estimated per participant)
7. Refine NDT charts (zoomed axes, bar→point switch)
8. Validate implementation against HDDM/HSSM architecture

---

## Citation

If you use this pipeline, please cite the version appropriate to your analysis:

**Current Pipeline (Bayesian):**
```
Amsaraj, R. (2025). SNL RT Research Pipeline — Hierarchical Bayesian
Drift-Diffusion Models for KINARM Interception Tasks.
Sensorimotor Neuroscience Laboratory.
```

**Earlier versions:** See individual script headers for method-specific citations.

---

## References (Selected)

- Anders, R., Alario, F.-X., & Van Maanen, L. (2016). The shifted Wald distribution for response
  time data analysis. *Psychological Methods*, 21(3), 309–327.
- Carpenter, R. H. S., & Williams, M. L. L. (1995). Neural computation of log likelihood in control
  of saccadic eye movements. *Nature*, 377, 59–62.
- Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences.
  *Statistical Science*, 7(4), 457–472.
- Haith, A. M., Pakpoor, J., & Krakauer, J. W. (2016). Independence of movement preparation and
  movement initiation. *Journal of Neuroscience*, 36(10), 3007–3015.
- Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of the diffusion model. *Psychonomic
  Bulletin & Review*, 9(3), 438–481.
- Tran, N., van Maanen, L., Heathcote, A., & Matzke, D. (2020). Systematic parameter reviews in
  cognitive modeling. *Frontiers in Psychology*, 11, 608287.
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the
  drift-diffusion model in Python. *Frontiers in Neuroinformatics*, 7, 14.

See [`REFERENCES.bib`](REFERENCES.bib) and [`REFERENCES.md`](REFERENCES.md) for the complete bibliography.
