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
