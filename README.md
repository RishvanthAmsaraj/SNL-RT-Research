# SNL RT Research

**KINARM Interception Reaction-Time Pipeline** — Hierarchical Bayesian Drift-Diffusion Analysis of Hand and Saccadic Reaction Times

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Active-red)](https://github.com/RishvanthAmsaraj/SNL-RT-Research)

---

## Overview

This repository contains the analysis pipeline for reaction-time research conducted at the **Sensorimotor Neuroscience Laboratory (SNL)**. The project applies a **single-boundary shifted-Wald drift-diffusion model** (hierarchical Bayesian and frequentist MLE) to hand and saccadic reaction times from a KINARM interception task — **16 participants × 3 target speeds** (0°, 75°, 150° deg/s).

### Headline Result

**Hand non-decision time (t₀) decreases with target speed, saccadic non-decision time does not.**  
HRT t₀: 170 → 158 → 148 ms (Friedman *p* = 0.003, 0/48 cells floored). Saccadic t₀ is fixed at the 70 ms physiological floor (not identifiable above it). See [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) for the full investigative arc that produced this dissociation.

---

## The Model

The interception task is a **go-type response** (no binary correct/incorrect choice), so the standard two-choice diffusion model does not apply. The correct descriptive model is the **single-boundary diffusion**, whose first-passage-time density is the **shifted Wald** (inverse Gaussian with a temporal shift).

| Parameter | Meaning | Justification |
|---|---|---|
| **Drift rate `v`** | Rate of evidence accumulation toward action threshold | Tran et al. (2020) envelope (`\|v\| ≲ 18.5` at s = 1) |
| **Boundary `a`** | Evidence required to initiate response | Tran et al. (2020); policy parameter, not physiological |
| **Non-decision time `t₀`** | Sensory + motor time outside the decision process | Hand: Haith et al. (2016); Saccade: fixed at 70 ms (Bompas et al. 2017; Ludwig et al. 2007) |

The likelihood is a **contamination mixture**: 95% shifted Wald + 5% uniform (Ratcliff & Tuerlinckx, 2002), which downweights outlier trials *without excluding any data*.

---

## Repository Layout

```
SNL-RT-Research/
├── README.md                         # This file
├── CHANGELOG.md                      # Version-by-version change log
├── DEVELOPMENT_HISTORY.md            # Full development narrative & decisions
├── REFERENCES.bib                    # LaTeX bibliography
├── REFERENCES.md                     # Human-readable reference list
├── LICENSE                           # MIT license
│
├── Current Pipeline/                 # Active production pipeline (v3.0)
│   ├── Bayesian/                     #   Hierarchical Bayesian models (Method B — the reported results)
│   ├── DDM/                          #   Frequentist MLE fits (Method A — comparison/diagnostic)
│   ├── NDT/                          #   Non-decision time analysis scripts
│   ├── Vincentile/                   #   Model-free RT distribution analysis
│   ├── figures/                      #   Generated publication figures
│   ├── RUN_GUIDE.md                  #   Step-by-step execution instructions
│   └── ISSUES_AND_IMPROVEMENTS.md    #   Known issues & roadmap
│
├── Deprecated Pipelines/             # Archived historical versions
│   ├── Deprecated Ver 1/             #   PyDDM prototypes
│   ├── Deprecated Ver 2/             #   Native scipy MLE pipeline
│   ├── Deprecated Ver 2.5/           #   Early Bayesian refinement
│   └── Deprecated Ver 3/             #   Pre-literature-anchored pipeline
│
└── pooled_data.csv                   # Canonical input (7,676 trials, 16 participants)
```

---

## Two Estimation Methods

| Method | Framework | Implementation | Status |
|---|---|---|---|
| **Method A — Frequentist MLE** | `scipy.optimize.differential_evolution` | `DDM/DDM_fit.py` | Comparison/diagnostic; exposes the floor-piling |
| **Method B — Hierarchical Bayesian** | PyMC / NUTS, partial pooling | `Bayesian/*.py` | **The reported results** — full credible intervals |

**The DDM exists only as a comparison tool** — it shows the non-decision-time floor-piling that the Bayesian model resolves. The Bayesian figures are the results; DDM figures are the diagnostic that validates the Bayesian improvement; vincentile figures are model-free raw data.

---

## Quick Start

### Prerequisites

**Method A (DDM)** — works with pip:
```bash
pip install numpy scipy pandas matplotlib scikit-learn diptest
```

**Method B (Bayesian)** — requires conda (PyMC needs a compiler):
```bash
conda create -n snl python=3.11
conda activate snl
conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib scikit-learn diptest
```

### Running the Pipeline

See [`Current Pipeline/RUN_GUIDE.md`](Current%20Pipeline/RUN_GUIDE.md) for detailed instructions.  
Basic workflow: **fits → figures → diagnostics** (each stage consumes CSV outputs from the previous).

```bash
cd "Current Pipeline"

# 1. Fits (DDM must run first — Bayesian SRT reads its single/mixture split)
python DDM/DDM_fit.py
python Bayesian/Bayesian_HRT_fit.py
python Bayesian/Bayesian_SRT_fit.py
python Bayesian/Bayesian_SRT_ndt.py

# 2. Figures (seven scripts producing all publication figures)
python DDM/DDM_figures.py
python Bayesian/Bayesian_figures.py
python NDT/NDT_barchart.py
python NDT/NDT_barchart_bayesian.py
python Vincentile/vincentile_figures.py
# ... see RUN_GUIDE.md for full list

# 3. Diagnostics
python Bayesian/SRT_identifiability_check.py
python Bayesian/SRT_fixed_t0_analysis.py
python Bayesian/why_saccadic_t0_floors.py
python Bayesian/LATER_analysis.py
```

---

## Data

`pooled_data.csv` is the exact concatenation of the 16 per-participant `CMT*_MASTER_Summary.csv` files — **7,676 trials**, participants `CMT001`–`CMT010`, `CMT0011`, `CMT0012`, `CMT0014`–`CMT0017` (the set skips `CMT0013`). Three target speeds (0 / 75 / 150 deg/s); hand and saccadic RT per trial.

**Data filters** (these are *data-cleaning cutoffs* on raw RTs, distinct from fitted-parameter floors):

| Stream | Lower Bound | Upper Bound | Justification |
|---|---|---|---|
| Hand RT | 150 ms | 800 ms | Anticipation / lapse removal (Whelan 2008; Luce 1986) |
| Saccadic RT | 80 ms | 600 ms | Human anticipation threshold (Fischer & Weber 1993; Knox & Wolohan 2015) |

**No participant data is dropped.** Express-saccade-dominant participants (`CMT0012`, `CMT002`, `CMT003`, `CMT004`) are handled with mixture models.

---

## Pipeline Version History

The repository traces the full evolution from initial prototypes to the current production pipeline:

| Version | Repo Folder | Key Method | Key Advancements |
|---|---|---|---|
| **Ver 1** | `Deprecated Pipelines/Deprecated Ver 1/` | PyDDM (two-choice) | Proof-of-concept on synthetic data |
| **Ver 2** | `Deprecated Pipelines/Deprecated Ver 2/` | Native scipy MLE | Real data loading; single-boundary Wald; contamination mixture |
| **Ver 2.5** | `Deprecated Pipelines/Deprecated Ver 2.5/` | Per-cell Bayesian (PyMC) | First Bayesian fix; express saccade handling; bimodal detection |
| **Ver 3** | `Deprecated Pipelines/Deprecated Ver 3/` | Hierarchical Bayesian | Participant-level SRT t₀; dissociation result; full credible intervals |
| **Current** | `Current Pipeline/` | Literature-anchored hierarchical Bayesian | Systematic-review bounds; flooring diagnosis; LATER model |

Detailed change logs, migration notes, and the rationale for each deprecation are available in:
- **[CHANGELOG.md](CHANGELOG.md)** — Version-by-version technical changelog
- **[DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md)** — Full narrative of every major decision
- **[Per-pipeline issues documents](#)** (in each deprecated pipeline folder)

---

## Key Methodological Decisions

These decisions were deliberately made and held throughout the project's evolution:

1. **Single-boundary shifted Wald (not two-choice DDM).** The interception task is go-type; a two-choice model is the wrong object.
2. **Contamination mixture (not excluding outliers).** Down-weights outliers mathematically; no data is excluded.
3. **Same Bayesian model for all participants.** No participant is "handed back" to the DDM — the ones you'd be tempted are the ones that floor hardest.
4. **Express-dominant participants kept (not excluded).** Bimodality handled with mixtures.
5. **No across-trial variability parameters (sv, sz, st₀).** Conservative choice given trial counts.
6. **Saccadic t₀ reported as fixed at 70 ms (not estimated).** Non-identifiability demonstrated; LATER provided as complement.

---

## Citation

If you use this pipeline in your research, please cite:

```
Amsaraj, R. (2025). SNL RT Research Pipeline — Hierarchical Bayesian
Drift-Diffusion Models for KINARM Interception Tasks.
Sensorimotor Neuroscience Laboratory.
GitHub: https://github.com/RishvanthAmsaraj/SNL-RT-Research
```

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE).

---

## Acknowledgments

- Sensorimotor Neuroscience Laboratory
- KINARM robotic platform by BKIN Technologies
- PyMC and ArviZ development teams for Bayesian modeling tools
- HDDM / HSSM teams for the hierarchical-Bayesian precedent

---

**Maintainer:** Rishvanth Amsaraj  
**Status:** Active Research
