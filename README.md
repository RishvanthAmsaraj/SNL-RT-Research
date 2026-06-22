# SNL RT Research

**Sensorimotor Neuroscience Laboratory - Reaction Time Analysis Pipeline**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Active-red)](https://github.com/RishvanthAmsaraj)

## Overview

This repository contains the analysis pipeline for reaction time (RT) research conducted at the Sensorimotor Neuroscience Laboratory (SNL). The project focuses on analyzing kinematic data from the Kinarm robotic platform to understand human motor control and decision-making processes.

### Research Focus

- **Hard Reaction Time (HRT)** - Analyzing rapid motor responses
- **Simple Reaction Time (SRT)** - Studying basic stimulus-response paradigms
- **Drift Diffusion Models (DDM)** - Modeling decision-making processes
- **Bayesian Analysis** - Probabilistic modeling of reaction time distributions
- **Non-Decision Time (NDT)** - Quantifying perceptual and motor execution components

## Repository Structure

```
SNL-RT-Research/
├── Current Pipeline/          # Active analysis pipeline
│   ├── Bayesian/             # Bayesian hierarchical models
│   ├── DDM/                  # Drift Diffusion Model fits
│   ├── NDT/                  # Non-Decision Time analysis
│   ├── Vincentile/           # Vincentile binning analysis
│   ├── figures/              # Generated figures and plots
│   └── RUN_GUIDE.md          # Pipeline execution guide
│
├── Deprecated Ver 1/          # Initial PyDDM implementations
│   ├── DualChoice*.py        # Dual choice task models
│   ├── SingleChoice*.py      # Single choice task models
│   └── *DataGen.py           # Synthetic data generators
│
└── Deprecated Ver 2/          # Previous lab pipeline version
    ├── Bayesian Model/       # Earlier Bayesian implementations
    ├── DDM Model/            # Previous DDM analysis code
    ├── NDT Code/             # Earlier NDT calculations
    ├── Vincentile Code/      # Previous vincentile methods
    └── Deprecated Figures/   # Legacy visualizations
```

## Current Pipeline

The active analysis pipeline (`Current Pipeline/`) implements a comprehensive workflow for RT data analysis:

### Components

1. **Bayesian Models** (`Bayesian/`)
   - Hierarchical Bayesian modeling of RT distributions
   - Separate models for HRT and SRT conditions
   - NDT extraction and analysis
   - Model comparison and validation

2. **Drift Diffusion Models** (`DDM/`)
   - Evidence accumulation modeling
   - Parameter estimation (drift rate, boundary separation, non-decision time)
   - Condition-specific fits (0°, 75°, 150° target angles)

3. **Non-Decision Time** (`NDT/`)
   - Perceptual processing time estimation
   - Motor execution time quantification
   - Comparative bar chart visualizations

4. **Vincentile Analysis** (`Vincentile/`)
   - Distribution binning for RT quantiles
   - Speed-accuracy trade-off visualization
   - Group-level summary statistics

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy
pip install pymc arviz  # For Bayesian models
pip install hddm        # For DDM analysis
```

### Running the Pipeline

See [`Current Pipeline/RUN_GUIDE.md`](Current%20Pipeline/RUN_GUIDE.md) for detailed execution instructions.

Basic workflow:

```bash
cd "Current Pipeline"

# 1. Run Bayesian models
python Bayesian/Bayesian_HRT_fit.py
python Bayesian/Bayesian_SRT_fit.py

# 2. Run DDM fits
python DDM/DDM_fit.py

# 3. Generate NDT analysis
python NDT/NDT_barchart.py

# 4. Create vincentile plots
python Vincentile/vincentile_figures.py

# 5. Generate publication figures
python Bayesian/Bayesian_figures.py
python DDM/DDM_figures.py
```

## Data Format

The pipeline expects Kinarm data in the following structure:
- Trial-level reaction times (ms)
- Target angle conditions (0°, 75°, 150°)
- Trial outcomes (correct/incorrect)
- Participant identifiers

## Version History & Evolution

This repository contains three major versions of the analysis pipeline, each
representing a significant methodological advancement. Detailed migration notes
explain why certain approaches were abandoned and what replaced them.

| Version | Status | Key Advancement |
|---------|--------|-----------------|
| **Current Pipeline** | ✅ Active | Hierarchical Bayesian framework; participant-level SRT t₀; full credible intervals |
| **Deprecated Ver 2** | ⚠️ Superseded | Native scipy MLE fitting; contamination mixture; per-participant fitting |
| **Deprecated Ver 1** | ⚠️ Archived | PyDDM proof-of-concept; synthetic data validation; basic visualization |

### Why Versions Were Deprecated

**Ver 1 → Ver 2:** PyDDM was too inflexible for custom likelihoods; switched to
native scipy optimization with single-boundary Wald model appropriate for
interception tasks. See [`Deprecated Ver 1/MIGRATION_NOTES.md`](Deprecated%20Ver%201/MIGRATION_NOTES.md).

**Ver 2 → Current:** Per-cell maximum likelihood cannot identify SRT non-decision
time (floor-piling artifact); hierarchical Bayesian estimation resolves this
through partial pooling. See [`Deprecated Ver 2/MIGRATION_NOTES.md`](Deprecated%20Ver%202/MIGRATION_NOTES.md).

### Documentation

- **[CHANGELOG.md](CHANGELOG.md)** — Complete version history with technical details
- **[Current Pipeline/ISSUES_AND_IMPROVEMENTS.md](Current%20Pipeline/ISSUES_AND_IMPROVEMENTS.md)** — Known issues and planned enhancements
- **[Current Pipeline/RUN_GUIDE.md](Current%20Pipeline/RUN_GUIDE.md)** — Step-by-step execution instructions

## Contributing & Development

This is an active research project. See [`ISSUES_AND_IMPROVEMENTS.md`](Current%20Pipeline/ISSUES_AND_IMPROVEMENTS.md)
for the development roadmap, known bugs, and planned features.

To contribute:
1. Check the issues list for open items
2. Create a feature branch
3. Add tests if applicable
4. Update CHANGELOG.md
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```
Amsaraj, R. (2024). SNL RT Research Pipeline. 
Sensorimotor Neuroscience Laboratory.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sensorimotor Neuroscience Laboratory
- Kinarm robotic platform by BKIN Technologies
- PyMC and ArviZ development teams for Bayesian modeling tools

---

**Last Updated:** June 2024  
**Maintainer:** Rishvanth Amsaraj  
**Status:** Active Research
