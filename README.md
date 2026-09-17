# SNL RT Research

**Hierarchical Bayesian drift-diffusion modeling of hand and saccadic reaction times from a KINARM interception task.**

A research project from the Penn State Sensorimotor Neuroscience Lab (ReSESENSE Labs). Fits single-boundary shifted-Wald models to reaction-time data from 16 participants performing an interception task at three target speeds (0, 75, 150 deg/s) — 7,676 trials total. The headline finding is a **dissociation in non-decision time**: hand t₀ decreases with target speed while saccadic t₀ does not.

---

## The Model

The KINARM interception task is *go-type* — participants initiate a movement when the target appears, with no binary choice. The correct descriptive model is the **single-boundary diffusion**, whose first-passage-time density is the **shifted Wald** (inverse Gaussian with temporal shift t₀).

| Parameter | Symbol | Meaning |
|---|---|---|
| Drift rate | *v* | Rate of evidence accumulation toward the action threshold |
| Boundary separation | *a* | Evidence required before response initiation |
| Non-decision time | *t₀* | Sensory + motor time outside the decision process |

**Two estimation methods:**

- **Method A (frequentist MLE).** Maximum likelihood via differential evolution with a 95% Wald + 5% uniform contamination mixture (Ratcliff & Tuerlinckx, 2002).
- **Method B (hierarchical Bayesian).** Partial pooling across participant × speed units using PyMC/NUTS, with non-centered parametrization and R-hat convergence diagnostics (Wiecki et al., 2013; Gelman et al., 2013).

Additional components: **express/regular two-component Wald mixture** for bimodal saccade cells (flagged by Hartigan's dip test).

---

## Key Findings

### The Dissociation

| Speed | Hand t₀ (95% CI) | Saccadic t₀ |
|---|---|---|
| 0 deg/s | 170 ms (154–182) | Fixed at 70 ms |
| 75 deg/s | 158 ms (139–174) | Fixed at 70 ms |
| 150 deg/s | 148 ms (130–162) | Fixed at 70 ms |
| **Friedman p** | **0.003** | Not testable (fixed) |

- **Hand t₀ decreases with target speed** — participants initiate hand responses faster when the target moves faster. The effect is triangulated by Friedman, participant-resampling bootstrap, and within-participant permutation tests.
- **Saccadic t₀ is not identifiable above the physiological floor.** The data cannot separate non-decision time from sensory/motor conduction in saccades. The model collapses to 70 ms for all participants when estimated at the participant level — a diagnosis, not a bug.
- **The saccadic floor-piling has a mechanism.** Saccadic latencies have low skew/CV (~3.4), which forces an implied t₀ of ~20–30 ms — below the 70 ms physiological minimum. Hand latencies have high skew/CV (~12.9), which pushes implied t₀ to ~191 ms — well above the 130 ms floor and identifiable.

### Parameter Bounds (Literature-Anchored, Phase 2)

| Bound | Value | Source |
|---|---|---|
| Drift cap V<sub>max</sub> | 20 | Tran et al. (2020) systematic review |
| Hand t₀ floor | 130 ms | Haith et al. (2016), reach-preparation minimum |
| HRT data filter | 150–800 ms | Whelan (2008) + Luce (1986) |
| SRT data filter | 80–600 ms | Fischer & Weber (1993) + Luce (1986) |
| Saccadic t₀ floor | 70 ms | Bompas & Sumner (2011); reported as fixed |

---

## Repository Structure

```
SNL-RT-Research/
│
├── Current Pipeline/              ← Active production pipeline (v2.0+)
│   ├── Code/
│   │   ├── Bayesian/              Hierarchical Bayesian fits (NUTS)
│   │   ├── DDM/                   Frequentist MLE fits (diagnostic/comparison)
│   │   ├── NDT/                   Non-decision time bar charts
│   │   ├── SRT Analysis/          Identifiability checks, sensitivity sweeps
│   │   ├── Vincentile/            Model-free raw-RT distribution figures
│   │   └── CODE_REFERENCE.md      Canonical model specification & run order
│   ├── Documents/                 Reports, justification PDFs, RUN_GUIDE
│   ├── Figures/                   All generated figures (PDF + PNG)
│   └── ISSUES_AND_IMPROVEMENTS.md
│
├── Deprecated Pipelines/          ← Preserved historical versions
│   ├── Deprecated Ver 1/          Phase 0: PyDDM prototypes, synthetic data
│   ├── Deprecated Ver 2/          Phase 0: Native MLE, first real pipeline
│   ├── Deprecated Ver 2.5/        Phase 1: Early Bayesian refinement
│   └── Deprecated Ver 3/          Phase 1–1.5: Full hierarchical, dissociation
│
├── kinarm-rt-app/                 ← Streamlit GUI + headless CLI
│   ├── app.py                     Interactive point-and-click interface
│   ├── run_pipeline.py            Batch mode (config-driven, no GUI)
│   ├── Dockerfile                 Reproducible container (conda-forge PyMC)
│   └── README.md                  App user guide — installation, usage, deployment
│
├── DEVELOPMENT_HISTORY.md         Complete project evolution narrative
├── CHANGELOG.md                   Keep a Changelog format with version-to-folder mapping
├── REFERENCES.md                  Organized bibliography by topic with species/task context
├── REFERENCES.bib                 BibTeX references
└── LICENSE                        MIT
```

Each deprecated version carries its own `ISSUES_AND_IMPROVEMENTS.md` documenting why it was superseded. See [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) for the full narrative arc.

---

## Documentation Index

| Document | What it covers |
|---|---|
| [`CODE_REFERENCE.md`](Current%20Pipeline/Code/CODE_REFERENCE.md) | Model specification, parameters, bounds, priors, run order, DOIs |
| [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) | Full evolution from PyDDM toy → literature-anchored Bayesian pipeline |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history with repo folder mapping, per-version changes, migration guide |
| [`REFERENCES.md`](REFERENCES.md) | Citations organized by role (Tier 1 core / Tier 2 context / Tier 3 general) |
| [`Current Pipeline/ISSUES_AND_IMPROVEMENTS.md`](Current%20Pipeline/ISSUES_AND_IMPROVEMENTS.md) | Known limitations, resolved items, future roadmap |
| [`Current Pipeline/Documents/RUN_GUIDE.md`](Current%20Pipeline/Documents/RUN_GUIDE.md) | Installation and execution order for the research pipeline |

---

## The App

The [`kinarm-rt-app/`](kinarm-rt-app/) directory contains a point-and-click Streamlit application and a headless CLI that reproduce the full pipeline. It fits the same models (shifted-Wald hierarchical Bayesian, MLE with contamination, express/regular mixtures) and adds analyses beyond the basic fit — dissociation test battery, parameter recovery, sensitivity sweeps, PSIS-LOO model comparison, and per-speed hierarchical models with LKJ correlated effects.

The app is the recommended entry point for anyone who wants to explore the models without writing code. It is validated against the real `pooled_data.csv`: hand t₀ per-cell correlation r = 0.999.

→ **[kinarm-rt-app/README.md](kinarm-rt-app/README.md)** — installation, usage, data format, and deployment options

---

## Quick Start

```bash
git clone https://github.com/RishvanthAmsaraj/SNL-RT-Research.git
cd SNL-RT-Research

# Pipeline (requires Python + PyMC — see RUN_GUIDE.md):
cd "Current Pipeline/Code"
# Run order: fits → figures → diagnostics

# App (no coding required):
cd kinarm-rt-app
pip install -r requirements.txt
streamlit run app.py
```

---

## Key References

- Anders, R., Alario, F.-X., & Van Maanen, L. (2016). The shifted Wald distribution for response time data analysis. *Psychological Methods*, 21(3), 309–327.
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. *Frontiers in Neuroinformatics*, 7, 14.
- Tran, N., van Maanen, L., Heathcote, A., & Matzke, D. (2020). Systematic parameter reviews in cognitive modeling. *Frontiers in Psychology*, 11, 608287.
- Haith, A. M., Pakpoor, J., & Krakauer, J. W. (2016). Independence of movement preparation and movement initiation. *Journal of Neuroscience*, 36(10), 3007–3015.
- Carpenter, R. H. S., & Williams, M. L. L. (1995). Neural computation of log likelihood in control of saccadic eye movements. *Nature*, 377, 59–62.
- Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of the diffusion model. *Psychonomic Bulletin & Review*, 9(3), 438–481.

Full bibliography: [`REFERENCES.md`](REFERENCES.md) | [`REFERENCES.bib`](REFERENCES.bib)

---

## License

MIT — see [LICENSE](LICENSE).

---

*Research conducted at **ReSESENSE Labs**, Department of Informatics and Intelligent Systems, Penn State University. For questions about the pipeline or the KINARM dataset, contact the repository owner.*
