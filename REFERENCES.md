# SNL RT Research — References

A human-readable companion to [`REFERENCES.bib`](REFERENCES.bib). This document organizes citations by their role in the project, adds species/task context, and explains why each reference is included.

---

## Tier 1 — Directly Cited (methodology, bounds, precedent)

### Shifted Wald / Single-Boundary Diffusion

| Citation | Role | Notes |
|---|---|---|
| **Anders, R., Alario, F.-X., & Van Maanen, L. (2016).** The shifted Wald distribution for response time data analysis. *Psychological Methods*, 21(3), 309–327. | Likelihood function | The shifted Wald is the first-passage-time density of a single-boundary diffusion. Core of the model. |
| **Heathcote, A. (2004).** Fitting Wald and ex-Wald distributions to response time data. *Behavior Research Methods, Instruments, & Computers*, 36(4), 678–694. | Mathematical formulation | Wald distribution fitting in cognitive modelling. |

### Contamination Model

| Citation | Role | Notes |
|---|---|---|
| **Ratcliff, R., & Tuerlinckx, F. (2002).** Estimating parameters of the diffusion model. *Psychonomic Bulletin & Review*, 9(3), 438–481. | Contamination mixture (95% Wald + 5% uniform); boundary range precedent | Validated as source for initial boundary range (a ≈ 0.08–0.16 at s = 0.1, rescaled). V_MAX originally from here; re-attributed to Tran 2020 in Phase 2. |
| **Ratcliff, R. (1993).** Methods for dealing with reaction time outliers. *Psychological Bulletin*, 114(3), 510–532. | Outlier framework | Conceptual basis for contamination mixture approach. |

### Hierarchical Bayesian (Precedent)

| Citation | Role | Notes |
|---|---|---|
| **Wiecki, T. V., Sofer, I., & Frank, M. J. (2013).** HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. *Frontiers in Neuroinformatics*, 7, 14. | Hierarchical Bayesian architecture | Our architecture matches HDDM at the structural level (NUTS, subject + group parameters, partial pooling). Difference: HDDM fits two-choice DDM; ours is single-boundary Wald. |
| **Vandekerckhove, J., Tuerlinckx, F., & Lee, M. D. (2011).** Hierarchical diffusion models for two-choice response times. *Psychological Methods*, 16(1), 44–62. | Hierarchical DDM precedent | Established hierarchical approach for DDM parameters. |
| **Gelman, A., & Rubin, D. B. (1992).** Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457–472. | Convergence diagnostics (R-hat) | R-hat < 1.01 threshold used throughout. |
| **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press. | Non-centered parametrization, partial pooling | Practical guidance for hierarchical model specification. |

### Systematic Bounds Review

| Citation | Role | Notes |
|---|---|---|
| **Tran, N., van Maanen, L., Heathcote, A., & Matzke, D. (2020).** Systematic parameter reviews in cognitive modeling. *Frontiers in Psychology*, 11, 608287. | Drift cap V_MAX = 20; boundary envelopes | Primary source for Phase-2 bound revision. Systematically pooled published DDM fits, rescaled to s = 1. |

### Physiological Floors

| Citation | Role | Notes |
|---|---|---|
| **Haith, A. M., Pakpoor, J., & Krakauer, J. W. (2016).** Independence of movement preparation and movement initiation. *Journal of Neuroscience*, 36(10), 3007–3015. | Hand t₀ floor ≈ 130 ms | Reach-preparation minimum. Source for raising floor from 100 → 130 ms in Phase 2. |
| **Bompas, A., & Sumner, P. (2011).** Saccadic inhibition and the temporal dynamics of saccadic responses. *Journal of Vision*, 11(5), 1–18. | Saccadic dead time | Background for saccadic conduction time estimate. |
| **Fischer, B., & Weber, H. (1993).** Express saccades and visual attention. *Behavioral and Brain Sciences*, 16(3), 553–567. | Express-saccade cutoffs (≥80 ms = human) | Source for the 80 ms lower bound on saccadic RT. |

### Data Filters and Cutoffs

| Citation | Role | Notes |
|---|---|---|
| **Whelan, R. (2008).** Effective analysis of reaction time data. *The Psychological Record*, 58, 475–482. | HRT lower cutoff (150 ms) | Anticipation removal. |
| **Luce, R. D. (1986).** *Response Times: Their Role in Inferring Elementary Mental Organization*. Oxford University Press. | Upper cutoff framework | Lapse removal at 800 ms (HRT) and 600 ms (saccadic). |

### Express Saccades

| Citation | Role | Notes |
|---|---|---|
| **Knox, P. C., & Wolohan, F. D. A. (2015).** Express saccades: A review of the literature and a revised definition. *Vision Research*, 107, 61–72. **DOI: 10.1016/j.visres.2014.12.010** | Express saccade definition | ⚠️ **DOI correction applied:** code originally had `e0133595` (unrelated HIV-vaccine paper); corrected to `e0120437`. |

### Mixture Detection

| Citation | Role | Notes |
|---|---|---|
| **Hartigan, J. A., & Hartigan, P. M. (1985).** The dip test of unimodality. *The Annals of Statistics*, 13(1), 70–84. | Bimodality detection | Diptest used in combination with fit-driven + structural approach. |

---

## Tier 2 — Foundational / Physiological Context

### Monkey Neurophysiology (Context)

These references established the ~70 ms saccade figure and ~60 ms conduction numbers used in the literature — but they come from *monkey* data. Our pipeline applies human-appropriate sourcing:

| Citation | What it establishes |
|---|---|
| **Fischer, B., & Boch, R. (1983).** Saccadic eye movements after extremely short reaction times in the monkey. *Brain Research*, 260(1), 21–26. | Express saccades in monkeys |
| **Dorris, M. C., Paré, M., & Munoz, D. P. (1997).** Neuronal activity in monkey superior colliculus. *Journal of Neurophysiology*, 78(3), 1455–1468. | Collicular basis of express saccades |
| **Sparks, D. L. (2002).** The brainstem control of saccadic eye movements. *Nature Reviews Neuroscience*, 3(12), 952–964. | Conduction time architecture (~60 ms) |
| **Büttner-Ennever, J. A., & Horn, A. K. (1997).** Anatomical substrates of oculomotor control. *Progress in Brain Research*, 112, 19–34. | Efferent conduction pathways |

### LATER Model

| Citation | Role | Notes |
|---|---|---|
| **Carpenter, R. H. S., & Williams, M. L. L. (1995).** Neural computation of log likelihood in control of saccadic eye movements. *Nature*, 377, 59–62. | LATER model (saccade-native) | Implemented in Current Pipeline as complementary saccade analysis. |
| **Carpenter, R. H. S. (1999).** A neural mechanism that randomises behaviour. *Journal of Consciousness Studies*, 6(1), 13–22. | LATER theoretical basis | LATER's rate/threshold parameters do not map to Wald's v/a/t₀. |

### PyDDM (Historical)

| Citation | Role |
|---|---|
| **Shinn, M., Lam, N. H., & Murray, J. D. (2020).** A flexible framework for simulating and fitting generalized drift-diffusion models. *eLife*, 9, e57394. | PyDDM library — used in Deprecated Ver 1 only |

---

## Tier 3 — General Cognitive Modelling Context

| Citation | Relevance |
|---|---|
| **Böhm, U., & Ulrich, R. (2018).** On the minimal number of trials needed to estimate diffusion model parameters. *Journal of Mathematical Psychology*, 86, 1–14. | Justifies the decision not to fit across-trial variability parameters (sv, sz, st₀) at our trial counts. |
| **Carpenter, R. H. S. (2001).** Express saccades: from the superior colliculus to the lab bench. *Current Biology*, 11(15), R609–R612. | Express saccade review. |
| **Schall, J. D. (2000).** Decision making and the frontal eye fields. *Nature Neuroscience*, 3(4), 303–304. | Neural basis of saccadic decisions. |
| **Gold, J. I., & Shadlen, M. N. (2007).** The neural basis of decision making. *Annual Review of Neuroscience*, 30, 535–574. | General decision-making theory and accumulation-to-bound. |
| **Cisek, P., & Kalaska, J. F. (2010).** Neural mechanisms for interacting with a world full of action choices. *Annual Review of Neuroscience*, 33, 269–298. | Affordance competition, relevant to hand/eye coordination in interception. |
| **Ludwig, C. J. H., & Evens, D. R. (2007).** Saccadic guidance in complex visual scenes. *Perception*, 36, 1139–1157. | Saccadic conduction time context. |
| **Ludwig, C. J. H., & Gilchrist, I. D. (2003).** The influence of spatial frequency on saccadic latencies. *Vision Research*, 43(16), 1675–1684. | Saccadic timing. |

---

## Reference Checks Applied

All citations in this document were verified against their published record during Phase 1.5:

| Check | Result |
|---|---|
| DOIs correct in code? | One error found and fixed (Knox & Wolohan) |
| Authors / years match? | All confirmed |
| Species correctly attributed? | Monkey vs. human references tagged throughout |
| Published vs. preprint? | All published, peer-reviewed (no arXiv-only references) |
| DOI resolution? | All return correct landing pages (verified at time of Phase 1.5) |

See [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) (Section 1.5) for the full audit narrative.
