# Code Reference — SNL RT Research

Shared model and parameter documentation used across all scripts in this repository.

---

## Model: Single-Boundary Shifted Wald

The KINARM interception task is **go-type** (initiate a movement when the target appears; no binary choice). The correct first-passage-time density is the **shifted Wald** (inverse Gaussian with temporal shift $t_0$), equivalent to a single-boundary diffusion.

### Parameters

| Symbol | Name | Meaning |
|---|---|---|
| $v$ | Drift rate | Rate of evidence accumulation toward the action threshold |
| $a$ | Boundary separation | Evidence required before response initiation |
| $t_0$ | Non-decision time | Sensory + motor time outside the decision process |

### Likelihood (shifted Wald, with 5% uniform contamination)

```
log PDF(τ; v, a) = log a − ½·log(2π) − 1.5·log(τ) − (a − v·τ)² / (2τ)
```
where $\tau = RT - t_0$ (decision time only). Full likelihood: $0.95 \cdot \text{Wald} + 0.05 \cdot \text{Uniform}(RT_{\min}, RT_{\max})$.

Core refs: Anders et al. (2016), Heathcote (2004), Ratcliff & Tuerlinckx (2002).

### Bounds

| Bound | Value | Source |
|---|---|---|
| Drift cap $V_{\max}$ | 20 | Tran et al. (2020) systematic review envelope at $s=1$: $\|v\| \lesssim 18.5$ |
| Hand $t_0$ floor | 130 ms | Haith et al. (2016): reach-preparation minimum |
| SRT data filter | 80–600 ms | Human anticipation threshold (Fischer & Weber 1993) + lapse removal (Luce 1986) |
| HRT data filter | 150–800 ms | Anticipation removal (Whelan 2008) + lapse removal (Luce 1986) |
| SR $t_0$ floor (per-cell) | 70 ms | Saccadic afferent+efferent conduction floor (Bompas & Sumner 2011; Ludwig et al. 2007) |

The 70 ms floor is the **physiological minimum**; fast saccadic cells have per-cell t₀ that tracks this floor rather than being identified by the data (see `SRT_identifiability_check.py`). For the Bayesian per-participant model, t₀ collapses to 70 ms for all participants — it is reported as a **fixed constant** rather than an estimated parameter.

### Mixture model for express/regular saccades

Cells flagged as bimodal (by the DDM fit-driven + structural approach) use a two-component mixture: $\pi \cdot \text{Wald}(\text{express}) + (1-\pi) \cdot \text{Wald}(\text{regular})$, with post-hoc label relabeling to assign the faster mode as "express."

---

## Hierarchical Bayesian Architecture (Method B)

Architecture follows HDDM/Wiecki et al. (2013) — hierarchical Bayesian with partial pooling using PyMC/NUTS:

- **Parameters drawn per participant** from a group-level distribution (e.g., $v_i \sim \text{Normal}(\mu_v, \sigma_v)$)
- Post-hoc R-hat convergence (Gelman & Rubin 1992; threshold $< 1.01$)
- Non-centered parametrization for efficient sampling (Gelman et al. 2013)
- All scripts resumable (completed cells saved to CSV immediately; skipped on restart)

HDDM/HSSM fit the *two-choice* DDM; our pipeline fits the *single-boundary* shifted Wald (appropriate for go-type tasks).

---

## Data: `pooled_data.csv`

Canonical input (7,676 trials). Key columns:
- `Participant`, `Speed_deg_per_s` (0 / 75 / 150)
- `HandRT_ms`, `GazeSRT_ms` — reaction times (filtered per bounds above)
- `BlockType` — filter to `'I'` (interception) trials

---

## Run Order

All scripts read CSV outputs from previous stages (fit tables → figures → diagnostics):

```
DDM_fit.py ──────────────────► DDM_hrt_fits.csv, DDM_srt_fits.csv
  ├─► Bayesian_HRT_fit.py ──► Bayesian_hrt_fits.csv, Bayesian_hrt_ndt.csv
  ├─► Bayesian_SRT_fit.py ──► Bayesian_srt_fits.csv
  ├─► DDM_figures.py / DDM_conceptual.py
  ├─► NDT_barchart.py
  └─► SRT_identifiability_check.py / SRT_fixed_t0_analysis.py
         └─► Bayesian_SRT_ndt.py (depends on DDM_srt_fits.csv for single/mixture split)
                ├─► Bayesian_figures.py / Bayesian_conceptual.py
                ├─► NDT_barchart_bayesian.py
                └─► why_saccadic_t0_floors.py / LATER_analysis.py

Vincentile/vincentile_figures.py (model-free; reads pooled_data.csv only)
```

**Bayesian scripts require PyMC** (conda install recommended on Windows). Pre-validated outputs (CSV fit tables, PDF/PNG figures) are saved alongside each script so results are complete even without re-running.

---

## DOIs

- Anders et al. (2016): 10.1037/met0000042
- Heathcote (2004): 10.3758/BF03206577
- Ratcliff & Tuerlinckx (2002): 10.3758/BF03196305
- Haith et al. (2016): 10.1523/JNEUROSCI.3609-15.2016
- Tran et al. (2020): 10.3389/fpsyg.2020.608287
- Knox & Wolohan (2015): **10.1016/j.visres.2014.12.010** → `e0120437`
- Carpenter & Williams (1995): 10.1038/377059a0
- Wiecki et al. (2013): 10.3389/fninf.2013.00014
- Gelman & Rubin (1992): 10.1214/ss/1177011136
- Fischer & Weber (1993): 10.1016/S0149-7634(05)80110-4
- Bompas & Sumner (2011): 10.1167/11.5.9
- Whelan (2008): 10.3758/BRM.40.3.725
- Luce (1986): ISBN 0-19-507001-X
- Shinn et al. (2020): 10.7554/eLife.57394
- Böhm & Ulrich (2018): 10.1016/j.jmp.2018.01.002
- Hartigan & Hartigan (1985): 10.1214/aos/1176346577

See [`REFERENCES.md`](../REFERENCES.md) and [`REFERENCES.bib`](../REFERENCES.bib) for full bibliography.
