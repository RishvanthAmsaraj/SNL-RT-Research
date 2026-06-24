# RUN GUIDE — KINARM RT analysis (DDM + Bayesian)

A start-to-finish walkthrough. Two halves: **Method A (DDM)** runs anywhere with plain `pip`;
**Method B (Bayesian)** needs `pymc`, which on Windows installs via **conda**. The Bayesian
results are already saved, so you can reproduce everything in Method A and still have the
Bayesian numbers/figures without installing pymc.

Everything assumes all scripts + `pooled_data.csv` sit in **one folder**, and you run commands
from inside that folder.

---

## 0. One-time setup

### A. For the DDM half (pip — works on your current Python)
```
pip install -r requirements.txt
```
That installs: numpy, pandas, scipy, matplotlib, scikit-learn, diptest.

### B. For the Bayesian half (conda — only if you want to RE-RUN the Bayesian fits)
`pip install pymc` fails on the Windows Store Python (no C compiler). Use conda:
```
# 1. install Miniconda (no admin needed): https://docs.conda.io/en/latest/miniconda.html
# 2. open "Anaconda Prompt", then:
conda create -n snl python=3.11
conda activate snl
conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib scikit-learn diptest
```
From then on, run the Bayesian scripts from inside the `snl` env (`conda activate snl` first).
If pymc is missing, the Bayesian scripts print these exact steps instead of crashing.

> You do **not** have to run the Bayesian fits — their outputs (`Bayesian_*_fits.csv`,
> `Bayesian_srt_ndt.csv`, and the two Bayesian figures) are already in the folder.

---

## 1. Run order — Method A (DDM)

Run these in order from your folder (Windows: replace `python` with your interpreter path if needed).

| # | Command | Produces | Needs |
|---|---------|----------|-------|
| 1 | `python DDM_fit.py` | `DDM_hrt_fits.csv`, `DDM_srt_fits.csv` | `pooled_data.csv` |
| 2 | `python DDM_figures.py` | `DDM_summary.pdf/.png` | step 1 |
| 3 | `python DDM_conceptual.py` | `ddm_hrt_{0,75,150}_degs.*`, `ddm_srt_{0,75,150}_degs.*` (6 schematics) | step 1 + `pooled_data.csv` |
| 4 | `python NDT_barchart.py` | `NDT_barchart.pdf/.png` | step 1 |
| 5 | `python vincentile_figures.py` | `vincentile_results_fig1..4_*.pdf` | `pooled_data.csv` only |
| 6 | `python SRT_identifiability_check.py` | `SRT_identifiability.pdf/.png` | step 1 + `pooled_data.csv` |
| 7 | `python SRT_fixed_t0_analysis.py` | `SRT_fixedt0_sensitivity.pdf/.png`, `SRT_fixedt0_fits.csv` | step 1 + `pooled_data.csv` |

**Step 1 must run first** (everything except step 5 reads its fit tables). Steps 2–7 can run in
any order after that. Step 5 (the Vincentile figures) only needs `pooled_data.csv`, so it can run
even before step 1.

---

## 2. Run order — Method B (Bayesian)  *(optional; needs the conda env from 0B)*

```
conda activate snl
```

| # | Command | Produces | Needs | Time |
|---|---------|----------|-------|------|
| 1 | `python Bayesian_HRT_fit.py` | `Bayesian_hrt_fits.csv` | `pooled_data.csv` | ~10 min |
| 2 | `python Bayesian_SRT_fit.py` | `Bayesian_srt_fits.csv` | `pooled_data.csv` | ~10 min |
| 3 | `python Bayesian_SRT_ndt.py` | `Bayesian_srt_ndt.csv`, `Bayesian_srt_ndt_cells.csv` | `pooled_data.csv` | ~12 min |
| 4 | `python Bayesian_figures.py` | `Bayesian_summary.pdf/.png`, `Bayesian_srt_ndt.pdf/.png` | steps 1–3 (esp. `Bayesian_srt_ndt.csv` for panel C) | seconds |

`Bayesian_figures.py` uses only matplotlib (no pymc), so it will regenerate the two Bayesian
figures from the CSVs even on your pip-only Python — handy if you just want to re-plot.

---

## 3. Quickest path (TL;DR)

```
pip install -r requirements.txt
python DDM_fit.py
python DDM_figures.py
python DDM_conceptual.py
python NDT_barchart.py
python vincentile_figures.py
python SRT_identifiability_check.py
python SRT_fixed_t0_analysis.py
```
That reproduces the entire DDM figure suite. For the Bayesian figures, either set up conda
(section 0B) and run section 2, or just use the Bayesian CSVs/figures already in the folder.

---

## 4. What each figure is

**Matched to the original suite**
- `ddm_{hrt,srt}_{0,75,150}_degs.pdf` — conceptual single-boundary diffusion schematic per measure × speed
- `DDM_summary.pdf` — A: HRT fit quality · B: SRT fit quality (mixture) · C: HRT non-decision time
- `NDT_barchart.pdf` — HRT & SRT non-decision time by speed, **with every participant as a dot** (so floor-piling is visible)
- `vincentile_results_fig1_kde_overlay.pdf` — SRT vs HRT density per speed
- `vincentile_results_fig2_histograms.pdf` — SRT (top) & HRT (bottom) histograms per speed
- `vincentile_results_fig3_vincentile_by_speed.pdf` — HRT−SRT Vincentile (per-trial paired), 3 panels ±1 SD
- `vincentile_results_fig4_combined_vincentile.pdf` — HRT−SRT Vincentile, all speeds overlaid

**Additional (kept because they're useful)**
- `SRT_identifiability.pdf` — shows per-cell saccadic t0 tracks whatever floor is imposed (the identifiability problem)
- `SRT_fixedt0_sensitivity.pdf` — drift/boundary conclusions are stable whether t0 is fixed at 50/70/90 ms
- `Bayesian_summary.pdf` — HRT degeneracy resolved + per-participant saccadic t0
- `Bayesian_srt_ndt.pdf` — per-participant saccadic t0, **estimated** (individual differences kept), each with its own 95% CI

---

## 5. If something errors

- `ModuleNotFoundError: No module named 'pymc'` → that's a Bayesian script; do section 0B (conda), or just use the saved Bayesian outputs.
- `ERROR: pooled_data.csv not found` → put `pooled_data.csv` in the same folder as the scripts.
- `ERROR: DDM_hrt_fits.csv not found` → run `python DDM_fit.py` first (section 1, step 1).
- On Windows, if `python` isn't found, use the full path, e.g.
  `& C:\path\to\python.exe DDM_fit.py`.

---

## REVISION NOTES (this version) — literature-anchored bounds

This version applies bound changes anchored to a systematic-review reading of the DDM
parameter literature (Tran et al. 2020; Ratcliff et al. 2016) plus reaching/saccade/eye-hand
timing evidence. All changes and their sources are documented in the references document
(`Methodological_References.pdf`, Section 9).

**Bound changes applied:**
- Drift cap `V_MAX` 40 → **20** (DDM + diagnostics): Tran et al. (2020) s=1 envelope |v| ≲ 18.5.
  Drift is a latent/policy parameter, so a literature cap is defensible; it binds on a few
  fast cells only.
- Hand non-decision floor 100 → **130 ms** (`DDM_fit.py`, `Bayesian_HRT_fit.py`):
  reach-preparation floor (Haith et al. 2016). Barely binds (fitted min was 129 ms);
  **the headline HRT result survives** — t₀ still decreases 170→158→148 ms with speed,
  Friedman p = 0.003, with 0/48 cells floored.
- Saccade non-decision floor (per-cell) **70 ms** (unchanged; already literature-aligned:
  Bompas et al. 2017; Ludwig et al. 2007).
- Saccade non-decision floor (per-participant NDT) 35 → **70 ms** (`Bayesian_SRT_ndt.py`),
  harmonized with the per-cell fit and the literature.

**Important — the SRT non-identifiability result.** Enforcing the 70 ms floor on the
per-participant SRT NDT model **collapses it**: all 14 participants pin at 70–71 ms because
the data favour a value below the floor (population mean ≈ 40 ms). This is honest evidence
that saccadic non-decision time is not identifiable above the physiological floor — a direct
instance of the Bompas et al. (2024) fitted-vs-physiological divergence, and consistent with
`SRT_identifiability_check.py` and `SRT_fixed_t0_analysis.py` (which already fix saccadic
t₀ = 70 ms for exactly this reason). **Saccadic t₀ is therefore reported as fixed at the
70 ms physiological floor, not estimated per participant.** `NDT_barchart_bayesian.pdf` now
shows this honestly (SRT panel labelled "not identifiable — pinned at the 70 ms floor").

**Interception caveat (report in the manuscript).** In moving-target conditions, fitted
non-decision time can absorb decision–action overlap and urgency (Barany et al. 2020;
Bompas et al. 2024; Weindel et al. 2020). The hand t₀ decrease with speed is therefore
consistent with *either* a genuine non-decision-time reduction *or* greater overlap-absorption
at higher speeds; the saccadic invariance partially (not fully) supports an effector-specific
reading. Report the dissociation as real in the fitted parameters and flag this alternative.

**Note on draws.** The shipped Bayesian fit scripts use full sampling (1500 draws / 1500 tune
/ 4 chains). The pre-computed CSVs in this package were generated with reduced draws
(500/600/2) for fast preview; reduced and full draws reproduce the point estimates. Re-run
the full scripts for final published numbers.
