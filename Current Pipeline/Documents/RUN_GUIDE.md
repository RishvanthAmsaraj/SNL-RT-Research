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

### B. For the Bayesian half (conda — only if re-running Bayesian fits)
On Windows, `pip install pymc` fails without a C compiler. Use conda:
```
conda create -n snl python=3.11
conda activate snl
conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib scikit-learn diptest
```
If pymc is missing, the Bayesian scripts print these steps. Pre-validated outputs are already
saved in the folder, so re-running is optional.

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

## 4. Figure output reference

- `ddm_{measure}_{speed}_degs.pdf` — conceptual single-boundary diffusion schematics
- `DDM_summary.pdf` — DDM fit quality + non-decision time
- `NDT_barchart.pdf` — NDT by speed (DDM fits, with per-participant dots)
- `vincentile_results_fig1..4_*.pdf` — KDE overlay, histograms, Vincentile differences
- `Bayesian_summary.pdf` — Bayesian fit summary + per-participant saccadic t0
- `NDT_barchart_bayesian.pdf` — NDT by speed (Bayesian fits)
- `Bayesian_srt_ndt.pdf` — forest plot of per-participant saccadic t0
- `SRT_identifiability.pdf` — saccadic t0 identifiability diagnostic
- `SRT_fixedt0_sensitivity.pdf` — sensitivity analysis (t0 = 50/70/90 ms)
- `LATER_reciprobit.pdf` — LATER complement figure

---

## 5. If something errors

- `ModuleNotFoundError: No module named 'pymc'` → that's a Bayesian script; do section 0B (conda), or just use the saved Bayesian outputs.
- `ERROR: pooled_data.csv not found` → put `pooled_data.csv` in the same folder as the scripts.
- `ERROR: DDM_hrt_fits.csv not found` → run `python DDM_fit.py` first (section 1, step 1).
- On Windows, if `python` isn't found, use the full path, e.g.
  `& C:\path\to\python.exe DDM_fit.py`.

---

> Bound changes for this version are documented in [`CHANGELOG.md`](CHANGELOG.md) [v2.0.0].
>
> The pre-computed Bayesian CSVs were generated with reduced draws (500/600/2) for fast
> preview; reduced and full draws reproduce the point estimates. Re-run the full scripts
> (1500/1500/4) for final published numbers.
