# RUN GUIDE — Paradigm 2 (CIR, 75/100/125/150 deg/s)

Same two halves as Paradigm 1: **Method A (DDM, MLE)** runs on plain `pip`; **Method B (Bayesian)** needs PyMC
(conda recommended on Windows, exactly as in the Paradigm 1 RUN_GUIDE). Every fit table and figure is already
saved in this folder, so re-running is optional.

## 0. Setup (unchanged from Paradigm 1)

```
pip install numpy pandas scipy matplotlib scikit-learn diptest          # Method A + all figures
conda create -n snl python=3.11 && conda activate snl
conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib scikit-learn diptest   # Method B
```

## 1. One command

```
cd "Paradigm 2 Pipeline/Code"
python run_all_P2.py                     # full rerun (~15 min on 4 cores; ~50 min on 1 core)
python run_all_P2.py --skip-bayes-fits   # no PyMC: reuse the saved Bayesian tables, rerun everything else
python run_all_P2.py --data "<folder with the 16 CIR*_TRIAL_Summary_v0_1_36.csv files>"   # rebuild the input first
```

`run_all_P2.py` copies the scripts and `pooled_data_P2.csv` into `Code/_run/`, runs every step in order, stops at
the first failure (log in `Code/_run/log_<script>.txt`), and files outputs back into `Code/<sub>/` and `Figures/<sub>/`.

## 2. By hand (the Paradigm 1 way)

Put every script and `pooled_data_P2.csv` in one folder and run, in this order:

| # | Script | Produces | Needs |
|---|---|---|---|
| 0 | `build_pooled_data_P2.py <CIR folder>` | `pooled_data_P2.csv`, `pooled_data_P2_audit.txt` | the 16 CIR files |
| 1 | `DDM_fit.py` | `DDM_hrt_fits.csv`, `DDM_srt_fits.csv` | 0 |
| 2 | `DDM_figures.py`, `DDM_conceptual.py`, `NDT_barchart.py` | DDM figures (8 schematics) | 1 |
| 3 | `vincentile_figures.py`, `why_saccadic_t0_floors.py` | 4 vincentile figures, floor mechanism | 0 |
| 4 | `SRT_identifiability_check.py` | `SRT_identifiability.csv/.pdf/.png` | 1 |
| 5 | `SRT_fixed_t0_analysis.py` | `SRT_fixedt0_fits.csv`, `SRT_fixedt0_sensitivity.csv/.pdf/.png` | 1 |
| 6 | `Bayesian_HRT_fit.py` (PyMC) | `Bayesian_hrt_fits.csv`, `Bayesian_hrt_ndt.csv` | 0 |
| 7 | `Bayesian_SRT_fit.py` (PyMC; resumable, or one speed: `python Bayesian_SRT_fit.py 125`) | `Bayesian_srt_fits.csv` | 1 |
| 8 | `Bayesian_SRT_ndt.py` (PyMC; `--replot` redraws the forest plot from the saved table, no PyMC needed) | `Bayesian_srt_ndt.csv`, `_cells.csv`, forest plot | 1 |
| 9 | `Bayesian_figures.py`, `Bayesian_conceptual.py`, `NDT_barchart_bayesian.py` | Bayesian figures | 6–8 |
| 10 | `HRT_floor_control.py` | hand/eye floor-sweep control | 1, 4 |
| 11 | `SRT_QA_flag_sensitivity.py`, `direction_check.py` | QA-flag and Left/Right checks | 0 |
| 12 | `dissociation_tests.py` | speed-effect battery | 1, 6 |
| 13 | `parameter_recovery_P2.py` | t0 recovery study | 5, 6 |

Validation (optional): copy the **original, unmodified** Paradigm 1 `DDM_fit.py` and `Bayesian_HRT_fit.py` with the
Paradigm 1 `pooled_data.csv` into an empty folder, run them, then
`python P1_parity_check.py "<repo>/Current Pipeline/Code" <that folder>`.

## 3. What differs from Paradigm 1

Only configuration: speeds `[75, 100, 125, 150]`, block type `"P2"`, input `pooled_data_P2.csv`, a 4-speed colour
palette (75 and 150 keep their Paradigm 1 colours), 4-speed figure layouts, and figure text computed from the data
instead of typed in. Model, likelihood, priors, bounds, floors, RT windows, optimiser, mixture rule and sampler
settings (1500/1500/4, target 0.95, same seeds) are identical — see `Validation/PORT_DIFF_P1_to_P2.diff`.

## 4. Errors

- `ModuleNotFoundError: pymc` → Bayesian step; use conda (section 0) or `--skip-bayes-fits`.
- `ERROR: pooled_data_P2.csv not found` → run step 0 or put the file next to the scripts.
- A Bayesian run that was interrupted: `Bayesian_SRT_fit.py` resumes where it stopped; the others restart.
