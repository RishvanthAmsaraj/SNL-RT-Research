# KINARM Interception RT — Analysis Pipeline (DDM + Bayesian)

Estimates the three diffusion-model variables — **drift rate (v)**, **boundary (a)**, and
**non-decision time (t₀)** — for hand RT (HRT) and saccadic RT (SRT) across 16 participants
× 3 target speeds (0 / 75 / 150 deg/s).

Two methods are included:

- **Method A — DDM (frequentist / MLE).** Runs with plain `pip`. Kept as the comparison/diagnostic.
- **Method B — Bayesian (hierarchical, partial pooling).** **This is the method to report.**
  It removes the non-decision-time floor-piling and returns full credible intervals, i.e. the
  most accurate estimates of the three variables.

> The DDM figures are included **only** so you can show, side by side, that the Bayesian model
> removes the flooring and gives more accurate estimates. Report the Bayesian numbers.

---

## The one methodological rule (read this)

For each variable, **all participants use the same Bayesian model.** Do **not** use DDM for some
participants and Bayesian for others. The estimates would not be comparable, and the participants
you'd be tempted to "leave on the DDM" (the fast, low-variance ones) are *exactly* the ones that
floor. The Bayesian model handles them correctly by regularizing toward the population with wide
credible intervals — that is the fix, not a reason to switch them back to the DDM.

## Why SRT non-decision time is treated differently from HRT

- **HRT t₀ is identifiable** per participant × speed, so it is reported **by speed** (it *decreases*
  with target speed; Friedman p ≈ 0.002 on the Bayesian fits, 0/48 cells floored).
- **SRT t₀ is NOT identifiable per cell** — fast saccades cannot separate non-decision time from
  decision time, so *any* per-cell estimate (DDM **or** Bayesian) slides to the floor. We verified
  this directly: the Bayesian **per-cell** SRT fit still floors **19/33** cells at 70 ms. The correct
  treatment estimates **one t₀ per participant, shared across speed** (`Bayesian_SRT_ndt.py`); this
  removes the flooring (**0 floored**) and is reported with per-participant credible intervals.
  So there is no valid "Bayesian SRT t₀ by speed" — the honest Bayesian SRT result is per-participant.

---

## Setup

**Method A (DDM) — pip, works on your current Python**
```
pip install numpy pandas scipy matplotlib scikit-learn diptest
```

**Method B (Bayesian) — conda (PyMC needs a compiler; on Windows use conda)**
```
# install Miniconda, then:
conda create -n snl python=3.11
conda activate snl
conda install -c conda-forge pymc arviz numpy scipy pandas matplotlib scikit-learn diptest
```

Put every script + `pooled_data.csv` in **one folder**, run commands from inside it.
`pooled_data.csv` (included) is the 16 `CMT*_MASTER_Summary.csv` files concatenated with a
`Participant` column.

---

## Run order

### Method A — DDM (pip)
| # | Command | Produces |
|---|---------|----------|
| 1 | `python DDM_fit.py` | `DDM_hrt_fits.csv`, `DDM_srt_fits.csv` |
| 2 | `python DDM_figures.py` | `DDM_summary.pdf` |
| 3 | `python DDM_conceptual.py` | `ddm_{hrt,srt}_{0,75,150}_degs.pdf` (6) |
| 4 | `python NDT_barchart.py` | `NDT_barchart.pdf` (DDM diagnostic — shows the flooring) |
| 5 | `python vincentile_figures.py` | 4 vincentile figures (model-free; identical for both methods) |

### Method B — Bayesian (conda) — **THE RESULTS**
| # | Command | Produces |
|---|---------|----------|
| 1 | `python Bayesian_HRT_fit.py` | `Bayesian_hrt_fits.csv`, `Bayesian_hrt_ndt.csv` |
| 2 | `python Bayesian_SRT_fit.py` | `Bayesian_srt_fits.csv` |
| 3 | `python Bayesian_SRT_ndt.py` | `Bayesian_srt_ndt.csv` (+ `_cells.csv`), `Bayesian_srt_ndt.pdf` |
| 4 | `python Bayesian_figures.py` | `Bayesian_summary.pdf` |
| 5 | `python Bayesian_conceptual.py` | `bayes_{hrt,srt}_{0,75,150}_degs.pdf` (6) |
| 6 | `python NDT_barchart_bayesian.py` | `NDT_barchart_bayesian.pdf` (the non-decision-time result) |

**Step 1 of each method must run first** (everything else reads the fit tables). After that,
the figure steps can run in any order. The vincentile figures need only `pooled_data.csv`.

---

##  Note on the CSVs / figures included in this package

The `Bayesian_*.csv` tables and the rendered figures here were produced with **reduced MCMC
sampling (600 draws / 2 chains)** so they could be generated quickly as a preview. The point
estimates match the full run (e.g. HRT t₀ = 168 / 156 / 147 ms identically). **For the final
paper numbers, re-run the Method B scripts** — they default to the full settings
(1500 draws / 4 chains), so simply running them regenerates these tables at full precision.

If you already have your full-draw Bayesian CSVs from earlier, just drop the **figure** scripts
(`Bayesian_figures.py`, `Bayesian_conceptual.py`, `NDT_barchart_bayesian.py`,
`vincentile_figures.py`) into that folder and run them — they read the standard CSV names and
will produce the final figures directly.

---

## Figure inventory (DDM ↔ Bayesian pairing)

| Result | DDM (Method A) | **Bayesian (Method B) — report this** |
|---|---|---|
| Non-decision time | `NDT_barchart.pdf` | `NDT_barchart_bayesian.pdf` |
| Model summary | `DDM_summary.pdf` | `Bayesian_summary.pdf` |
| Per-participant saccadic t₀ | — | `Bayesian_srt_ndt.pdf` |
| Process schematics (v, a, t₀) | `ddm_*_degs.pdf` | `bayes_*_degs.pdf` |
| RT distributions / vincentiles | `vincentile_results_fig1–4` (same — model-free) | |

---

## Headline results (Bayesian)

- **HRT non-decision time decreases with target speed:** 168 → 156 → 147 ms (Friedman p ≈ 0.002);
  **0 / 48 cells floored** (the DDM floored 3 at 150 deg/s).
- **SRT non-decision time does not vary with speed** and is estimated per participant:
  ~39–81 ms (population mean ~60 ms), **0 floored** (the DDM/per-cell floored ~50%).
- **Dissociation:** the hand system tunes its motor-preparation time to task demand; the saccadic
  system runs at a roughly fixed latency.
