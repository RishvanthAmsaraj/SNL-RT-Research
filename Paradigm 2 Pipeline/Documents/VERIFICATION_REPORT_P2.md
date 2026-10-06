# Paradigm 2 — Verification Report

**2026-10-03 · scope: repo commit `4b273f6` ("Add Paradigm 2 (CIR) pipeline"), `P2_Technical_Breakdown.pdf`, `P2_Results_and_P1_Comparison.pdf`**

## 1. Verdict

The Paradigm 2 results are correct and reproducible, and after the corrections in §4 the code tables, the technical breakdown, the results document, the README and the CHANGELOG all say the same thing. One finding changes in substance: the claim that "the Paradigm 1 speed effect does not replicate" is replaced by a more precise one — Paradigm 1's effect lives in its stationary → moving step, and between moving speeds hand t₀ is flat in **both** paradigms (§3).

## 2. What was checked

| # | Check | How | Outcome |
|---|---|---|---|
| 1 | Repo matches what was delivered | byte comparison of every file in `Paradigm 2 Pipeline/` | identical; the only addition is `P2_Results_and_P1_Comparison.md`. No `Current Pipeline/` (Paradigm 1) file was touched by the commit |
| 2 | Clean rerun from the repo | fresh copy, `python run_all_P2.py --skip-bayes-fits` | exit 0 (735 s on one core). All 12 regenerated tables — Method A, SRT diagnostics, supplementary — are **byte-identical** to the committed ones, and the regenerated breakdown is byte-identical to the committed breakdown (after fix C1) |
| 3 | Bayesian reproducibility | second full run of `Bayesian_SRT_ndt.py`; Paradigm 1 parity | ndt tables byte-identical; Paradigm 1 hand model r = 0.9995 (largest cell difference 1 ms). The hand and per-speed saccade Bayesian fits were not rerun a second time (~25 min on one core); their code path is covered by the parity check |
| 4 | Environment parity with Paradigm 1 | original Paradigm 1 scripts on Paradigm 1 data | Method A exact (48/48 + 48/48 cells, identical model choices); Method B to Monte Carlo error |
| 5 | Code adjusted for Paradigm 2 | unified diff (`PORT_DIFF_P1_to_P2.diff`) + search for Paradigm 1 leftovers | only speeds, block type, input file, palette, four-speed layouts and computed figure text change. No stationary-condition code remains: no `[0, 75, 150]`, `piv[0]`, colour entry for 0 deg/s, 3-panel layout, block type `"I"` or Paradigm 1 input name. The speed battery's slowest-vs-fastest contrast is 75 vs 150 (Paradigm 1: 0 vs 150) |
| 6 | Every Paradigm 1 figure exists for Paradigm 2 | figure names compared with speeds normalised | complete: 44 Paradigm 1 files → 54 Paradigm 2 files. Same families — DDM and Bayesian schematics per measure per speed (8 each instead of 6), DDM and Bayesian summaries, saccadic t₀ forest plot, both NDT charts, identifiability, fixed-t₀ sensitivity, floor mechanism, 4 vincentile figures — plus `HRT_floor_control`. New: `Figures/Comparison/` (3 figures) |
| 7 | Documents vs tables | every number in both PDFs traced to a table | the technical breakdown is generated from the tables and matches; the results document had the errors in §4A |
| 8 | Cross-paradigm comparability | same pipeline version, windows, filters and fitters on both; Paradigm 1 floor sweep rerun with the Paradigm 2 code | comparable, with one standing limit: different participants, so every comparison is between cohorts |

## 3. The reframed central finding

Within Paradigm 1 (`Code/Comparison/P1_vs_P2_summary.csv`; per-participant change, mean [95% bootstrap CI], Wilcoxon p):

| | P1: 0 → 75 | P1: 75 → 150 | P2: 75 → 150 |
|---|---|---|---|
| Hand t₀ (Method B) | −11.5 [−19.3, −5.1], 0.005 | −10.1 [−20.1, +1.0], 0.066 | +3.9 [+0.8, +6.9], 0.062 |
| Decision time a/v | +1.1, 0.60 | +17.0 [+5.0, +28.3], 0.013 | −6.3 [−10.4, −2.1], 0.013 |
| Raw median HRT | −10.2, < 0.001 | +3.2, 0.29 | −2.8, 0.059 |
| Raw 10th-percentile HRT | −9.6, < 0.001 | −2.5, 0.13 | +0.8, 0.47 |

Between moving speeds, both paradigms' raw RTs are flat and each paradigm's t₀ shift is cancelled by an opposite decision-time shift (r = −0.80 in both). The step that moves the raw distribution — median and leading edge — is Paradigm 1's stationary → moving step, which Paradigm 2 has no condition to test. Figure: `Figures/Comparison/P1_vs_P2_hand.png`.

## 4. Corrections

**A. `P2_Results_and_P1_Comparison.md` (revised in place)**

1. *"16 (CMT001–CMT017, no CMT004/CMT013)"* — CMT004 is in the data. The 16 are CMT001–CMT010, CMT0011, CMT0012, CMT0014–CMT0017; only participant 13 is absent. (IDs from 11 up carry an extra 0 — worth normalising in the extraction.)
2. *"Trials kept ~7,676 rows"* — 7,676 is every row in the file, including 1,916 saccade-only (`S`) trials the pipeline never uses. Paradigm 1 analysed 5,760 interception trials: 5,690 hand and 5,164 saccade after the RT windows.
3. *"passed a byte-level parity check"* — the Paradigm 1 parity is exact for Method A at stored precision and within Monte Carlo error for Method B. Byte-for-byte holds for the Paradigm 2 reruns (§2, rows 2–3).
4. *"Adding the 5% contamination term would remove that dependence"* — the participant-level saccade model already has the 5% term. What would remove the dependence on the single fastest saccade is letting that term account for saccades faster than t₀.
5. *"identifiability structure (hand clean, saccade floor-bound)"* — in Paradigm 2 saccadic t₀ is held by the floor **or** the fastest-saccade ceiling.
6. *"The Paradigm 1 dissociation did not replicate"* — the identifiability dissociation (hand t₀ identified, saccadic t₀ not) did replicate. What changes is the interpretation of the hand speed effect (§3).
7. *Central finding ("the Paradigm 1 speed effect does NOT replicate … reframes Paradigm 1's central claim")* — restated as in §3: Paradigm 1's 75 → 150 step was not reliable on its own; its effect is stationary vs moving.
8. *"Paradigm 2 effectively measures a lead-dominant regime"* — conditional on the v0_1_36 measurement definitions, which are still unconfirmed.

**B. `P2_Technical_Breakdown.md` (regenerated by the updated `make_breakdown.py`)** — errors in my earlier version:

1. *"saccades never hit that bound in Paradigm 1"* was false. At the production floor, 4/32 Paradigm 1 saccade cells sit at their fastest saccade (Paradigm 2: 10/52); in the floor sweep, 1/32 is stuck there at every floor (Paradigm 2: 7/52).
2. The Paradigm 1 floor-sweep reference (median 0.975, 9/17) came from a different fitter (profile likelihood, `v3_mode_and_control.py`) without the ceiling exclusion. Replaced by the Paradigm 2 code run on Paradigm 1 data: eye 9/16 fits follow the floor, hand 5/43 — versus Paradigm 2's 19/38 and 8/60. Compare these shares, not the medians: the eye slopes form two clusters (near 0 and near 1), so the median lands in the gap when the split is close to even (0.99 vs 0.70 looks like a difference; 56% vs 50% is not, Fisher p = 0.77).
3. Bottom line 4, §4 and §8 restated as in §3.
4. Panel D of `P1_vs_P2_saccade.png` (and the legend of `HRT_floor_control.png`) summarised the floor sweep by its median, which made the two paradigms' eyes look different. Both now report the share of fits that follow the floor. The "weaker per-cell flooring in Paradigm 2" statement is also withdrawn (48% vs 66% of cells on the floor, Fisher p = 0.18); the participant-level ceiling (6/15 vs 0/14, p = 0.017) is the difference that holds.

**C. Code**

1. `run_all_P2.py --skip-bayes-fits` did not copy the Bayesian run logs into its work folder, so the regenerated breakdown silently dropped the saccadic t₀ convergence numbers. It now copies them; with the fix, the clean rerun reproduces the committed breakdown byte-for-byte.

**D. Repo root**

1. README: Paradigm 1 "7,676 trials" → 5,760 interception trials (the 7,676 rows include saccade-only trials); the headline sentence restated as in §3.
2. CHANGELOG: new `[2.3.1]` entry records the restatement; `[2.3.0]` is left as history.

## 5. Not verified, or outside Paradigm 2

1. The README's Paradigm 1 key-findings table gives hand t₀ intervals of 154–182, 139–174 and 130–162 ms. The mean per-cell 95% intervals in Paradigm 1's `Bayesian_hrt_fits.csv` are 153.2–181.8, 145.1–168.4 and 137.6–157.6 ms. The first matches; the source of the other two is not in the pipeline. Worth tracing separately.
2. Figures were regenerated from identical tables, but PDFs embed timestamps, so they cannot be compared byte-for-byte.
3. The aiming results depend on measurement definitions that only the extraction's author can confirm (`HandDir_deg`, `TargetDir_deg`, the `_HRT50` variants).

## 6. Pre-publication checks (added after the review above)

| Check | Result |
|---|---|
| Likelihood | The Method A Wald density equals scipy's inverse Gaussian (mean a/v, shape a²) to 8e-16 relative error. The three Bayesian scripts use the log of the same density (`log a − ½ log 2π − 1.5 log τ − (a − vτ)²/2τ`), with log-sum-exp for the mixture and contamination terms. |
| Method A optima | An independent negative log-likelihood (scipy densities, Nelder–Mead, 25 random starts) on 20 random cells never beat the pipeline's optimum by more than 0.03 log-likelihood units (stored-rounding level). |
| Counts | Recomputed from the 16 raw CIR files, independently of the builder: 10,239 trials, 10,158 hand and 10,017 saccade RTs kept, 16 participants. |
| Table consistency | Every Bayesian interval is ordered (lower ≤ estimate ≤ upper) and lies between the floor and the fastest RT; every two-component cell has its fast component below its slow one. |
| Model fit (posterior predictive) | Plug-in posterior means reproduce the observed 10/30/50/70/90% quantiles within 2.4/1.6/1.7/2.8/5.4 ms (hand) and 4.9/4.4/3.3/3.8/7.5 ms (saccade) on average; the predicted spread is slightly too wide in both tails (10% quantile 2–4 ms early, 90% quantile 3–5 ms late). |
| Bayesian recovery (`Validation/bayesian_recovery_P2.py`) | One simulated dataset with Paradigm 2's design and a true 10 ms hand t₀ drop from 75 to 150 deg/s: detected (15/16, p = 0.003) but shrunk to −5.9 ms [−8.2, −3.4]; −4.3 ms moved into decision time (truth 0, p = 0.013); t₀/decision-time changes correlate at r = −0.75 with no true variation; absolute t₀ +6.0 ms high (RMSE 7.2 ms); per-cell 95% coverage 70%; raw fast end −9.9 ms. Decision time from posterior-mean a and v differs from the per-draw mean by ≤ 0.16 ms. |
| Figures | Every figure inspected after its last change. Fixed in this pass: panel C of `P1_vs_P2_saccade` now uses the Bayesian fits (interval reaches the floor / the fastest RT / both / neither); panel D reports the share of fits that follow the floor instead of the median; panel C of `P1_vs_P2_hand` says the anticorrelation arises from noise. |
| Claims ledger (`Validation/verify_claims.py`) | Recomputes the documents' key numbers from the tables and raw trials and checks each appears where it is quoted; see `verify_claims_report.csv`. |

**What the recovery test changes in the interpretation.** (1) Paradigm 2's flat hand t₀ is informative: its design detects a 10 ms moving-speed change, and its raw fast end rules out a 10 ms drop. (2) Decision-time differences between speeds are not evidence by themselves — the model produces a significant one from a pure t₀ change. (3) The r ≈ −0.8 in the real data is what estimation noise produces. (4) Absolute Bayesian hand t₀ values run about 6 ms high and per-cell intervals under-cover, so group-level comparisons are the claims to publish.

**Environment note.** `Bayesian_*_fit.py` try to save the posterior as `.nc`; without `netCDF4` or `h5netcdf` the save is skipped silently (it was skipped in this environment, so no `.nc` files exist here).

## 7. Files changed in this update

- `Paradigm 2 Pipeline/Documents/` — `P2_Results_and_P1_Comparison.md` (revised), `P2_Technical_Breakdown.md` (regenerated), `RUN_GUIDE_P2.md` (comparison step), `VERIFICATION_REPORT_P2.md` (new)
- `Paradigm 2 Pipeline/Code/` — `run_all_P2.py` (C1), `Validation/make_breakdown.py` (B1–B4), `Validation/bayesian_recovery_P2.py` + its summary/cell tables, `Validation/verify_claims.py` + `verify_claims_report.csv`, `Comparison/` (`P1_vs_P2_comparison.py`, `P1_vs_P2_basic_charts.py`, `P1_vs_P2_NDT_bayesian_overlay.py`, `P1_vs_P2_summary.csv`, `P1_floor_sweep_rerun.csv`, `P1_SRT_identifiability_rerun.csv`)
- `Paradigm 2 Pipeline/Documents/FIGURE_GUIDE.md` (new) — every figure explained and placed in the bigger picture
- `Paradigm 2 Pipeline/Figures/Comparison/` (new: hand, saccade/identifiability, distributions; PDF + PNG)
- `README.md`, `CHANGELOG.md` (D1–D2)

Everything else in `Paradigm 2 Pipeline/` is unchanged from commit `4b273f6`.
