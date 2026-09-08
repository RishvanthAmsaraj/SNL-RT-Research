# Validation checklist — running everything, what to check, what to compare against

Companion to `CODEBASE_AUDIT.md`. Work top to bottom. Nothing in Phase 0–1 modifies
your repo; every script only reads your CSVs and writes new `v*_*.csv` files.

Item codes (A1, E1, …) refer back to the audit.

---

## Phase 0 — Freeze the baseline

Do this before running anything. You cannot prove a change was safe without a
before.

- [ ] **0.1 Tag the current state**
  ```bash
  cd SNL-RT-Research-main
  git add -A && git commit -m "baseline before audit fixes"
  git tag baseline-pre-audit
  ```

- [ ] **0.2 Snapshot the ground-truth CSVs**
  ```bash
  mkdir -p baseline_outputs
  cp "Current Pipeline/Code/DDM/DDM_hrt_fits.csv"        baseline_outputs/
  cp "Current Pipeline/Code/DDM/DDM_srt_fits.csv"        baseline_outputs/
  cp "Current Pipeline/Code/Bayesian/Bayesian_hrt_fits.csv" baseline_outputs/
  cp "Current Pipeline/Code/Bayesian/Bayesian_srt_fits.csv" baseline_outputs/
  cp "Current Pipeline/Code/Bayesian/Bayesian_srt_ndt.csv"  baseline_outputs/
  cp "Current Pipeline/Code/LATER Model/LATER_fits.csv"     baseline_outputs/
  ```

- [ ] **0.3 Capture the environment**
  ```bash
  conda list --explicit > baseline_outputs/env_conda.txt   # or: pip freeze > ...
  python -c "import numpy,scipy,pandas;print(numpy.__version__,scipy.__version__,pandas.__version__)"
  ```
  *Why:* SciPy changed `differential_evolution` internals across versions. If you
  ever need to re-derive the baseline, you need the version that produced it.

- [ ] **0.4 Confirm the existing parity suite is green**
  ```bash
  cd kinarm-rt-app && python -m pytest tests/test_parity.py -q
  ```
  **Pass:** all 27+ tests pass. If any already fail, fix that first — you cannot
  distinguish new breakage from old.

- [ ] **0.5 Record the headline numbers you must not silently change**

  | Quantity | Current value | Source |
  |---|---|---|
  | HRT group t₀ by speed | 169.5 / 158.0 / 147.9 ms | `Bayesian_hrt_fits.csv` |
  | HRT cells floored | 0 / 48 | `Bayesian_hrt_ndt.csv` |
  | SRT per-participant t₀ | 70–71 ms, all | `Bayesian_srt_ndt.csv` |
  | LATER median reciprobit r² | 0.971 | `LATER_fits.csv` |
  | Hand skew/CV | 12.9 | `why_saccadic_t0_floors.py` |
  | Mixture cells selected | 16 | `DDM_srt_fits.csv` |
  | Max R-hat | 1.011 | `Bayesian_hrt_fits.csv` |

  Fill in any I got wrong from your files. Every later phase gets checked against
  this table.

---

## Phase 1 — Run the diagnostics (read-only, nothing changes)

- [ ] **1.1 Stage the validation folder**
  ```bash
  cp -r validation/ "Current Pipeline/Code/DDM/"
  cd "Current Pipeline/Code/DDM/validation"
  cp ../pooled_data.csv ../DDM_hrt_fits.csv ../DDM_srt_fits.csv .
  cp ../../Bayesian/Bayesian_hrt_fits.csv ../../Bayesian/Bayesian_srt_fits.csv .
  ```
  Everything must sit in one folder — same convention as your scripts.

- [ ] **1.2 Run everything**
  ```bash
  python run_all.py --quick     # first look, ~2 min
  python run_all.py             # report-quality, ~10 min (B=200)
  ```
  Only needs numpy/scipy/pandas. No PyMC.

### 1.3 — `v1_parity_check.py` — can you trust the fast fitter? **(gates E1)**

**Compare against:** `DDM_hrt_fits.csv`, `DDM_srt_fits.csv`.

Read the NLL column, not the parameters. The profile fitter is exact given t₀, so
it can only match or beat differential evolution.

| Verdict | Meaning | Action |
|---|---|---|
| `PASS` | params and NLL agree | none |
| `FLOOR` | agrees, t₀ pinned at floor | expected for most SRT cells |
| `FLAT` | NLL agrees, params moved | **inspect** — flat ridge means weak identifiability, a finding |
| `CAP` | v or a at a bound | expected to differ; count these, they feed A7 |
| `FAIL` | profile did *worse* | **stop** — should be impossible, investigate before proceeding |

- [ ] **Gate: zero `FAIL` cells.** If any appear, do not adopt the profile fitter.
- [ ] `max |dt0|` among PASS/FLOOR under **1.0 ms** (your CSVs round t₀ to whole ms).
- [ ] Note the `CAP` count. Expect roughly a fifth of SRT cells. That number goes in
      the paper as an `at_bound` column (A7).
- [ ] Check "profile beat DE by > 0.5 log-lik". Small gaps are CSV rounding and are
      meaningless. Cells beyond 0.5 are places DE genuinely undershot — list them.

### 1.4 — `v2_ks_calibration.py` — which fits actually fail? **(A1)**

**Compare against:** your `ks` / `ks_single` columns and the 0.10 threshold.

- [ ] `max |ks_new − ks_pub|` **< 0.005**. If not, that is a v1 parity problem, not a
      calibration one — go back to 1.3.
- [ ] Median `crit95` should land near **0.08–0.10**, well below the naive 0.13. If it
      comes out near 0.13 the refit inside the bootstrap is broken.
- [ ] Read the "REJECTED but reported as model='single'" list. Expected from the
      published table: **CMT0012@0, CMT009@0, CMT0012@75, CMT009@150**, marginally
      CMT0014@0 and CMT0011@150. If your run names roughly these, the finding
      replicates.
- [ ] Check the median SRT p-value. **If it is comfortably above 0.05, soften the
      "SRT fits remain imperfect" framing** — KS 0.08–0.09 is near the null 90th
      percentile, not a failure.
- [ ] HRT should be clean. Any HRT rejection deserves a direct look at that cell.

*Runtime note:* ~8 min at B=200 for 96 cells. Use `--B 200` minimum for anything you
report — at B=50 the smallest possible p-value is 0.02.

### 1.5 — `v3_mode_and_control.py` — two independent checks

**Check A — `express_mode` (A6). Compare against:** the `express_mode` / `reg_mode`
columns in `DDM_srt_fits.csv`.

- [ ] Read `max express mean-mode gap`. Expect ~27 ms on CMT008@75.
- [ ] `30 ms separation rule flips` — if **> 0**, mixture selection itself changes and
      this is no longer cosmetic.
- [ ] `'<130 ms express' flips` — if **> 0**, your express-cell count changes.
- [ ] Decision: gap < 5 ms and no flips → just rename the columns. Otherwise pick
      mode or mean and make the **column name, the separation rule, and the <130 ms
      test all use the same one**.

**Check B — the hand control (A11). Compare against:** `SRT_identifiability.pdf`,
which currently has no hand panel.

- [ ] HRT median slope near **0**, tracking count near **0/48**.
- [ ] SRT median slope near **0.9**, most cells tracking.
- [ ] Script prints `CONTROL HOLDS`. If it does not, **do not make any hand-vs-eye
      identifiability claim** until you understand why.
- [ ] Add the HRT panel to `SRT_identifiability_check.py`. This is the cheapest
      high-value change in the whole audit.

### 1.6 — `v4_intervals_and_AvB.py` — uncertainty for Method A **(A10)**

**Compare against:** `Bayesian_hrt_fits.csv` `t0_lo95` / `t0_hi95`.

- [ ] Median hand per-cell CI width. Expect **~40–60 ms**, versus Method B's ~27 ms.
      That gap *is* the value of partial pooling — state it in the paper rather than
      leaving the hierarchical model unjustified.
- [ ] `CI spans the floor` will be high, **including for hand cells**. This is normal
      and is a statement about single-cell frequentist estimation, not about your
      group result. Do not let it spook you into rewriting the flooring narrative.
- [ ] Method A vs B correlation **r > 0.9** for hand.
- [ ] Mean |A − B| **< 15 ms**. If larger, that is mostly the contamination mismatch
      (A2/A6) — re-run this after Phase 3.1 and compare the two runs.

---

## Phase 2 — Safe fixes (reported numbers must NOT change)

Each of these should leave every number in your 0.5 table identical. If one moves,
you have found a second bug.

- [ ] **2.1 Rename or recompute `express_mode`** (A6). Per the 1.5 decision.
      **Verify:** if renaming only, `DDM_srt_fits.csv` values byte-identical apart
      from the header.

- [ ] **2.2 Add `at_bound` and `t0_at_floor` columns** to both DDM fit tables (A7).
      **Verify:** existing columns unchanged; new columns only.

- [ ] **2.3 Add `ks_p` and `crit95` columns** from `v2_ks_calibration.csv` (A1).
      Update `DDM_figures.py` panels A and B to draw the **per-cell** critical value
      instead of the flat 0.10 / 0.12 lines.
      **Verify:** the `ks` column itself is unchanged.

- [ ] **2.4 Add `t0_lo95` / `t0_hi95`** from `v4_profile_intervals.csv` (A10).
      **Verify:** point estimates unchanged.

- [ ] **2.5 Add the HRT panel to `SRT_identifiability_check.py`** (A11).
      **Verify:** the existing SRT panel is pixel-identical.

- [ ] **2.6 Centralise the figure style** (S2). One `figstyle.py` with the palette,
      Arial detection, and rcParams; import it everywhere. Currently four different
      0 deg/s colours across `DDM_figures.py`, `LATER_analysis.py`,
      `vincentile_figures.py`, and `_speeds.py`.
      **Verify:** regenerate every figure and diff visually. Colours *will* change —
      that is the point — but no data should move.

- [ ] **2.7 Replace blanket `warnings.filterwarnings("ignore")`** with specific
      categories (A12). **Verify:** re-run `DDM_fit.py`; note any overflow warnings
      that now surface, especially on the `CAP` cells from 1.3.

- [ ] **2.8 Fix `goodness_of_fit()` dropping sub-t₀ trials** in
      `kinarm_rt/diagnostics.py` (A12). Count them as failures, not omissions.
      **Verify:** app-only change; re-run `tests/test_parity.py`.

- [ ] **2.9 Unify the speed column** — `LATER_analysis.py` uses `SpeedCode`,
      everything else uses `Speed_deg_per_s` (A12).
      **Verify:** `LATER_fits.csv` byte-identical. If it is not, those two columns
      disagree in your data and you have a bigger problem.

**Gate for Phase 2:** re-run `pytest tests/test_parity.py` and confirm your 0.5
table is unchanged.

---

## Phase 3 — Fixes that DO change reported numbers

Each needs a documented before/after. Do them **one at a time**, re-running the full
pipeline between each. Never batch two.

- [ ] **3.1 Align contamination between methods** (A2). Add the uniform term to the
      Bayesian likelihood, set `contamination = 0.05` in both.
      **Compare:** new vs baseline `Bayesian_hrt_fits.csv`. Expect small shifts.
      **Red flag:** any HRT cell newly flooring, or max R-hat rising above 1.01.
      Keep the pure-Wald run as the documented sensitivity check.

- [ ] **3.2 Replace the `min(RT)` t₀ ceiling** (A3). Either drop it (contamination now
      absorbs sub-t₀ trials) or switch to the 3rd percentile to match `DDM_fit.py`.
      **Compare:** per-cell t₀ against baseline. **Expect the largest movement in
      cells with an unusually fast single trial** — find those first
      (`min(RT)` far below the 3rd percentile) and check them individually.

- [ ] **3.3 Swap in the profile fitter** (E1), gated on 1.3 passing.
      **Compare:** `v1_parity_results.csv` is the evidence. Re-run
      `SRT_identifiability_check.py` and `SRT_fixed_t0_analysis.py` with it and
      confirm the conclusions (not necessarily the third decimal) are unchanged.

- [ ] **3.4 Bootstrap LRT for mixture selection** (A1). Replaces the KS threshold +
      `0.10 ≤ π ≤ 0.90` + 30 ms separation heuristics.
      **Compare:** the set of 16 mixture cells. **If the set changes, `Bayesian_SRT_fit.py`
      must be re-run** — it reads `model == "mixture"` from `DDM_srt_fits.csv`.
      Report both sets in the supplement.

- [ ] **3.5 Unify the mixture floor at 70 ms** (A8) and add the ordering constraint.
      **Compare:** `pi`, `express_mode`, `reg_mode` per mixture cell. Ordering should
      *tighten* the posteriors — if `pi` intervals get wider, label switching was
      worse than assumed.

- [ ] **3.6 Truncation correction** (A12). One line: divide by `F(hi−t₀) − F(lo−t₀)`.
      **Expect nearly nothing** — verified as < 0.05 ms on typical cells. Check the
      low-`v`/high-`a` cells specifically (CMT008@75, CMT008@150), where it moved t₀
      ~8 ms in simulation. Report as a one-sentence robustness result either way.

- [ ] **3.7 LATER by MLE, and lead with a calibrated statistic** (A9).
      `μ̂ = mean(1/RT)`, `σ̂ = sd(1/RT)`; report Shapiro–Wilk or a bootstrap KS
      p-value instead of the reciprobit r².
      **Compare:** MLE vs the OLS-on-plotting-positions fit. Expect close agreement
      on parameters. The r² does not change — you are demoting it from evidence to
      decoration.

---

## Phase 4 — Structural / model changes

Bigger, and each is a real methodological decision. Sequence them last.

- [ ] **4.1 Reparameterize (v, a) → (μ, λ)** (E2). Pure reparameterization, so the
      likelihood is unchanged.
      **Verify:** transform the posterior back to (v, a) and confirm it matches
      baseline within MCMC error. **Then check:** leapfrog steps per draw, divergence
      count, and how many cells still sit at a cap. All three should improve.

- [ ] **4.2 Test the numpyro backend** (E3). `pm.sample(nuts_sampler="numpyro",
      chain_method="vectorized")`.
      **Check first:** does `pip install pymc numpyro "jax[cpu]"` work on Windows
      **without** a C++ compiler? That is the whole prize — it would let the desktop
      build drop conda-pack.
      **Verify:** posteriors match the default sampler within MCMC error; time both.
      **This is unverified.** It may fail on a PyTensor/JAX incompatibility. Timebox it.

- [ ] **4.3 Promote the per-speed hierarchical model** (A4). `hierarchical.py` already
      exists — move it from "advanced" to primary for the hand.
      **Compare:** group t₀ per speed. **The new speed effect should be LARGER than
      169.5/158.0/147.9**, because the current pooled model shrinks toward the grand
      mean across speeds. If it comes out smaller, something is wrong.

- [ ] **4.4 Joint hand+eye model with an effector × speed interaction** (A5/N2).
      Replaces Friedman + bootstrap + permutation with one posterior contrast.
      **Compare:** the interaction credible interval against the current
      "hand p ≈ 0.005, eye p ≈ 0.74". They should agree in direction. If the
      interaction CI includes zero, the current two-test framing was overstating —
      better to know now than at review.

- [ ] **4.5 Extract a shared `snlrt` package** (S1). Four copies of `wald_pdf` today.
      **Verify:** every script produces byte-identical output afterwards; rewrite
      `test_parity.py` to test one implementation instead of asserting four agree.

- [ ] **4.6 Posterior predictive KS** (N4), **SBC** (N5), **trial-order covariate** (N6),
      **hierarchical LATER** (N3). Additive; no existing number changes.

---

## Reference — what compares to what

| Check | New output | Ground truth |
|---|---|---|
| Profile fitter | `v1_parity_results.csv` | `DDM_{hrt,srt}_fits.csv` |
| Calibrated GoF | `v2_ks_calibration.csv` | `ks` / `ks_single` columns |
| Mode vs mean | `v3_mode_vs_mean.csv` | `express_mode` / `reg_mode` |
| Hand control | `v3_floor_control.csv` | `SRT_identifiability.pdf` (SRT panel only) |
| Method A CIs | `v4_profile_intervals.csv` | `Bayesian_*_fits.csv` `t0_lo95/hi95` |
| Any Bayesian change | new `Bayesian_*_fits.csv` | `baseline_outputs/` |
| App vs scripts | `pytest tests/test_parity.py` | the scripts |

## Reference — acceptance thresholds

| Quantity | Accept | Investigate |
|---|---|---|
| Profile vs DE, NLL | `nll_new ≤ nll_pub + 1e-6` | any `FAIL` |
| Profile vs DE, t₀ | ≤ 1.0 ms | > 1.0 ms on a non-CAP cell |
| Profile vs DE, v / a | ≤ 2% relative | > 2% with matching NLL → flat ridge |
| KS reproduction | \|Δks\| < 0.005 | ≥ 0.005 |
| Bootstrap replicates | B ≥ 200 for reporting | B < 100 caps p at 1/(B+1) |
| Max R-hat | < 1.01 | ≥ 1.01 |
| Divergences | 0 | any |
| A vs B t₀ correlation | r > 0.9 | ≤ 0.9 |
| A vs B mean \|diff\| | < 15 ms | ≥ 15 ms |

## Reference — things that will look alarming but are fine

- **Most cells show `CI spans the floor` in v4, including hand cells.** Per-cell
  frequentist t₀ is genuinely weakly identified. The hierarchical model is what
  fixes it; this is the evidence that it earns its place.
- **Most SRT cells come back `FLOOR` in v1.** t₀ pinned at 70 ms is the published
  result, not a fitting failure.
- **`profile beat DE` on nearly every cell by a tiny margin.** Your CSVs store
  rounded parameters, so re-evaluating the NLL there is always fractionally worse.
  Only gaps beyond ~0.5 log-likelihood units mean anything.
- **Figure colours change after 2.6.** Intended — you currently have four schemes.

## Reference — things that should stop you

- Any `FAIL` in v1 → do not adopt the profile fitter.
- `max |ks_new − ks_pub| ≥ 0.005` → parity problem upstream; fix before reading v2.
- v3 Check B prints anything other than `CONTROL HOLDS` → make no hand-vs-eye
  identifiability claim until resolved.
- Max R-hat ≥ 1.01 or divergences > 0 after any Phase 3/4 change → the model got
  harder to sample; diagnose before trusting the numbers.
- An HRT cell newly flooring after 3.1 or 3.2 → 0/48 is a headline result. Any
  change there needs explaining, not absorbing.

---

## Minimum viable path

If you only have an afternoon, do these five and stop:

1. `python run_all.py` — 10 minutes, changes nothing, tells you which findings replicate.
2. **2.5 / 1.5B** — add the hand control panel. Ten lines, biggest argumentative gain.
3. **2.1** — fix the `express_mode` naming. It is currently wrong in a shipped CSV.
4. **2.3** — add `ks_p` to the fit tables and drop the flat 0.10 line from the figures.
5. **2.4** — add profile CIs to the Method A tables.

All five are Phase 2, so none of your reported numbers move.
