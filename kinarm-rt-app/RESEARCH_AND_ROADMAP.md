# Code review and improvement roadmap

Now that I have the actual repository, this is a real review of the code rather
than a clean-room reconstruction. It covers what I found reading the scripts,
where the app already helps, and the model, prior, and robustness changes worth
making — separated into what aligns with your own `ISSUES_AND_IMPROVEMENTS.md`
and what is genuinely new.

---

## 1. The app now reproduces the real pipeline

I rebuilt the engine to match the repository exactly, not approximately:

- The shifted-Wald likelihood is the single-barrier form used in your scripts,
  `log a − ½·log 2π − 1.5·log τ − (a − vτ)²/2τ`, parameterised directly in drift
  *v* and boundary *a*.
- Priors, bounds, and floors come straight from `CODE_REFERENCE.md`: `mu_lv ~
  N(log 10, 0.5)`, `mu_la ~ N(log 1, 0.5)`, `mu_z ~ N(0.5, 1)` for the hand and
  `N(0, 1)` for saccades; drift cap 20; hand *t₀* floor 130 ms; saccade floor 70
  ms; hand filter 150–800 ms; saccade filter 80–600 ms.
- Hand: one hierarchical fit pooled across all participant × speed units.
- Saccades: per speed, a pooled hierarchical single-Wald for unimodal cells plus
  the two-component express/regular mixture for bimodal cells, flagged by
  Hartigan's dip test — the same structure as `Bayesian_SRT_fit.py`.
- LATER: the reciprobit line fit to the central 10–90% of the distribution, per
  participant × speed, matching `LATER_analysis.py`.

On the built-in synthetic data this returns the right picture: hand *t₀* around
175/168/157 ms (identified, decreasing with speed), saccadic *t₀* pinned at ~70
ms, express/regular mixtures recovered with express modes near 100–120 ms.

**Validated against the real `pooled_data.csv`.** Running the app's hand Bayesian
fit on your actual data reproduces `Bayesian_hrt_fits.csv` almost exactly: group
*v*, *a*, and *t₀* match to two decimals (9.39/9.61/8.36; 0.77/0.78/0.80;
169.5/158.0/147.9 ms), and the per-cell *t₀* estimates correlate with your
published values at r = 0.999 with a mean absolute difference of 0.4 ms. LATER
returns a median reciprobit r² of 0.971 (your reported 0.97) with median
latencies 171/177/181 ms (matching your report), and the hand skew/CV is 12.9, as
reported. The reimplementation is faithful, not approximate.

Two corrections to my earlier notes, now that I can see the code: the
express/regular **mixture and the LATER model already exist** in your pipeline. I
had proposed them as additions; they are already there, so they are not on the
list below.

---

## 2. Findings from reading the code

These are specific, mostly small, and each is actionable.

- **Contamination is documented but not in the Bayesian likelihood.**
  `CODE_REFERENCE.md` describes the full likelihood as 0.95·Wald + 0.05·Uniform,
  and `DDM_fit.py` implements exactly that (the `contam/Tr` term). But
  `Bayesian_HRT_fit.py` and the `hierarchical_single` model in
  `Bayesian_SRT_fit.py` use a **pure Wald** `pm.Potential` with no uniform
  component. So the contamination is in Method A (frequentist) but not Method B
  (Bayesian). Either the doc should be scoped to the frequentist fit, or the
  uniform term should be added to the Bayesian likelihood. The app exposes
  contamination as an optional slider (default 0, matching your Bayesian code) so
  you can see the effect either way.

- **The hand model has no per-speed group parameter.** `Bayesian_HRT_fit.py`
  pools every participant × speed unit under one global prior, so the speed effect
  is read off the per-unit posterior means rather than modelled. That works, but a
  per-speed hierarchical structure (speed as a modelled factor, with a group mean
  per speed) would give you group-level *v*, *a*, *t₀* per speed **with credible
  intervals** and more power to test the speed trend. This is the single change I
  would prioritise for the hand analysis.

- **The Bayesian *t₀* upper bound is `min(RT)`, which is noisy.** Both hierarchical
  models set `t0 = floor + (min_RT − floor)·sigmoid(z)`, tying the ceiling to a
  single fastest trial per unit. The frequentist fit already uses a more robust
  cap, the 3rd percentile − 2 ms. Using that same robust quantile in the Bayesian
  models, and letting a uniform contaminant absorb the few sub-*t₀* trials, would
  reduce sensitivity to one extreme RT.

- **A couple of stale comments.** `Bayesian_HRT_fit.py` refers to a "100 ms floor"
  in two comments while `FLOOR = 0.130`. Harmless, but worth fixing so the code
  reads consistently with `CODE_REFERENCE.md`.

- **Mixture convergence uses a hand-rolled relabelled R-hat.** That is reasonable
  given label switching, but it means the mixture cells are not covered by the
  same LOO/WAIC comparison you would want (see §3), so single-vs-mixture is
  currently a structural choice rather than a scored one.

None of these are errors that would overturn the results; the flooring diagnosis
and the hand/eye dissociation are sound. They are refinements.

---

## 3. Improvements that align with your own roadmap

Your `ISSUES_AND_IMPROVEMENTS.md` already lists these; here is where the app helps
or how I would implement them.

- **Dockerfile for a reproducible environment (your high-priority item).** Done —
  there is a `Dockerfile` and an `environment.yml` in this app. `docker build`
  then `docker run` gives an identical setup on macOS, Windows, and Linux, using
  the conda-forge PyMC build so there is no compiler step. This also removes the
  conda/pip friction you flagged as limitation #3.

- **LOO-CV / WAIC (your medium-priority item, and limitation #5) — now in the app.**
  The pooled fit uses a `pm.Potential`, which ArviZ cannot decompose, so the
  "Model comparison" tab refits with an equivalent `pm.CustomDist` likelihood that
  records the pointwise log-likelihood and runs `arviz.compare`. It currently
  compares estimated vs fixed *t₀* (on the example data the hand prefers estimated
  *t₀*, weight ~0.83), and the same machinery extends to single vs mixture per
  saccade cell.

- **Bootstrap / permutation test for the dissociation (your medium-priority item)
  — now in the app.** The "Advanced analyses" tab runs a participant-resampling
  bootstrap and a within-participant permutation test alongside Friedman. On the
  example data the hand shows a −20 ms speed effect (95% CI excludes zero,
  permutation p ≈ 0.005) while the eye does not (p ≈ 0.74) — the dissociation,
  now triangulated by three tests.

- **Sensitivity analysis on the express-saccade mixture threshold (your item) —
  now in the app.** The "Advanced analyses" tab sweeps the dip-test significance
  level and reports how the bimodal-cell count changes, so you can show the
  mixture assignment is stable.

- **Full-draw run (1500/1500/4).** The app's "Thorough" preset matches this
  exactly, so a final run is a menu choice rather than an edit.

---

## 4. Genuinely new suggestions (beyond your current list)

- **Correlated participant effects (LKJ) — now in the app.** The "Model
  comparison" tab can model the participant *v*, *a*, and *t₀* offsets as
  multivariate normal with an LKJ prior instead of independently. On your real
  hand data this reveals structure the independent model cannot represent — the
  boundary and non-decision-time offsets correlate about −0.57 (participants who
  set a higher boundary tend to have a shorter non-decision time), with clean
  convergence.

- **Per-speed hierarchical structure — now in the app.** The same tab fits a model
  that treats speed as a factor with participant random effects, giving
  group-level *v*, *a*, and *t₀* per speed **with credible intervals** rather than
  post-hoc means. On your real hand data it returns t₀ = 168 [161, 174] / 152
  [148, 157] / 142 [137, 147] ms — the speed effect with clearly separated
  intervals.

- **Empirical or literature-informed priors (see §5).** Your priors are sensible
  and weakly informative; anchoring drift and boundary to published diffusion
  fits, or to your own first-pass posterior, would sharpen them without being
  circular if done as a documented two-stage procedure.

- **A parameter-recovery study — now in the app.** The "Advanced analyses" tab
  simulates from known parameters, refits, and reports recovery. On the example
  data hand *t₀* recovers within a few ms while saccadic *t₀* (true ≈ 30 ms, below
  the floor) cannot be recovered and pins at 70 ms — the strongest possible
  support for reporting saccadic *t₀* as fixed, now shown quantitatively rather
  than asserted.

- **A frequentist Method A fit and a headless CLI — now in the app.** The
  differential-evolution MLE with contamination from `DDM_fit.py` is available in
  the "Model comparison" tab (Method A vs Method B), and `run_pipeline.py` runs the
  whole analysis from a config file with no GUI, writing repo-format CSVs, figures,
  and a report.

- **Prior and posterior predictive checks as first-class figures.** Prior
  predictive draws confirm the priors imply plausible RTs; posterior predictive
  quantile overlays per cell confirm the fit. The KS numbers already summarise the
  latter; a quantile-overlay figure would make it visual.

---

## 5. Priors, with concrete values

Your priors are defensible. If you want to sharpen them:

- **Non-decision time.** Physiology is tightest here. Hand: a prior centred near
  0.15 s with mass in 0.10–0.22 s (reach preparation, Haith et al. 2016). Saccade:
  the true non-decision time is ~20–50 ms, below the express floor, which is
  exactly why it floors; fixing it at 70 ms is the honest choice, and if you ever
  estimate it, use a tight 0.02–0.06 s prior and expect the lower bound.
- **Drift and boundary.** Rather than generic log-normal priors, anchor to
  published diffusion fits: Matzke & Wagenmakers (2009) give plausible ranges and
  the ex-Gaussian/shifted-Wald ↔ diffusion mapping, and the HDDM defaults (Wiecki
  et al. 2013) are a ready reference. Your `mu_lv ~ N(log 10, 0.5)` and `mu_la ~
  N(log 1, 0.5)` are already close to these; the point is to cite them.
- **LATER rate.** Median saccadic latencies near 200 ms imply a rate mean around 5
  s⁻¹ with SD ~1–2 s⁻¹ — good hierarchical priors if you move LATER to a
  hierarchical fit.

---

## 6. Cross-platform robustness (what I changed)

- The app degrades gracefully: if PyMC is not installed it disables only the
  Bayesian fit and still runs the preview, LATER, figures, and export, with a
  clear message on how to enable PyMC.
- Bimodality detection prefers `diptest` (your method) and falls back to a
  Gaussian-mixture BIC comparison if it is absent, so the app runs even on a
  minimal install.
- Sampling runs single-core inside the app, which avoids the multiprocessing
  pickling problems that PyMC can hit on Windows and macOS when launched from a
  GUI process. (For a batch script you can raise the core count.)
- Loading is robust to encoding and delimiter, accepts the wide `pooled_data.csv`
  or a long format, applies the `BlockType == I` filter, and reads either
  `SpeedCode` or `Speed_deg_per_s`.
- Fits are wrapped so a failure shows an error in the page rather than crashing
  the app.
- `Dockerfile` + `environment.yml` give a reproducible, identical environment on
  every OS.

---

## 7. Suggested reading

Your `REFERENCES.md` already covers the pipeline. The additions specific to the
suggestions above:

- Matzke, D., & Wagenmakers, E.-J. (2009). Psychological interpretation of the
  ex-Gaussian and shifted Wald parameters. *Psychonomic Bulletin & Review.*
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation
  using leave-one-out cross-validation and WAIC. *Statistics and Computing.*
- Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation
  matrices (the LKJ prior).
- Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical
  models (why the non-centred parameterisation you use matters).
