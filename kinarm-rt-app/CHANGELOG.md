# Changelog — kinarm-rt-app

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] — 2026-07-22

The app is meant to be the pipeline with an interface on it, not a second
implementation of it. Auditing the fitting code against the standalone scripts
turned up seven places where it had drifted. All are now literal ports, and
`tests/test_parity.py` pins them there.

### Fixed — analysis logic

- **The per-cell fit used a different optimiser.** `mle_preview` minimised a
  reparameterised likelihood with Nelder-Mead. `DDM_fit.py` uses differential
  evolution with seeds 42 and 7, `maxiter=400`, `tol=1e-9`, `popsize=12`. The
  optimiser is now a literal port, verified to reproduce the script's parameters,
  log-likelihood and KS statistic to the last bit.
- **Saccade cells were all fitted as a single Wald.** The pipeline tries an
  express/regular mixture wherever the single Wald fails, and accepts it only on
  the rule in `DDM_fit.py`: single KS > 0.10, mixture KS < 0.10, mixing weight
  between 0.10 and 0.90, and the two modes at least 30 ms apart. That rule is now
  applied in the preview, in Method A, and when choosing which cells go to the
  Bayesian mixture.
- **Method A had no mixture path at all**, and ran with `maxiter=200`,
  `tol=1e-8`.
- **Bayesian mixture cells were selected by Hartigan's dip test.**
  `Bayesian_SRT_fit.py` takes the selection from the DDM fit table and only falls
  back to the dip test when that table is missing. The DDM rule is now recomputed
  from the same data, so the split matches without needing a CSV from an earlier
  run.
- **The fixed-t₀ refit dropped the contamination component** and trimmed trials
  below the fixed t₀. `SRT_fixed_t0_analysis.py` keeps the uniform contamination
  term and rejects the whole cell instead, since trimming changes the sample being
  fitted. It also used Nelder-Mead rather than differential evolution.
- **The identifiability sweep computed a different diagnostic.**
  `SRT_identifiability_check.py` refits t₀ at imposed floors of 40–90 ms and takes
  the slope of fitted t₀ against floor; a slope near 1 means the floor is setting
  t₀ rather than the data. The app instead fitted t₀ once with no floor and counted
  how many cells fell below each candidate. The figure now shows the script's two
  panels: per-cell traces coloured by whether they track the floor, and the
  distribution of slopes.
- **The sampler defaulted to 1000 draws / 1000 tuning / 4 chains.** The scripts use
  1500 / 1500 / 4, which is now the default.

### Changed

- Fitting modes are labelled **Method A (frequentist MLE)** and **Method B
  (hierarchical Bayesian)**, the names the scripts themselves use.
- The fixed-t₀ figure reproduces `SRT_fixed_t0_analysis.py`: drift by speed for
  each assumed t₀, and mean KS per assumed t₀ against the 0.10 acceptability line.
- Section cards animate in with a stagger, and the card styling applies again.
  Both had silently stopped working: the stylesheet targeted
  `[data-testid="stVerticalBlockBorderWrapper"]`, which current Streamlit no longer
  emits — the wrapper is now `stLayoutWrapper`, which also wraps every column, so
  it is narrowed with `:has(.kx-sec)` to the four step cards. The stagger used
  `:nth-of-type(1..4)` when the cards sit at positions 5 to 8 among their siblings,
  and now uses the general sibling combinator instead, which does not depend on
  where the cards fall.

### Added

- `tests/test_parity.py` — reference implementations copied out of `DDM_fit.py`,
  `LATER_analysis.py` and the Bayesian scripts, with assertions that the app
  reproduces them exactly: fitted parameters, negative log-likelihood, KS,
  the model-selection rule, prior means, sampler seeds, optimiser settings, bounds,
  floors and filter windows. An edit that changes a fitted number now fails a test
  rather than quietly disagreeing with the published CSVs.

### Verified unchanged

These already matched the scripts and were left alone: the hierarchical model's
priors, non-centred parameterisation, `t0 = floor + (min_RT − floor)·sigmoid(z)`
link and log-likelihood; `mu_z ~ Normal(0.5, 1)` for the hand against
`Normal(0.0, 1)` for saccades; one pooled fit across speeds for the hand against
per-speed fits for saccades; the Bayesian mixture's priors, shift link and
component relabelling; sampler seeds 7 and 11 at `target_accept = 0.95`; the LATER
reciprocal transform, central-90% line fit, 130 ms express cutoff and 40-trial
minimum; and every published constant (floors of 130 and 70 ms, the 40 ms mixture
shift floor, `V_MAX = 20`, `A_MAX = 2.5`, 5% contamination, and the 150–800 ms and
80–600 ms filter windows).

### Known differences

- The sampler runs chains sequentially (`cores=1`) where the scripts run four in
  parallel. Multiprocessing inside Streamlit is unreliable on Windows. Same seeds
  and same results; it only costs wall-clock time.
- Exact settings cost runtime. Method A takes a minute or two on 16 participants,
  and the identifiability sweep a few minutes, because both now do the same work
  the scripts do.

## [1.1.0] — 2026-07-21

The figures in the app had drifted from the ones the standalone scripts produce.
This release brings them back into line, and fixes two interface problems.

### Added

- **Conceptual DDM schematics** (`figures.ddm_schematic_figs`), one per target
  speed per effector — the annotated single-boundary diffusion diagram from
  `DDM_conceptual.py`: starting point, drift arrow, threshold, non-decision
  shading, noisy accumulation paths, and the Wald density band. The app had no
  equivalent of this figure at all.
- **Full vincentile suite** (`figures.vincentile_suite`) — the four figures from
  `vincentile_figures.py`: SRT/HRT KDE overlay, histograms, the HRT − SRT
  vincentile difference per speed, and the combined all-speeds view. The app
  previously drew a single quantile-vs-RT line.
- **Combined non-decision-time figure** (`figures.ndt_by_speed`) — hand and eye
  panels side by side with Friedman p-values, matching `NDT_barchart.py`.
- `tests/test_figures.py` — 13 regression tests covering every figure function.
  Figures had no test coverage before.

### Changed

- `figures.why_floors` now reproduces `why_saccadic_t0_floors.py`: per-effector
  RT densities annotated with the shape-implied t₀ (mean − 3·SD/skew), the
  physiological floor, and the skew/CV ratio against the pure-Wald value of 3.
  It was previously a scatter of skew against skew/CV.
- `figures.reciprobit` now reproduces `LATER_analysis.py`: a regular participant,
  an express-dominant participant with the express early line, and LATER median
  latency by speed.
- Non-decision-time panels are anchored to the published axis windows (hand
  118–205 ms, saccade 55–150 ms) and expand only when the data fall outside them.
  Fitting the window to the data would shrink-wrap the saccadic panel — where t₀
  piles on the floor — into a span of a few milliseconds, visually exaggerating
  differences that are not there. This is the same distortion that motivated the
  earlier switch away from bars truncated at a non-zero baseline.
- The explanatory callouts (tinted panel with a blue left border) are now plain
  secondary text, consistent with the other inline explanations.
- The command-line pipeline emits the same figure set as the app's Graphs tab.
- Section cards enter with a staggered rise instead of a flat fade, tab panels
  ease in, and `prefers-reduced-motion` is honoured.

### Fixed

- **The minimise control in fullscreen.** The stylesheet targeted
  `[data-testid="StyledFullScreenButton"]`, which current Streamlit builds no
  longer emit — the control is now `[data-testid="stBaseButton-elementToolbar"]`
  with an aria-label of `Fullscreen` / `Close fullscreen`. The rule matched
  nothing, so the button kept its default appearance: a pale icon on the
  figure's white background, effectively invisible, leaving Escape as the only
  way out. All four selectors (current and legacy) are now styled, and the
  control renders as a bordered button with a dark icon.
- Entrance animations use `animation-fill-mode: backwards` rather than `both`.
  With `both`, the final `translateY(0)` is retained, and any retained transform
  on an ancestor of a figure establishes a containing block that collapses
  Streamlit's `position: fixed` fullscreen overlay.

### Notes

- The tidy table carries no trial identifier, so hand and eye trials cannot be
  paired within a trial after the wide-to-long conversion. The vincentile
  difference is therefore computed as a vincentile shift — each effector is
  vincentized per participant and the vincentiles are differenced — rather than
  by differencing paired trials as `vincentile_figures.py` does. The quantity
  plotted per bin is the same; the pairing is not.
