# Changelog — kinarm-rt-app

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.6.0] — 2026-07-22

### Fixed — the vincentile difference figures were computing the wrong quantity

`vincentile_figures.py` takes the HRT minus SRT difference **per trial** -- hand
and eye from the same trial, keeping only trials where both measurements fall
inside their own filter window -- and vincentizes those differences. The app was
vincentizing each effector separately and subtracting the two curves, which sorts
the effectors independently and pairs the slowest hand trial with the slowest eye
trial. That is a different quantity and produces a visibly different figure: on the
bundled sample it gave a near-flat line around 113 ms, where the correct
computation rises from about -7 ms in the first bin to about 247 ms in the
twentieth.

The cause was upstream. Converting the wide file to the tidy long format split each
source row into a hand row and an eye row and discarded which trial they came from,
so the two could not be paired again. The tidy table now carries a `trial` column
taken from the source row index, and the paired difference is computed the way the
script computes it. Verified equal to the script's output to floating-point
precision across all participants, bins and speeds.

The fallback for long-format files that genuinely have no trial identity is still
the subtract-the-vincentiles form, but it is now clearly documented as an
approximation rather than presented as the same thing.

### Verified — the rest of the figure inputs

Every other quantity behind the advanced figures was checked against its
standalone script on the same data and matches exactly:

- pooled marginals for the KDE overlay and the histograms (trial counts and values)
- the shape-implied non-decision time, skewness and skew/CV ratio in the flooring
  diagnostic
- the vincentile binning helper itself
- the non-decision-time figure's group means, standard deviations and Friedman test
  (p = 0.00338805 against the published hand table, to eight decimals)
- the LATER reciprocal transform and line fit, and the drift-diffusion fits and
  selection rule, which were already covered

Six new parity tests pin the vincentile pairing, trial identity, the pooled
marginals and the flooring diagnostic, so a regression here fails a test rather
than quietly changing a figure.

### Changed — the download tab builds on request, with progress

Preparing a report redraws every figure, and all three artefacts were being built
eagerly on every rerun of the tab, including whenever an unrelated control changed.
Each is now built only when asked for, behind a progress bar that names the figure
being drawn and reports the finished file size. Anything already prepared is
offered as a direct download until the next fit, at which point it is discarded
because it describes the previous run.

## [1.5.0] — 2026-07-22

### Fixed — runs could stall instead of finishing

Cell fitting was being spread across worker processes by default. That was a
mistake, and an inconsistent one: the sampler is deliberately run with `cores=1`
because multiprocessing under Streamlit is unreliable on Windows, and then the same
hazard was introduced for the per-cell fits. joblib's process backend starts fresh
interpreters that re-import the module, and under `streamlit run` that can hang
rather than fail cleanly, which leaves the run sitting on a stage forever.

Fitting is now single-core by default. Using several cores is an explicit choice in
the interface, and it is guarded three ways: a couple of trivial tasks are run first
to confirm workers actually start (with a short timeout, and the result remembered
for the session), each cell has a result timeout, and any failure or partial result
finishes the remaining cells in this process. The setting only affects how long a
run takes, never what it produces -- cells are independent and seeded individually.

### Fixed — time estimates that could not be trusted

Three separate faults:

- The estimate was the average pace since the start. Cells cost very different
  amounts -- a two-component fit runs several times longer than a single one -- and
  cheaper cells often complete first, so the estimate promised a finish nowhere
  near the truth and then climbed as the expensive cells arrived.
- The saccade branch of the Bayesian fit changed its own total part-way through,
  once it knew how many cells needed two components, so the bar and its estimate
  jumped.
- Reusing an already-computed selection restarted the count from one for whatever
  remained, so the bar went backwards.

The estimate now comes from the median pace over a trailing window, and is withheld
entirely when the recent per-item times are uneven enough that no figure would mean
anything -- "estimating time left" instead of a confident wrong number. Selection
and sampling are reported as separate phases with their own totals, the bar never
moves backwards, and the elapsed time and item count are always shown so a long
stage is visibly alive.

## [1.4.0] — 2026-07-22

### Fixed — figures were rendering at their natural size

The rule that fits an expanded figure to the viewport was written against
`[data-testid="stFullScreenFrame"] img`. That wrapper is present whether or not a
figure is expanded, so `width: auto !important` also applied inline and overrode
`use_container_width`: a 1400-pixel-wide PNG rendered at 1400 pixels inside a
960-pixel card and spilled out of it. The viewport-fitting rules are now scoped
with `:has(button[aria-label="Close fullscreen"])`, which matches only the frame
that is actually expanded. Inline figures are pinned to the column width.

Expanded figures were also capped at their in-page width, because the wrappers
between the frame and the image keep their column sizing; those are released in
the expanded state so the figure uses the viewport (about 1400 x 505 in a
1500 x 950 window, against 882 wide inline).

The minimise control is 44 x 44 with a larger icon **in the expanded state only**.
Enlarging it inline is what previously pushed it out from under its own toolbar and
stopped clicks reaching it.

Display resolution dropped from 150 to 110 dpi. At typical column widths that is
still about twice the displayed pixel size, and a tab holding a dozen figures no
longer ships several megabytes of PNG. Exported figures are unaffected — those are
written at 300 dpi.

### Fixed — no repeated work in Method B

Method B ran the single-versus-two-component selection twice: once while deciding
which cells need two components, and again in the Method A cross-check that
follows. `fit_effector` now returns the selection it computed and `mle_preview`
accepts it, which removes the most expensive step of the run from the second pass.
Verified to give identical parameters, KS and model choice.

### Changed — progress reporting

Every stage of a run is listed before any of it starts, each with its own bar, plus
an overall bar counting completed stages and elapsed time. Previously some stages
had bars and others were plain text lines written between them, and the LATER stage
created a bar it never updated, so it jumped straight from "starting" to "done".
Status messages from the sampler now update their own stage's bar rather than
appearing as separate lines.

### Changed — spacing in the results section

The results section holds seven tabs, each a stack of figures, tables and captions.
The tab strip is now separated from its panel by a rule and more padding, stacked
figures have room between them, and headings, tables and expanders have consistent
margins.

## [1.3.0] — 2026-07-22

### Fixed — the minimise control in fullscreen

Two separate faults, both in the stylesheet. Bare Streamlit closes fullscreen
correctly; the themed app did not, and clicks on the control never reached it.

- A blanket `.block-container > div{ animation:kxFade .5s both; }` with staggered
  `nth-child` delays. With fill mode `both` and a delay, a matched div holds the
  from-state — `opacity: 0` — for the length of its delay: invisible, but still
  laid out and still taking pointer events. Streamlit re-indexes those divs when
  the fullscreen overlay mounts, which left a transparent div sitting over the
  control. Removed; section entrances are handled by a scoped rule instead. No
  animation anywhere now uses `both`, since a retained fill on anything that can
  host the fullscreen frame causes the same class of problem.
- Every element toolbar was given `z-index: 2147483000`. With eleven figures on
  the Graphs tab, the other ten toolbars painted above the fullscreen overlay and
  intercepted the click. Only the expanded frame — the one containing a
  "Close fullscreen" button — is lifted now.

The fullscreen image is also fitted to the viewport (`max-width`/`max-height` with
`object-fit: contain`) rather than rendering at its natural pixel size, which had
made the scaling look wrong.

### Fixed — wording that overstated the express-saccade result

The two-component saccade fit was labelled "express/regular" throughout. In the
published data sixteen cells across nine participants need two components, but
only one of those sixteen has its faster mode below the 130 ms express cutoff —
the rest sit between 135 and 185 ms, which is ordinary saccade latency. Only one
participant is genuinely express-dominant by LATER (express fraction 0.74–0.84;
the next highest is 0.17). Describing nine participants as express was wrong.

The interface now says "two-component", shows a `fast mode < 130 ms` column, and
states how many cells are actually in express territory. The CSV column names
(`t0e`, `t0r`, `express_mode`, `reg_mode`, `pi`) are unchanged, so exported tables
still line up with the pipeline's own files. The mixture toggle's help text also
described the dip test, which stopped being the selection rule in 1.2.0.

### Changed — speed

Exact optimiser settings made fitting substantially slower: a cell needing two
components costs about nine times a single fit (roughly 5.4 s against 0.6 s).
Cells are independent and the optimiser is seeded per cell, so they are now fitted
across cores, leaving one free. Verified to give byte-identical parameters, KS and
model choice to the sequential path. The saccade selection inside the Bayesian fit
was running serially inside the sampling loop and is now part of the same parallel
step.

### Added — progress and time estimates

`ui.StepBar` gives each stage of a run its own labelled progress bar with a time
estimate derived from the rate observed so far, so it adapts to the machine and to
how many cells need the slower fit. Method A reports per cell; Method B reports
across cell selection, the per-speed hierarchical fits, and each two-component
cell. The panel lists the stages, and the run reports how many cores it used.

### Changed — motion

Section cards rise 26 px over 0.62 s with a stagger. Previously the blanket fade
above was fighting the scoped section rule, so neither read as motion. The glow
behind the selected tab (`box-shadow: 0 4px 14px`) is gone; the filled pill is the
highlight. The hover lift on tabs was removed too.

### Added

- `test_selection_rule_reproduces_recorded_output` replays the single-versus-
  two-component rule against the published `DDM_srt_fits.csv` and requires all 48
  cells to be classified identically, using the recorded statistics so it does not
  depend on refitting.
- `test_two_component_is_not_the_same_as_express` records the distinction the
  interface wording depends on.

### Known issues

- Streamlit reports `use_container_width` as deprecated and due for removal. The
  app still uses it throughout. It works today but will need replacing with
  `width="stretch"` before a future Streamlit drops it.

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
