# Changelog — kinarm-rt-app

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
