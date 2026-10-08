# KINARM RT 2.0 — plan, decisions, and what comes next

## What was asked

Experiments instead of paradigms; switch between Experiment 1 and 2 with each one's own models and figures; choose
which statistical model to run, including deprecated ones (LATER, two-boundary), with an explanation of why they were
set aside; choose which figures to see; fast, robust, user-friendly, accurate enough to publish; code as close to
one-to-one with the repository scripts as possible; a modern liquid-glass interface.

## The decisions, and why

1. **Run the scripts; do not re-implement them.** One-to-one is only guaranteed if it is the same code. The app vendors
   the repository's scripts unchanged and executes them with `python <script>` in a run folder, the way they are run
   by hand. Every analysis, number and figure is the script's own. This removes the v1 problem of parity tests
   chasing a re-implementation, and the simplified figure copies. It is enforced by the manifest, `--check`, and
   engine tests that require byte-identical tables.
2. **Ship the lab's results.** Browsing needs no computation; runs are for new data.
3. **Isolate runs.** One folder per experiment × dataset (content hash), one subprocess per step: a crash cannot take
   the app down, steps resume and are reused, logs are kept, and two experiments never share files.
4. **Deprecated means labelled, not hidden.** Each deprecated model states what it is, why it was set aside (with the
   numbers), and what it is still useful for, and says so again after it runs.
5. **Glass without breaking Streamlit.** No transform, filter or backdrop-filter on Streamlit containers (each breaks
   the fullscreen overlay); glass comes from layered translucency, motion is opacity-only, and the one moving element
   (the orbit) lives inside its own SVG. Panels are keyed containers, so the CSS targets stable `st-key-*` classes.

## Delivered in 2.0.0

- Engine: registry, background runner (progress, cancel, resume, reuse, per-step logs), data intake and checks,
  Experiment 2 raw-file builder, results reader with PDF previews, headline numbers.
- Pipelines: Experiment 1 (15 scripts), Experiment 2 (20), LATER for both experiments; manifest and sync tool.
- Reference results: both experiments, LATER for both, two-boundary (Experiment 1), comparison figures and tables.
- Interface: experiment switch and Compare view; Overview, Run, Figures, Tables and Models sections; status badges;
  deprecation notes; figure detail with explanations and downloads; light and dark themes.
- Tests: manifest and repository sync, registry consistency, end-to-end byte-identical reproduction (Experiment 2
  Method A and fixed-t₀ fits, Experiment 1 LATER), step skipping, and every view of the interface.

## Delivered in 2.1.0

The fixes and additions listed in CHANGELOG.md [2.1.0]: navigation callbacks, the WebKit-safe orbit, panel spacing, every figure in the gallery, side by side, deep links, error boundaries, the missing Experiment 1 tables, and a real-browser check (`desktop/ui_check.py`) run on Streamlit 1.50 (Python 3.9) and 1.65 (Python 3.12).

## Delivered in 2.2.0

Every figure, table and document in the repository is in the app (checked by content in `tests/test_completeness.py`), a Documents section, gallery thumbnails, calmer transitions, run hardening and edge-case tests, and a desktop build matrix that really covers Intel Macs (`macos-15-intel`).

## Next

1. **Two-boundary live runs** (Experiment 1): the readiness check and the Bayesian two-boundary fit from
   `Deprecated Pipelines/Two-boundary DDM/Experiment 1/code`, as optional deprecated steps.
2. **Live comparison**: run the `P1_vs_P2_*` comparison scripts against the user's own runs, not only the committed ones.
3. **Experiment 1 supplementary analyses**: configuration copies of the speed-effect tests and recovery scripts.
4. **Visual QA on the desktop builds** (Windows WebView2, macOS WebKit). The interface is now checked in headless Chromium
   on two Streamlit versions; the packaged windows use the system web engines, worth one look each before release.
5. **Repository naming**: rename `Current Pipeline` / `Paradigm 2 Pipeline` to Experiment 1 / 2, then update the paths
   in `tools/sync_pipelines.py`.

## Limits worth knowing

- Figures are the scripts' images, so they are not interactive; tables are.
- Run time is the scripts' run time: Method A in about a minute, the full Bayesian set in roughly 15–40 minutes on
  four cores. Reuse makes repeated runs instant.
