# Experiment 1 figures — what was corrected

Every figure was redrawn from the **committed Paradigm 1 tables** — no model was refitted, so no number changed. The scripts that do refit for a diagnostic (fixed-t₀ and floor sweeps) reproduce the committed tables exactly. The figures were made with the Paradigm 2 scripts (same code, corrected labels) set to Paradigm 1's speeds; they are in `Code (scripts that made these figures)/` and can replace the figure scripts in `Current Pipeline/Code/`.

| Figure | What was wrong | Now |
|---|---|---|
| `DDM/DDM_summary` | panel C drew the retired 100 ms hand floor | floor line at 130 ms; "0% Poor" is computed, not typed in |
| `Bayesian/Bayesian_summary` | panel A drew the 100 ms floor; panel C said "individual differences preserved; not floored" although all 14 participants sit on the floor | floor line at 130 ms; panel A states the counts (MLE 11/48 cells at the floor, Bayesian 0/48); panel C says 14 at the floor, 0 at the ceiling |
| `Bayesian/Bayesian_srt_ndt` | title said saccadic t₀ was "ESTIMATED, not fixed; individual differences preserved"; the legend covered CMT0012 and CMT009 | title: 14/14 intervals reach the 70 ms floor — not identifiable; legend clear of the data; grey ticks mark each participant's fastest-saccade ceiling; population mean μ = 40 ms kept from the Paradigm 1 fit |
| `SRT Analysis/SRT_fixedt0_sensitivity` | the legend partly covered the first point of the 90 ms line | legend below the axes; panel titles report the measured numbers (drift profiles correlate at r ≥ 0.96; KS range 0.002) — Paradigm 1's "stable" and "identical" claims hold |
| `NDT/NDT_barchart_bayesian` | footnote numbers were typed in (they were correct) | footnote computed from the tables; the saccade panel adds the ceiling ticks and states the bound counts; the "data favour < 70 ms" note sits where the legend no longer hides it |
| `SRT Analysis/SRT_identifiability` | the one cell stuck at its fastest saccade was counted as "identified" | shown in orange; title: 19/32 cells track the floor, 1 sits at the fastest saccade, 12 identified |
| `SRT Analysis/why_saccadic_t0_floors` | footnote numbers typed in ("skew/CV ≈ 3") | computed: hand skew/CV 12.9, implied t₀ 191 ms; eye 3.4, implied 20 ms — same conclusion |
| `DDM/ddm_srt_0_degs` (schematic) | the "91 ms (t₀)" label collided with the "100" tick | the colliding tick label is skipped |
| all other figures | — | unchanged apart from "Paradigm 1" in the title |
| `Supplementary/HRT_floor_control` (**new**) | — | the hand vs saccade floor test, as in Paradigm 2: 5/43 hand fits and 9/16 saccade fits follow the floor |
