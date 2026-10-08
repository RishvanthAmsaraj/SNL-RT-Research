# Answers to the three questions — and a correction

## 1. Do we have the data?

**Hand: yes. Eye: no.** The missing data is a partial gap, not a blocker.

| | status |
|---|---|
| Hand outcome variable | `SignedError_deg`, 83.2% complete |
| Cells fittable | 39/40 per-cell, 40/40 hierarchically |
| Minority share | 32–38% per speed — six times clear of the wall |
| Eye outcome variable | **does not exist** |

The gap I flagged costs you the 0 deg/s cells for CMT003–CMT010 — 16 cells become 8.
It does **not** stop the analysis. All 40 remaining cells fitted successfully, and no
parameter hit a bound.

Worth recovering if you can, because 0 deg/s is the condition where lead/lag is pure
bias rather than prediction, but the analysis runs without it.

---

## 2. Gaze spatial variables — checked exhaustively, they don't exist

I took the union of every column across `pooled_data.csv` and all 16 master summary
files: **38 columns total**. No master file contains anything `pooled_data.csv`
doesn't. Only two are gaze-related:

- `GazeSRT_ms` — saccadic reaction time
- `GazeWindow_ms` — a **time**, confirmed four ways:
  - integer-valued on every trial (angles wouldn't be)
  - does not scale with target speed (mean 18.0 / 6.4 / 10.5 ms at 0 / 75 / 150) — a
    spatial gaze error necessarily would
  - correlations with every spatial variable are near zero: `SignedError_deg` +0.10,
    `UnsignedError_deg` −0.06, `HandDir_deg` +0.13, `TargetDir_deg` +0.15
  - correlates ±0.62 with the two RTs, which is what a timing variable does

Every directional column is `HandDir_*` or `TargetDir_*`. There is no gaze direction,
no saccade landing position, no gaze error, under any name.

**So the eye genuinely cannot take a two-boundary model from these files.** If you
want it, saccade landing position has to come out of the raw KINARM traces.

---

## 3. Why is t₀ higher? — and I need to walk back what I said

### It is not the trial set

I refitted the single-boundary Wald on the *exact* trials used for the two-boundary
fit and on all hand trials. **The difference is 0.00 ms in 40/40 cells** — within
these cells the two trial sets are identical. So the shift is entirely a model
effect.

| speed | Wald (all trials) | Wald (same trials) | two-boundary | difference |
|---|---|---|---|---|
| 0 | 166.7 | 166.7 | 185.1 | +18.4 |
| 75 | 158.4 | 158.4 | 179.8 | +21.4 |
| 150 | 143.6 | 143.6 | 169.0 | +25.4 |

Mean model effect **+22.4 ms**, positive in 40/40 cells (Wilcoxon p = 1.8e-12).

### The mechanism looked flattering at first

The Wald sits **on the 130 ms floor in 14/40 cells**. The two-boundary model is
floored in **0/40**. The obvious reading is that the Wald was being held down by the
constraint and the two-boundary model, having an extra shape parameter (w) to absorb
the RT skew, is free to put t₀ where the data actually want it.

That reading is probably wrong.

### The cross-fit recovery test says the opposite

Simulate from a known process, fit both models, see who recovers t₀ = 170 ms:

| truth | Wald recovers | two-boundary recovers |
|---|---|---|
| two-boundary DDM | 163.7 ms (**−6.3**) | 170.3 ms (**+0.3**) |
| single-boundary Wald | 169.2 ms (**−0.8**) | 200.5 ms (**+30.5**) |

**The two-boundary model over-estimates t₀ by ~30 ms when the data don't actually
have two-boundary structure.** The Wald's bias in the other direction is small
(−6 ms). The +22.4 ms shift you're seeing is the right size and direction to be that
artifact.

### And the head-to-head goes against the two-boundary model

Both models have 4 free parameters on the same joint (RT, direction) data, so
log-likelihoods compare directly with no penalty adjustment:

- **two-boundary DDM wins in 7/40 cells**
- mean Δ log-likelihood = **−14.5 per cell** against it (total −580)
- loses at every speed: 0/8 at 0 deg/s, 3/16 at 75, 4/16 at 150

A shifted Wald on RT plus an *independent* Bernoulli on direction — a model that says
RT and direction are unrelated — fits better than the diffusion that couples them.

### So: correction to my last message

I said the two-boundary fit "buys something real" and that t₀ "survives." Overstated.
What the evidence now supports:

- The **+22 ms is most likely misspecification bias, not a corrected estimate.** Do
  not report the two-boundary t₀ as the better number.
- **w = 0.403 is also suspect.** In a misspecified model, a free shape parameter
  absorbs misfit. Its distance from 0.5 is not by itself evidence of a real
  start-point bias.
- Your **published single-boundary t₀ values stand.** The flooring in 14/40 cells is
  still a genuine limitation of the Wald and worth stating — but the two-boundary
  model is not the fix.

### The likely reason it fits badly

This is the plain 4-parameter DDM with no across-trial variability in drift (`sv`),
start point (`sz`) or non-decision time (`st0`). The pure DDM makes a tight,
specific prediction about how the two response types' RT distributions relate, and
that prediction is usually wrong without the variability parameters. Adding them is
the standard fix — but they are the hardest parameters to identify, and at ~119
trials per cell you would be trading one identifiability problem for a worse one.

---

## What is solidly true regardless of model

Lead responses are **slower** than lag responses, and the gap scales with speed:

| speed | mean(lead RT − lag RT) | cells positive | Wilcoxon p |
|---|---|---|---|
| 0 | −2.0 ms | 2/8 | 0.547 |
| 75 | +20.2 ms | **16/16** | <0.001 |
| 150 | +37.7 ms | **16/16** | <0.001 |

No difference when the target is stationary, a large and perfectly consistent
difference once it moves, growing with speed. That is a real result, it needs no
model to state, and it is arguably more interesting than the DDM parameters — it says
aiming ahead of a moving target costs time, and costs more the faster it moves.

---

## Suggested line for your professors

The two-boundary model is now built, validated and fitted rather than argued about,
and the answer is empirical: on this data it fits worse than treating RT and movement
direction independently (7/40 cells, ΔLL = −14.5), and it inflates t₀ by about the
amount a misspecified two-boundary model inflates t₀ in simulation (+30 ms). The
single-boundary Wald stays as the primary model — not because the two-boundary
alternative was rejected on principle, but because it was fitted and lost.

Two honest caveats to include: this is per-cell MLE without across-trial variability
parameters, and the hierarchical Bayesian version with LOO (`Bayesian_2B_fit.py`) is
the proper test. It may soften the conclusion. It is unlikely to reverse it.

---

## Files

- `t0_decomposition.csv` — per cell: Wald on all trials, Wald on identical trials, two-boundary
- `t0_recovery.csv` — cross-fit recovery simulations
- `model_headtohead.csv` — per-cell log-likelihood comparison
- `real_2b_mle.csv` — the 40 two-boundary fits
