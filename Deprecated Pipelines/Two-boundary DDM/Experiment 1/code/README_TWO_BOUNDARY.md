# Two-boundary DDM — what it needs, what it gives, how to run it

Companion to the existing SNL-RT pipeline. Everything here follows the same
conventions as `Bayesian_HRT_fit.py` / `Bayesian_SRT_fit.py`: PyMC/NUTS, partial
pooling across participants, non-centred parametrisation, the same t₀ floors and
the same bounded t₀ transform. The only thing that changes is the likelihood.

---

## 1. The prerequisite

A single-boundary shifted Wald is a model of **reaction time**. A two-boundary DDM
is a model of **reaction time and which of two responses was made, jointly**. That
second dimension is not optional — it is where the extra parameters get their
information.

`pooled_data.csv` contains:

| column | what it is |
|---|---|
| `Participant` | participant id |
| `Speed_deg_per_s`, `SpeedCode` | target speed, 0 / 75 / 150 |
| `BlockType` | `I` for interception |
| `HandRT_ms` | hand reaction time |
| `GazeSRT_ms` | saccadic reaction time |

**RESOLVED — see `FINDINGS_REAL_DATA.md`.** The `pooled_data.csv` in this project
has 38 columns, not the 6 listed in `CODE_REFERENCE.md`, and includes
`SignedError_deg` = wrap(HandDir - TargetDir). That is the outcome variable, and it
is well balanced (32-38% minority share per speed). The section below is kept as the
general guide for choosing an outcome; for this dataset the answer is
`--sign-col SignedError_deg`, hand only. There is no gaze spatial variable, so the
eye cannot take this model.

### What to look for in the raw files

Ranked best to worst for this task:

**1. Signed spatial error at interception — recommended.**
Did the response land *ahead of* or *behind* the target. Either a precomputed signed
error column, or the ingredients: interception/endpoint position, target position at
that moment, and target motion direction. For the eye, the first-saccade landing
position relative to the target at landing.

Why this one:
- It is defined identically for the hand and the eye, so the cross-effector
  comparison — the reason for dropping LATER — survives intact.
- It *is* the spatial uncertainty raised in the methodological feedback. The two
  boundaries become "aim ahead" versus "aim behind", which is a decision the task
  genuinely poses.
- The speed manipulation should move it, which makes it testable rather than
  decorative. At 0 deg/s the target is stationary so lead/lag should be
  near-symmetric noise; at 150 deg/s it should skew toward lead. The symmetric
  0 deg/s condition is also where the model is best identified (see §3).

**2. A genuine directional choice**, if the task ever presented two possible
targets or two possible motion directions and the participant had to pick. This is
the textbook case the two-boundary DDM was built for. Use `--choice-col`.

**3. Hit / miss.** Works mechanically, but it is the weakest option conceptually:
the two DDM boundaries represent competing *response alternatives*, not success
versus failure of execution. Dichotomising on hit/miss mixes the decision process
with motor execution, and a reviewer will say so. Use only if nothing else exists,
and say plainly what the boundaries are being asked to mean.

If none of these exist in the raw files, the honest conclusion is that the task
does not produce the discrete outcome a two-boundary DDM requires, and the
single-boundary shifted Wald stays. That is a defensible answer to the question,
not a failure to answer it.

---

## 2. Run order

```
python two_boundary_readiness.py --data raw_trials.csv                    # inspect
python two_boundary_readiness.py --data raw_trials.csv --sign-col HandErr_deg
python identifiability_2b.py                                              # optional
python Bayesian_2B_fit.py --data raw_trials.csv --effector hand --sign-col HandErr_deg
python Bayesian_2B_fit.py --data raw_trials.csv --effector eye  --sign-col GazeErr_deg
```

`two_boundary_readiness.py` with no outcome argument prints every column with its
type, unique-value count and sample values, then names the columns that could serve
as an outcome. Run it first — it is the fastest way to find out what the raw files
actually offer.

### Files

| file | what it does |
|---|---|
| `wfpt.py` | Navarro-Fuss Wiener first-passage density. NumPy reference (adaptive terms), differentiable PyTensor version (fixed terms, for NUTS), exact inverse-CDF sampler. |
| `test_wfpt.py` | Four independent validations of the density. Run after any edit. |
| `two_boundary_readiness.py` | Data audit. Per-cell minority shares and a verdict: per-cell, hierarchical-only, or not at all. |
| `identifiability_2b.py` | Parameter recovery as a function of n and minority share. The quantitative version of "can we fit this?". |
| `Bayesian_2B_fit.py` | The hierarchical model, plus posterior predictive checks and a LOO comparison. |

---

## 3. The identifiability wall

Simulate from known parameters, refit by MLE, measure recovery. From
`identifiability_2b.py` (a = 1.5, w = 0.58, t₀ = 165 ms):

| n | minority share | v RMSE | a RMSE | w RMSE | t₀ RMSE |
|---|---|---|---|---|---|
| 50 | 38.5% | 0.19 | 0.12 | 0.050 | 23 ms |
| 50 | 12.2% | 0.30 | 0.13 | 0.051 | 19 ms |
| 50 | 2.7% | 0.44 | 0.73 | 0.092 | 22 ms |
| 50 | 1.2% | 0.55 | 1.13 | 0.134 | 25 ms |
| 50 | 0.0% | 0.95 | 1.80 | 0.239 | 23 ms |
| 120 | 41.2% | 0.12 | 0.05 | 0.026 | 12 ms |
| 120 | 13.4% | 0.17 | 0.05 | 0.032 | 18 ms |

Read it as: boundary separation and start point collapse below roughly **5%
minority responses**, and are fine above ~12% even at n = 50. This independently
reproduces the ~5% error-rate figure already established for this project.

Note the last column. **t₀ recovery is flat at 15–25 ms across the whole range**,
including the cells where a and w are unrecoverable. The non-decision-time question
survives even where the rest of the two-boundary model does not — which matters,
because t₀ is the actual scientific question.

`two_boundary_readiness.py` applies the 5% threshold (and a ≥10 minority trials
floor) per cell and reports how many cells clear it.

---

## 4. What changes relative to the single-boundary model

| | single-boundary shifted Wald | two-boundary DDM |
|---|---|---|
| data | RT | RT **and** response |
| `a` | height of one barrier | **separation** between two barriers |
| `v` | drift, positive by construction | drift, **sign free** |
| `w` | — | relative start point z/a, in (0,1) |
| `t₀` | non-decision time | non-decision time, same meaning, same floor |

**`a` is not comparable across the two models.** As a separation it lands at
roughly twice the single-boundary value for comparable RTs. Never put them on the
same axis without saying so; the numbers in `Bayesian2B_*_cells.csv` and
`Bayesian_hrt_fits.csv` are on different scales.

`w` is the parameter the single-boundary model structurally cannot express, so it
is the thing the two-boundary model is really being bought for. If `w` comes back
at 0.5 with credible intervals spanning most of (0,1), the two-boundary model is
not telling you anything the Wald wasn't.

Not included, deliberately: across-trial variability in drift (`sv`), start point
(`sz`) and non-decision time (`st0`) — the "full" DDM. These are the hardest
parameters to identify even in well-powered two-choice designs, and adding them to
cells with a few dozen trials would make the identifiability problem worse rather
than better. Mention them as a limitation rather than fitting them.

---

## 5. Model comparison

`Bayesian_2B_fit.py` fits a second model on the *same joint data* — a hierarchical
shifted Wald on RT times an independent hierarchical Bernoulli on the response —
and compares by LOO. Both models give a joint likelihood over (RT, response), so
the comparison is valid.

The question it answers: **does coupling RT and response through a single diffusion
buy anything over modelling them independently?** If the two-boundary model does
not win, the honest report is the single-boundary Wald plus a separate choice
analysis, and you can say so with a number attached.

This is the cleanest available answer to "is a two-boundary DDM more appropriate
here?" — it converts a modelling-philosophy argument into a model comparison.

---

## 6. Validation

`test_wfpt.py` checks the density four independent ways:

1. **Quadrature** — integrating the defective density over time reproduces the
   analytic barrier probability `(1 − e^(−2vaw)) / (1 − e^(−2va))` to 1e-6 across
   nine parameter sets spanning drift −1.5 to 10, separation 0.6 to 2.5, and start
   points 0.3 to 0.7.
2. **Monte Carlo** — matches an independent Euler-Maruyama simulation.
3. **Agreement** — the fixed-term PyTensor version matches the adaptive-term NumPy
   reference to 3e-14 in log density, both barriers, across the same grid.
4. **Gradients** — analytic gradients w.r.t. v, a and w are finite and match
   central finite differences to 1e-9. Without this NUTS cannot sample.

Run it after any edit to `wfpt.py`.

One note on the sampler: `sample_ddm` (exact, inverse-CDF) is preferred over
`simulate_ddm` (Euler-Maruyama) everywhere. Euler overshoots the barrier by
O(√dt), which biases RTs fast; at n ≈ 20,000 that bias is large enough to fail a KS
test against the correct density even though the density is right. It also runs
about 300× faster.

---

## 7. Requirements

```
conda install -c conda-forge pymc arviz
```

Same constraint as the rest of the pipeline: pip-only PyMC on Windows is not a
viable substitute. `wfpt.py`, `two_boundary_readiness.py` and `identifiability_2b.py`
need only NumPy, SciPy, pandas and matplotlib; `test_wfpt.py` additionally needs
PyTensor (which arrives with PyMC).
