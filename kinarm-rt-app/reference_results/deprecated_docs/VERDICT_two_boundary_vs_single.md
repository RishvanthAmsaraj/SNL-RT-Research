# Two-boundary vs single-boundary: the full verdict

**Recommendation: stay with the single-boundary shifted Wald.**

Not on principle — the two-boundary model was built, validated, fitted to all 40
cells, extended with drift variability, and tested four separate ways. It lost every
test. This document sets out what was checked, every alternative considered, why the
two-boundary model fails, and what would change the answer.

---

## Executive summary

| finding | evidence |
|---|---|
| The outcome variable exists (hand) | `SignedError_deg`, 83.2% complete, 32–38% minority share |
| No gaze spatial variable exists | exhaustive column audit, 38 columns, four independent tests |
| The comparison procedure is fair | two-boundary wins 37/40 on data it generated itself |
| Two-boundary **loses** on real data | 7/40 cells, mean ΔLL −14.5 |
| Its **decision structure works** | choice-coupling term **+2.41**, scaling with speed |
| Its **RT distribution does not** | RT-shape term **−16.90**, worse in 38/40 cells |
| Drift variability doesn't rescue it | likelihood monotonically decreasing in `sv` |
| The +22 ms t₀ shift is an artifact | +30 ms bias in recovery when truth is single-boundary |

---

## Part 1 — What was re-verified

### 1.1 The density is correct

`test_wfpt.py` re-run from scratch: **38/38 checks pass.**

- Quadrature: integrating the defective density reproduces the analytic barrier
  probability `(1 − e^(−2vaw))/(1 − e^(−2va))` to 1e-6, across nine parameter sets
  spanning drift −1.5 to 10, separation 0.6 to 2.5, start points 0.3 to 0.7.
- Monte Carlo: matches independent Euler-Maruyama simulation.
- PyTensor vs NumPy: agree to 3e-14 in log density, both barriers.
- Gradients: finite and matching central finite differences to 1e-9.

The implementation is not the problem.

### 1.2 The comparison procedure is fair

The obvious worry: maybe the head-to-head is rigged against the two-boundary model.
Test — simulate data **from the fitted two-boundary parameters** for each of the 40
cells, then run the identical comparison.

- Two-boundary wins on its own data in **37/40 cells**, mean ΔLL **+8.22**
- t₀ recovery on its own data: two-boundary **+0.1 ms** bias, Wald **−9.0 ms**

The procedure detects two-boundary structure when it is present. On the real data,
the two-boundary model loses on merit.

### 1.3 The trial sets are identical

The Wald refitted on exactly the trials used for the two-boundary fit versus all
hand trials: **0.00 ms difference in 40/40 cells.** Every comparison below is
like-for-like.

---

## Part 2 — Every approach considered

### A. Single-boundary shifted Wald on RT (current pipeline)

Uses RT only. 3 parameters. Fits the marginal RT distribution to KS = 0.071, within
the calibrated 5% critical value of ~0.088.

Limitations, both real and worth stating: t₀ is only weakly identified
(Ratcliff & Van Dongen 2011), it sits on the 130 ms floor in 14/40 hand cells, and
it cannot represent a directional decision at all.

**Verdict: the recommended primary model.**

### B. Two-boundary DDM on dichotomised `SignedError_deg` (lead vs lag)

The main candidate. Positive signed error = hand leads the target; verified that no
per-direction sign flip is needed, because both Left and Right targets travel
counter-clockwise (target angle displaces +17.8°/+35.3° left, +19.0°/+39.9° right at
75/150 deg/s).

Fitted to all 40 cells by MLE. All converged, no parameter at a bound.

**Verdict: tested and rejected.** Loses to Wald × independent Bernoulli in 33/40
cells. See Part 3 for why.

### C. Two-boundary DDM with across-trial drift variability (`sv`)

The standard fix when a pure DDM cannot match RT distribution shape. Implemented via
15-node Gauss-Hermite quadrature over `v ~ Normal(v, sv)`; verified to reduce exactly
to the plain WFPT at `sv = 0`.

Fitted to all 40 cells, and profiled with the other four parameters re-optimised at
each fixed `sv`:

| cell | sv=0 | sv=0.5 | sv=1.0 | sv=2.0 | sv=3.0 |
|---|---|---|---|---|---|
| CMT0011@75 | **161.53** | 161.43 | 161.14 | 160.06 | 158.49 |
| CMT006@150 | **129.41** | 129.39 | 129.34 | 129.27 | 129.03 |
| CMT002@0 | **174.56** | 174.53 | 174.43 | 174.03 | 173.33 |

Monotonically decreasing. `sv = 0` is the genuine optimum — not an optimiser
artifact. Mean log-likelihood gain from adding the parameter: **+0.505**, below the
1.0 needed to justify it. By AIC, Wald × Bernoulli wins 33/40 and the plain
two-boundary wins the other 7. The `sv` model wins nothing.

**Verdict: tested and rejected.** And it fails in the informative direction — see
Part 3.

### D. Two-boundary DDM with start-point variability (`sz`)

Not fitted. `sz` makes fast responses at the near boundary *more* likely, widening
the lower part of the RT distribution. The diagnosed problem is that the
two-boundary marginal RT is **already too wide** (Part 3). `sz` moves in the wrong
direction, as `sv` did.

**Verdict: predicted to fail for the same reason; not worth the compute.**

### E. Hierarchical Bayesian two-boundary (`Bayesian_2B_fit.py`)

Built, validated, ready to run. Partial pooling will shrink noisy per-cell estimates
and tighten the group parameters.

But shrinkage addresses *estimation noise*, not *structural misfit*. The failure here
is that the two-boundary RT distribution has the wrong shape in 39/40 cells —
consistently, not noisily. Pooling a consistent bias does not remove it.

**Verdict: worth running for the LOO number to report, but unlikely to reverse the
conclusion.** The script is included.

### F. Circular / 2D continuous diffusion (Smith 2016; Ratcliff 2018)

The most principled alternative, and the one most worth raising with your professors.

The real response in this task is **continuous** — `SignedError_deg` is an angle, not
a binary. Dichotomising to lead/lag throws away the magnitude and imposes a
two-alternative structure the task does not have. The proper model class for a
continuous spatial response is a 2D diffusion to a circular boundary, which gives the
joint density of (RT, response angle):

- Smith (2016), *Diffusion theory of decision making in continuous report*,
  Psychological Review 123:425–451
- Ratcliff (2018), *Decision making on spatially continuous scales*,
  Psychological Review 125:888–935

This would use the full spatial error rather than its sign, and would model exactly
the spatial uncertainty your professor raised.

Two cautions. First, it is a substantial implementation — the density involves Bessel
function series and is considerably harder than the WFPT. Second, and more important:
the diagnosis in Part 3 is that the observed RT distribution is **too tight** to be
produced by any accumulation process that also generates this much directional
spread. A circular diffusion has the same structural tension. It may well fail the
same way.

**Verdict: the right answer to "what if not two boundaries?", but likely to inherit
the same problem. Raise it, scope it, don't start it until the tension in Part 3.4
is resolved.**

### G. Two-boundary on hit/miss (`HasTgtReached`)

Hit rates 99.9% / 99.1% / 93.0% by speed. At 0 deg/s that is a 0.1% minority share,
an order of magnitude below the identifiability wall. It also conflates the decision
with motor execution — the two DDM boundaries represent competing response
alternatives, not success versus failure.

**Verdict: unusable.**

### H. Two-boundary on a genuine discrete choice

The textbook case. Requires the task to have presented two response alternatives.
`Direction` (Left/Right) labels which side the target starts on, not a choice the
participant made; `TPcol` is confounded with it one-to-one.

**Verdict: no such variable exists. The task did not pose a discrete choice.**

### I. Single-boundary Wald on RT **plus** a separate model of spatial error

What the data actually support. Wald × Bernoulli beats the two-boundary DDM in 33/40
cells with the same number of parameters. Extended properly, the spatial side would
be a regression on the continuous `SignedError_deg` — by speed, direction and
participant — rather than a Bernoulli.

**Verdict: the recommended structure. It matches how the data behave and it keeps
hand and eye on the same RT model.**

### J. LATER

Excluded by your decision with the professors. Noting only that the reason for
excluding it — cross-effector comparability — is the same reason the two-boundary
model cannot be adopted: there is no gaze spatial variable, so the eye cannot take
a two-boundary model either.

---

## Part 3 — Why the two-boundary model fails

### 3.1 The decision structure actually works

Decomposing the joint likelihood into `marginal RT × P(choice | RT)` separates the
two things the model is doing. Against Wald × Bernoulli:

| speed | RT-shape term | choice-coupling term | total | 2B wins |
|---|---|---|---|---|
| 0 | −16.06 | **+0.69** | −15.37 | 0/8 |
| 75 | −14.77 | **+1.75** | −13.02 | 3/16 |
| 150 | −19.46 | **+3.94** | −15.52 | 4/16 |
| **all** | **−16.90** | **+2.41** | −14.49 | 7/40 |

The choice-coupling term is **positive in 26/40 cells and grows monotonically with
speed** — exactly as it should, since the lead/lag decision only exists once the
target moves. The DDM's RT-dependent choice probability genuinely beats a constant
Bernoulli.

Supporting this, the fitted model predicts the observed lead/lag RT gap almost
exactly:

| speed | observed lead − lag | DDM predicts |
|---|---|---|
| 0 | −2.0 ms | −9.2 ms |
| 75 | +20.2 ms | +27.4 ms |
| 150 | +37.7 ms | +33.8 ms |
| overall | **+22.7 ms** | **+22.6 ms** |

Correlation across cells r = 0.72. The two-boundary account of *how RT and direction
relate* is right.

### 3.2 The RT distribution is what kills it

The RT-shape term is **−16.90 per cell, negative in 38/40**. Locating it by
quantile — predicted minus observed, in ms:

| quantile | two-boundary | Wald |
|---|---|---|
| 5% | **−12.9** | −3.7 |
| 10% | −12.7 | −3.1 |
| 25% | −9.9 | −1.6 |
| 50% | −0.4 | +1.6 |
| 75% | +15.3 | +3.1 |
| 90% | +37.2 | +3.5 |
| 95% | **+51.4** | +0.5 |

The two-boundary model is 13 ms too fast at the 5th percentile and 51 ms too slow at
the 95th. The Wald tracks within ±4 ms across the whole range.

Marginal-RT KS: **two-boundary 0.148, Wald 0.071**, worse in 39/40 cells. Against
your calibrated 5% critical value of ~0.088, the Wald passes and the two-boundary
model fails.

**The two-boundary marginal RT distribution is far too dispersed.**

### 3.3 Which explains why `sv` made it worse

Drift variability *widens* the RT distribution. The problem is that it is already too
wide. Same for start-point variability. Every standard DDM elaboration pushes in the
wrong direction — which is why the likelihood profile in Part 2C decreases
monotonically rather than showing an interior optimum.

### 3.4 The underlying tension

To produce a 35/65 lead/lag split, the diffusion must place a boundary close enough
to the start point that roughly a third of trials cross the wrong way. Once the
boundary is that close, RTs become highly variable: short when the process goes
straight there, long when it wanders. The observed hand RTs are far tighter than
that.

Put plainly: **the hand RT distribution is too precise to have been produced by an
accumulation race that also generates this much directional variability.** Under a
diffusion account, RT precision and choice variability are inconsistent here.

The natural reading is that the directional error largely does **not** come from the
same process that determines RT. It comes from aiming imprecision, motor execution
noise and online correction — sources downstream of the decision to move. That is a
substantive claim about the task, and it is testable: it predicts that a continuous
spatial model with a *separate* noise process will beat any single accumulator
coupling the two.

Consistent with this, the two-boundary model does **worst at 0 deg/s** (0/8 cells),
where there is no lead/lag decision to make and the signed error is pure bias plus
motor noise.

---

## Part 4 — The t₀ question, resolved

Your original observation. The decomposition:

| speed | Wald (all trials) | Wald (same trials) | two-boundary | shift |
|---|---|---|---|---|
| 0 | 166.7 | 166.7 | 185.1 | +18.4 |
| 75 | 158.4 | 158.4 | 179.8 | +21.4 |
| 150 | 143.6 | 143.6 | 169.0 | +25.4 |

**Trial-set effect: 0.00 ms in 40/40 cells. Model effect: +22.4 ms, positive in
40/40** (Wilcoxon p = 1.8e-12).

The flattering reading was that the Wald sits on the 130 ms floor in 14/40 cells
while the two-boundary model is floored in 0/40, so the extra parameter frees t₀.
Cross-fit recovery says otherwise:

| generating process | Wald recovers t₀ = 170 | two-boundary recovers |
|---|---|---|
| two-boundary DDM | 163.7 (−6.3) | 170.3 (+0.3) |
| single-boundary Wald | 169.2 (−0.8) | **200.5 (+30.5)** |

The two-boundary model inflates t₀ by ~30 ms when the data lack genuine two-boundary
structure. Your observed +22.4 ms is the right size and direction to be that artifact.

**The mechanism follows directly from 3.2.** The two-boundary decision-time
distribution has a higher coefficient of variation than the Wald's. To match the
observed absolute RT spread it must shrink the decision stage — mean decision time
drops from 92 ms (Wald) to 70 ms (two-boundary) — and everything it takes off the
decision stage goes onto t₀.

**So: do not report the two-boundary t₀ as the better number. Your published
single-boundary values stand.**

The same caution applies to `w = 0.403` (p = 3e-8 vs 0.5). In a model whose RT shape
is wrong in 39/40 cells, a free shape parameter absorbs misfit. Its distance from 0.5
is not by itself evidence of a real start-point bias.

---

## Part 5 — What holds regardless of model

Lead responses are slower than lag responses, and the gap scales with target speed:

| speed | mean(lead RT − lag RT) | cells positive | Wilcoxon p |
|---|---|---|---|
| 0 | −2.0 ms | 2/8 | 0.547 |
| 75 | **+20.2 ms** | **16/16** | <0.001 |
| 150 | **+37.7 ms** | **16/16** | <0.001 |

Nothing when the target is stationary; large, perfectly consistent, and growing once
it moves. No model is needed to state it. Aiming ahead of a moving target costs time,
and costs more the faster it moves.

This is arguably a better result than any DDM parameter, and it survives everything
above.

---

## Part 6 — Data issues, unchanged

**Eight participants have no direction data at 0 deg/s.** CMT003–CMT010: all 120 of
their 0 deg/s interception trials are NaN in `HandDir_deg`, `TargetDir_deg` and
`SignedError_deg`, while `HandRT_ms` is 99% present. A kinematic extraction gap, not
missing trials. Costs 8 of 16 cells at 0 deg/s. Systematic, so probably fixable.

**No gaze spatial variable exists.** 38 columns across `pooled_data.csv` and all 16
master files; no master file contains anything pooled does not. `GazeWindow_ms` is a
time, confirmed four ways: integer-valued on every trial; does not scale with speed
(18.0 / 6.4 / 10.5 ms at 0/75/150, where a spatial error necessarily would);
near-zero correlation with every spatial variable (`SignedError_deg` +0.10,
`HandDir_deg` +0.13, `TargetDir_deg` +0.15); ±0.62 with the two RTs.

**Measurement timing is unresolved.** `SignedError_deg_HRT50` gives a materially
different picture (89.0% lead at 75 deg/s versus 67.4%; sign agreement only 79.9%)
but is only 25% complete. A DDM boundary is crossed at movement *initiation*, so in
principle the earlier measurement is the more correct input. Worth finding out what
`HRT50` means and whether it can be backfilled — though given Part 3.4, it is
unlikely to change the verdict.

**One label discrepancy.** `CMT002_MASTER_Summary.csv` has `BlockName` values reading
`CMT019_INTER_*`. The other seven mismatches are zero-padding only
(`CMT0011` → `CMT011`).

---

## Part 7 — Recommendation

**Primary model: single-boundary shifted Wald, unchanged, for both hand and eye.**

Report alongside it:

1. **The two-boundary test as a result, not an omission.** It was fitted and lost:
   7/40 cells, ΔLL −14.5, marginal-RT KS 0.148 vs 0.071.
2. **The decomposition**, because it is the interesting part. The decision structure
   helps (+2.41, scaling with speed); the RT distribution sinks it (−16.90). This is
   a specific, diagnosable failure, not a vague one.
3. **The lead/lag RT effect** (+20.2 and +37.7 ms) as a model-free finding.
4. **The Wald's honest limitations**: weak t₀ identifiability, the 130 ms floor
   binding in 14/40 hand cells, no directional decision represented.

### What to say to your professors

The two-boundary model was implemented, validated four ways, fitted to all 40 cells,
extended with drift variability, and checked against a fairness control. It fits
worse than treating RT and movement direction independently, and it inflates t₀ by
about the amount a misspecified two-boundary model inflates t₀ in simulation.

The reason is specific and worth reporting: the observed hand RT distribution is too
tight to be produced by an accumulation race that also generates this much
directional variability. The directional error appears to arise downstream of the
decision — from aiming and execution noise — rather than from the accumulation
process itself.

Two caveats to state plainly: this is per-cell MLE, and the hierarchical Bayesian
version with LOO would be the more rigorous test. And a continuous circular diffusion
(Smith 2016; Ratcliff 2018) is the principled model for a spatially continuous
response and has not been tried.

### What would change the answer

- **The hierarchical LOO comes out the other way.** Run `Bayesian_2B_fit.py`. Unlikely
  given a misfit consistent in 39/40 cells, but it is the proper test.
- **A genuine discrete choice turns up** in the raw KINARM files — two targets, a
  cued alternative, anything the participant actually chose between.
- **Saccade landing position is recoverable**, which would at minimum restore
  cross-effector symmetry and let the same test be run on the eye.
- **A circular diffusion beats both.** It would need the RT-precision tension in
  Part 3.4 resolved first, since that tension is not specific to having two boundaries.

---

## Files

| file | contents |
|---|---|
| `two_boundary_verdict.pdf` / `.png` | four-panel summary of the evidence |
| `wfpt.py` | Navarro-Fuss WFPT density, NumPy + PyTensor + exact sampler |
| `ddm_sv.py` | drift-variability extension, Gauss-Hermite |
| `test_wfpt.py`, `validation_log.txt` | density validation, 38/38 passing |
| `Bayesian_2B_fit.py` | hierarchical Bayesian two-boundary, ready to run |
| `two_boundary_readiness.py` | data audit and identifiability verdict |
| `identifiability_2b.py` | parameter recovery sweep |
| `real_2b_mle.csv` | the 40 two-boundary fits |
| `t0_decomposition.csv` | Wald all trials / Wald same trials / two-boundary |
| `loglik_decomposition.csv` | RT-shape vs choice-coupling terms per cell |
| `model_headtohead.csv` | per-cell log-likelihood comparison |
| `validity_check.csv` | fairness control |
| `t0_recovery.csv` | cross-fit recovery simulations |
| `fit_sv_all.csv` | drift-variability fits and AIC |
| `leadlag_prediction.csv` | predicted vs observed lead/lag RT gap |
