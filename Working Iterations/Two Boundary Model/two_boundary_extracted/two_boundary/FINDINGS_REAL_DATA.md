# Two-boundary DDM on the real data — findings

Run against the uploaded `pooled_data.csv` (7,676 rows) and the 16
`CMT*_MASTER_Summary.csv` files. The two sources agree exactly: same 5,760
interception trials, identical HandRT and GazeSRT medians per speed.

---

## Summary

| question | answer |
|---|---|
| Is there a per-trial binary outcome? | **Yes, for the hand.** `SignedError_deg` |
| Is it well enough balanced to fit? | **Yes.** 32–38% minority share per speed |
| Does the model fit? | **Yes.** 40/40 cells converged, no parameter at a bound |
| Does it buy anything over the Wald? | **Yes.** start point w = 0.403, p = 3e-8 vs 0.5 |
| Can the eye get the same model? | **No.** there is no gaze spatial variable |

---

## 1. The outcome variable

`SignedError_deg` is exactly `wrap(HandDir_deg − TargetDir_deg)`, verified to 1e-12
on all 5,760 interception trials.

**Positive = the hand leads the target. No per-direction sign flip is needed.**
`Direction` (Left/Right) labels which *side* of the workspace the target starts on,
not which way it travels. Both sides move counter-clockwise — circular means of
`TargetDir_deg` displace by +17.8° and +35.3° (Left) and +19.0° and +39.9° (Right)
at 75 and 150 deg/s. Same sense on both sides, so `sign(SignedError_deg)` is a
consistent lead/lag code across the whole dataset.

### The other three variants are not usable

| variant | complete | note |
|---|---|---|
| `SignedError_deg` | 83.2% | **use this one** |
| `SignedError_deg_HRT50` | 25.0% | different measurement time, see §5 |
| `SignedError_deg_HRTavg5` | 12.5% | identical to `SignedError_deg` where both exist (r = 1.000, 100% sign agreement) |
| `SignedError_deg_HRT50avg5` | 12.5% | identical to `_HRT50` (r = 1.000) |

So there are really only two distinct measures, and only one of them is populated
widely enough to use.

### `HasTgtReached` is not a usable outcome

Hit rates are 99.9% / 99.1% / 93.0% by speed. At 0 deg/s that is a 0.1% minority
share, an order of magnitude below the identifiability wall. It also conflates the
decision with motor execution. Do not use it.

---

## 2. Readiness audit — hand

5,760 trials → **4,736 usable** (interception, 150–800 ms, outcome present).
Overall split 0.589 lead / 0.411 lag.

| speed | cells | median n | median minority share | per-cell identifiable |
|---|---|---|---|---|
| 0 | 8 | 119 | 31.9% | 8 / 8 |
| 75 | 16 | 119 | 32.4% | 15 / 16 |
| 150 | 16 | 118 | 35.8% | 16 / 16 |

**39 / 40 cells clear the per-cell bar.** Only CMT010@75 falls short (93.3% lead)
and it will lean on the group in a hierarchical fit. 1,576 minority responses total.

This is comfortably clear of the identifiability wall — the recovery sweep put the
collapse below ~5% minority share and clean recovery above ~12%, and every cell
here is above 30%. **Per-cell fitting is viable, and hierarchical fitting is
comfortable.** This is a much better position than the saccadic t₀ situation.

---

## 3. Per-cell MLE on the real data

Two-boundary DDM fitted per participant × speed by differential evolution
(`real_2b_mle.csv`). All 40 cells converged.

| speed | n cells | v | a | w | t₀ (ms) | observed lead |
|---|---|---|---|---|---|---|
| 0 | 8 | −1.76 ± 1.28 | 0.53 ± 0.08 | 0.544 ± 0.070 | 185 ± 18 | 34.2% |
| 75 | 16 | +2.99 ± 1.73 | 0.56 ± 0.13 | 0.364 ± 0.041 | 180 ± 18 | 67.7% |
| 150 | 16 | +2.05 ± 1.45 | 0.63 ± 0.14 | 0.372 ± 0.057 | 169 ± 18 | 62.5% |

**Model adequacy.** Predicted P(lead) matches observed to 2.1% mean absolute error
(max 5.4%) across all 40 cells.

**No parameter is at a bound.** 0/40 cells at the 130 ms t₀ floor, 0/40 with w at a
bound, 0/40 with |v| at the cap. Contrast the saccadic single-boundary fits, where
t₀ pinned to the floor systematically. Nothing here is being held up by a constraint.

**The start point is real.** w = 0.403 ± 0.089, significantly below 0.5
(t = −6.89, p = 3.0e-8), range 0.292–0.711. This is the parameter the single-boundary
shifted Wald structurally cannot express, and it is not sitting at its neutral value.
Drift also reverses sign between 0 deg/s (−1.76, lag) and the moving conditions
(+2.99, +2.05, lead), which is only expressible because v is sign-free here.

**Non-decision time.** 185 / 180 / 169 ms, decreasing with speed — the same
direction as the published single-boundary values (~168 / 152 / 142 ms) but roughly
15–27 ms higher throughout. The 75-vs-150 contrast is not significant in this
per-cell MLE (Wilcoxon p = 0.117, n = 16); the hierarchical fit will have more power
and should be the version reported.

---

## 4. The eye — this is the problem

**There is no gaze spatial variable anywhere in the data.** The only gaze columns
are `GazeSRT_ms` and `GazeWindow_ms`, and the latter is a time (range −887 to
+663 ms), not a position. No saccade landing position, no gaze direction, no gaze
error.

So the two-boundary DDM can be fitted to the hand and **not** to the eye. That
directly undercuts the reason for dropping LATER — cross-effector comparability.

Three ways forward:

**A. Extract saccade landing position from the raw KINARM files.** The master
summaries are derived; if the raw gaze traces survive, a gaze direction at saccade
landing relative to target direction would give the eye exactly the same lead/lag
code as the hand, and the comparison would be genuinely symmetric. This is the real
fix and worth checking before choosing between B and C.

**B. Report the single-boundary Wald as the cross-effector comparison, and the
two-boundary hand fit as an additional analysis.** The Wald comparison stays exactly
as it is today, and the two-boundary fit answers the methodological question on its
own terms: does the interception task's spatial structure change the hand picture?
Given the results in §3 it does, and t₀ survives it. This is the safe option and
costs nothing already done.

**C. Fit the two-boundary model to the hand only and say so.** Weakest — it
abandons the comparability constraint without gaining anything B does not already
give.

**Recommendation: B, with A investigated in parallel.**

---

## 5. Open decision: when is hand direction measured?

This one matters and is not a data-quality issue — it is a modelling choice.

A DDM boundary is crossed at *movement initiation*. The choice the model predicts
is therefore the **initial movement direction**, not where the hand ended up. If
`HandDir_deg` is measured at interception, it includes online correction during the
reach, which is post-decision and does not belong in the likelihood.

The `_HRT50` variant appears to measure direction at a different (earlier) point,
and it gives a materially different answer:

| | 0 deg/s | 75 deg/s | 150 deg/s |
|---|---|---|---|
| `SignedError_deg` % lead | 34.2% | 67.4% | 62.2% |
| `SignedError_deg_HRT50` % lead | 40.8% | 89.0% | 82.7% |

They agree in sign on only 79.9% of trials. That is a large enough disagreement to
change the fitted drift and start point.

Two things needed here:

1. **What exactly does `HRT50` mean** — direction at 50% of the movement, at
   RT + 50 ms, at half the distance? Whoever wrote the extraction will know.
2. **Why is it only computed for 25% of trials**, and can it be computed for all
   of them? If it is the earlier measurement, it is arguably the more correct input
   and worth backfilling.

Until that is settled, `SignedError_deg` is the defensible default because it is the
one that is populated — but the choice should be stated explicitly in the writeup,
not left implicit.

---

## 6. Data issues to flag

**Eight participants have no direction data at 0 deg/s.** CMT003 through CMT010:
all 120 of their 0 deg/s interception trials have NaN in `HandDir_deg`,
`TargetDir_deg` and `SignedError_deg`. `HandRT_ms` is present (99% complete), so
this is a gap in the kinematic extraction, not missing trials.

Effect: the 0 deg/s condition drops from 16 cells to 8. That is the condition with
the most symmetric lead/lag split and the cleanest baseline (stationary target, so
any lead/lag is pure bias rather than prediction), so it is the most costly place to
lose half the sample. Worth checking whether it can be recovered — it looks
systematic rather than random, which usually means it is fixable.

**One participant label discrepancy.** `CMT002_MASTER_Summary.csv` contains
`BlockName` values reading `CMT019_INTER_*`. The other seven mismatches
(`CMT0011` → `CMT011` etc.) are just zero-padding and are cosmetic, but
CMT002 → CMT019 is a different number. Worth confirming the file is the participant
its filename claims.

---

## 7. How to run it

`pooled_data.csv` works directly — no merging needed, no column renaming.

```bash
python two_boundary_readiness.py --data pooled_data.csv --sign-col SignedError_deg --effector hand

python Bayesian_2B_fit.py --data pooled_data.csv --effector hand \
       --sign-col SignedError_deg --draws 1500 --tune 1500 --chains 4
```

The second one needs PyMC and will take a while. It writes group parameters with
credible intervals, per-cell estimates, posterior predictive checks, and the LOO
comparison against a shifted Wald × independent Bernoulli — which is the direct
test of whether coupling RT and direction through one diffusion earns its keep.
