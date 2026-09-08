# Next steps — the professor's directives, and what the analysis actually shows

Two directives, plus the confirmed items from the code audit. This memo says what
each analysis does, what to expect, and where your draft reply to him needs adjusting.

---

## Directive 1 — "try the LATER model on the arm data and compare goodness of fit"

Right instinct, but the comparison as posed is unfair in three ways at once, and each
is fixable.

**1. Your LATER fit isn't a likelihood.** `LATER_analysis.py` fits by `linregress` on
reciprobit plotting positions — OLS on order statistics. That can't be compared to an
MLE-fit Wald by AIC, BIC, or a likelihood ratio. `core/later.py` fits by proper MLE,
which is closed form (`μ̂ = mean(1/RT)`, `σ̂ = sd(1/RT)`).

**2. The parameter counts don't match.** Plain LATER has 2 free parameters; the
shifted Wald has 3. A raw log-likelihood or KS comparison rewards the Wald for
flexibility it's entitled to. So the package also fits **shifted LATER**
(`RT = t₀ + 1/r`) — 3 parameters, like-for-like.

**3. The contamination conventions differ.** Your Wald likelihood carries a 5% uniform
contaminant; LATER as normally fitted doesn't. The contaminated Wald gets to explain
away outliers LATER must fit. Everything here uses pure likelihoods for both.

### The result you should prepare for: a tie

On data generated from a shifted Wald, the Wald and shifted LATER come out
**indistinguishable** — Vuong p ≈ 0.80, KS 0.0369 vs 0.0367. These two distributions
are genuinely hard to separate at n ≈ 110–160.

So "if it is better for LATER this is probably a good enough reason" most likely has
no trigger. **Decide now what a tie means**, before you see the numbers:

- A tie on the **hand** means the Wald's extra structure buys nothing there, and
  LATER-for-both becomes attractive on parsimony and comparability grounds.
- A tie on **saccades** you already found (KS ≈ 0.12 each). The case for LATER there
  was never about fit — it's that LATER has no non-decision parameter to floor.

### The test that's actually more informative

Shifted LATER **nests** plain LATER at `t₀ = 0`. That makes *"does this effector's RT
distribution demand a non-decision term?"* a nested likelihood-ratio test **inside the
LATER family**, with no Wald assumptions anywhere in it.

On simulated cells matching your parameters:

| cell | fitted shift | LRT D | p |
|---|---|---|---|
| hand, true t₀ = 165 ms | 93.1 ms | 5.12 | 0.012 |
| hand, true t₀ = 162 ms | 128.6 ms | 9.42 | 0.001 |
| saccade, true t₀ = 30 ms | 0.0 ms | 0.00 | 1.000 |
| saccade, true t₀ = 35 ms | 0.0 ms | 0.00 | 1.000 |

If that pattern holds on your data, **you've replicated the entire dissociation in a
second model family** — which is stronger evidence than any goodness-of-fit horse
race, and it comes free from the same fits.

One caveat the package handles: `t₀ = 0` sits on the parameter-space boundary, so the
null is a 50:50 mixture of a point mass and χ²₁ (Self & Liang 1987), not a plain χ²₁.
Using the naive χ²₁ roughly doubles the p-value.

### Run it at the effector level, not per cell

Per-cell LRTs at n ≈ 110 are underpowered. On mock data the per-cell counts looked
weak (4/18 hand vs 2/18 saccade) while the effector-level test was decisive:

```
median LRT D:  hand 1.205   saccade 0.000
Mann-Whitney (hand D > saccade D):  p = 0.00016
Fisher combined:  hand p = 1.4e-07    saccade p = 0.76
```

**Report the Mann-Whitney and the median D, not the count of significant cells.**
Counting would have led you to conclude the pattern doesn't replicate when it does.

### One correction to your reply

You told him a truncated (left-censored) rate distribution would probably be needed
for hand data because longer RTs push the Gaussian rate toward zero. The relevant
quantity is `μ/σ`, and on simulated data matching your parameters it runs **higher**
for hand than saccades:

| | μ/σ | P(rate < 0) |
|---|---|---|
| hand | 8.08 | 3.1e-16 |
| saccade | 5.31 | 5.5e-08 |

Both negligible; hand is safer than saccades, not riskier. The intuition that longer
RTs mean lower rates is right, but the rate SD scales down with them. `check_truncation()`
reports it per cell so you can confirm on real data rather than assume either way.

---

## Directive 2 — two-boundary DDM, and "I am skeptical there will be enough"

He's right on both halves, and the package makes the second half quantitative.

**Why errors matter structurally.** With an unbiased start, `P(error) = 1/(1 + exp(v·a))`.
The error rate *is* a statement about `v·a`. All the extra information a two-boundary
model has over a single-boundary one lives in the error RT distribution.

**Simulated recovery**, n = 110/cell, `t₀` = 160 ms, `w` = 0.5:

| P(error) | error trials | RMSE v | MAE t₀ | MAE w | fits converged |
|---|---|---|---|---|---|
| 0.02 | 2.4 | 0.475 | 5.9 ms | 0.046 | **9/12** |
| 0.05 | 4.8 | 0.331 | 8.8 ms | 0.037 | 11/12 |
| 0.10 | 11.6 | 0.282 | 11.1 ms | 0.032 | 12/12 |
| 0.20 | 24.0 | 0.298 | 7.0 ms | 0.027 | 12/12 |
| 0.30 | 32.4 | 0.195 | 8.9 ms | 0.024 | 12/12 |

Below ~5% error the fits start failing outright and `w` — the parameter *only* error
trials can identify — degrades by roughly double. His skepticism is correct at the
cell level.

**But per-cell infeasible is not project-infeasible.** At 16 participants × 110
trials, a 6% error rate gives ~106 pooled error trials per speed condition. A
**hierarchical** two-boundary model with participant random effects can be identified
from the pooled error distribution even when no single cell can — the same
partial-pooling argument that already justifies your hierarchical Wald. That's the
version worth attempting, and it's a concrete answer rather than a shrug.

**Diagnostic if you do fit it:** check whether the `w` posterior actually moves away
from its prior. If it doesn't, the error trials aren't informing the model and you
should report the single-boundary result.

**The gate first.** `q1_direction_audit.py` checks whether a directional outcome
exists in `pooled_data.csv` at all. If it doesn't, that's not the end — the professor
is asking about trials that head *initially* in the wrong direction, which is a
kinematic property, not a scored outcome. It's usually recoverable: find movement
onset, take the sign of lateral velocity over the first ~100 ms, compare to the
target's side. q1 tells you whether that trip back to the raw KINARM files is
necessary.

---

## The tension nobody has named yet

Your framing — "one model class across both modalities, or the best model per
modality?" — has a third consideration that matters more than either.

**Your project's central question is whether non-decision time is identifiable.**
LATER has no non-decision parameter. If you adopt LATER for both effectors, you no
longer have a t₀ to ask about, and the research question dissolves rather than gets
answered.

That's not an argument against the professor's suggestion — it's an argument for
running it as a **model comparison** rather than a **replacement**:

- If the Wald clearly beats LATER on hand but not on saccades, that's a beautiful
  result: hand RT distributions carry shape information that identifies t₀; saccade
  distributions don't. It's your central claim, restated as model selection.
- If they tie on hand, the honest reading is that your hand t₀ is identified but not
  *uniquely required* — worth saying plainly.
- Either way, the shift-LRT test gives you a t₀ question answerable inside the LATER
  family, so you don't have to choose between comparability and keeping the parameter
  you care about.

---

## Audit fixes folded in

The package carries these forward so the new analyses don't inherit old problems:

| From the audit | Status here |
|---|---|
| **E1** profile-likelihood fitter | `core/wald_fast.py`, validated 80/80 cells, zero failures |
| **A1** calibrated goodness of fit | every KS in `compare.py` gets its own bootstrap null |
| **A9** LATER's r² is a Q-Q correlation, not a fit statistic | reported for continuity, explicitly demoted |
| **A2** contamination mismatch | both models fitted under the same convention |
| **A7** boundary solutions invisible | `at_bound` / `v_at_cap` flags carried through |
| **S2** four different colour schemes | one palette in `q4_figures.py` |

Still outstanding and **unaffected by this work**: fix `express_mode` (it holds the
mean, not the mode — max 27 ms off), add the hand floor-control panel, add `ks_p` and
profile CIs to the fit tables.

---

## Running it

```bash
cp pooled_data.csv snl-rt-nextsteps/analyses/
cd snl-rt-nextsteps
python run_all.py                                  # q1, q2, q4
python run_all.py --error-rate 0.06 --n 110        # add q3 once q1 gives you the rate
```

numpy, scipy, pandas, matplotlib. No PyMC. q2 at B=200 is the slow step.

## Suggested reply to him

1. LATER on the arm is running, fit by MLE rather than reciprobit OLS, against both a
   like-for-like 3-parameter shifted LATER and the 2-parameter standard version.
2. Flag up front that a tie is the likely outcome and say what you'd conclude from it.
3. Lead with the nested shift test — it answers the t₀ question in a second model
   family and is a stronger result than the fit comparison he asked for.
4. On two-boundary: give him the recovery table and the pooled-error-trial count. It
   converts "I'm skeptical there will be enough" into a number, and it identifies the
   hierarchical version as the one worth attempting.
5. Ask whether initial movement direction was ever scored, or whether you should
   derive it from the raw kinematics.
