# SNL-RT-Research — full codebase audit

Accuracy and efficiency review across `Current Pipeline/Code/` and `kinarm-rt-app/`.
Findings marked **[verified]** were checked numerically; the scripts are in
`audit_checks/`. Everything else is a code reading.

Ordered by impact, not by file.

---

## Part 1 — Accuracy

### A1. The KS = 0.10 threshold is uncalibrated, and it is the load-bearing decision in the whole SRT analysis **[verified]**

`DDM_fit.py` selects the mixture when `ks_single > 0.10` and accepts it when `ks_mix < 0.10`.
That number appears to derive from the standard KS critical value, but the standard critical
value assumes the parameters are **known**. Here `(v, a, t₀)` are estimated from the same data,
which shrinks the null distribution substantially (the Lilliefors problem).

Simulating from a **true** single shifted Wald and refitting, 300 replicates per condition:

| | median KS | 90th pct | **95th pct (correct 5% crit)** | naive 1.358/√n | P(KS > 0.10 \| single true) |
|---|---|---|---|---|---|
| SRT-like, n = 110 | 0.064 | 0.083 | **0.093** | 0.130 | 0.013 |
| HRT-like, n = 160 | 0.056 | 0.073 | **0.081** | 0.107 | 0.000 |

Three consequences:

1. **The trigger is far too conservative.** `ks_single > 0.10` is a ~1.3% level test. Cells with a
   genuinely rejectable single Wald (KS 0.093–0.10) never get the mixture tried at all.
2. **The acceptance is too liberal.** Requiring `ks_mix < 0.10` from a 7-parameter model against a
   threshold that a 3-parameter model passes 98.7% of the time is not a real test.
3. **The reported KS values cannot be read.** A cell at KS = 0.08 is fine (≈85th pct of the null);
   a cell at 0.13 is rejected. Both currently sit under the same "OK / Poor" lines at 0.10 / 0.12
   in `DDM_figures.py` panel A.

Against the calibrated 0.093 cutoff, these cells in `DDM_srt_fits.csv` are **reported as `single`
with a formally rejected fit**: CMT0012@0 (0.1221), CMT009@0 (0.1317), CMT0012@75 (0.1304),
CMT0011@150 (0.106), CMT0014@0 (0.107), CMT009@150 (0.1584). Note CMT0012 and CMT009 recur —
CMT0012 is the express-dominant participant (`express_frac` = 0.82 in `LATER_fits.csv`).

**Fix.** Replace the fixed threshold with a per-cell **parametric bootstrap p-value**: simulate B
datasets from the fitted parameters, refit each, and locate the observed KS in that null. Report
`ks`, `ks_p`, and a `rejected` flag per cell. With the profile-likelihood fitter in Part 2 this
costs ~0.3 s/cell for B = 200. This turns "KS = 0.09, is that good?" into "p = 0.31, not rejected"
— which is what a reviewer will want, and it likely **improves** the story: the current framing of
SRT fits as "imperfect at KS 0.08–0.09" is too pessimistic. Those are near the null 90th
percentile, not failures.

For single-vs-mixture specifically, use the **bootstrap likelihood-ratio test** (McLachlan 1987).
The LRT null for mixture components is non-standard (boundary + non-identifiability), so the
bootstrap is the standard remedy. That replaces all three current heuristics (KS threshold,
`0.10 ≤ π ≤ 0.90`, `≥ 30 ms` separation) with one calibrated p-value.

---

### A2. Contamination is documented, implemented in Method A, and absent from Method B

`CODE_REFERENCE.md` states the likelihood is 0.95·Wald + 0.05·Uniform. `DDM_fit.py` implements
exactly that. `Bayesian_HRT_fit.py` and `hierarchical_single()` in `Bayesian_SRT_fit.py` use a
**pure Wald** `pm.Potential` with no uniform term.

This is not just a documentation mismatch. It interacts with A3 below, and it means the Method A
vs Method B comparison confounds two things at once: frequentist-vs-Bayesian **and**
contaminated-vs-pure likelihood. As a convergent-validity check that is weaker than it looks.

**Fix.** Add the uniform term to the Bayesian likelihood:

```python
Tr = rt.max() - rt.min()
tau = rt - t0[uidx]
logw = pt.log(a[uidx]) - 0.5*_LOG2PI - 1.5*pt.log(tau) - (a[uidx]-v[uidx]*tau)**2/(2*tau)
# trials below t0 contribute only the contaminant
logp = pt.switch(tau > 0,
                 pt.logaddexp(pt.log1p(-p_c) + logw, pt.log(p_c) - pt.log(Tr)),
                 pt.log(p_c) - pt.log(Tr))
pm.Potential("lik", logp.sum())
```

Then set `contamination = 0.05` in both methods and re-run. Keep the pure-Wald run as the
sensitivity check.

---

### A3. The Bayesian t₀ ceiling is `min(RT)` — the most fragile line in the model

Both hierarchical models use `t0 = FLOOR + (minrt − FLOOR)·sigmoid(z)`.

`min(RT)` is an extreme order statistic. It is exactly where contamination lives, its sampling
variability is large, and a single anomalous fast trial that survived the 80 ms / 150 ms filter
compresses the entire t₀ posterior for that unit. `DDM_fit.py` already uses the more robust
`percentile(rts, 3) − 2 ms` cap for the same quantity, so the two methods do not even agree on the
constraint.

The `*0.98` factor in `bayesian_mixture()`'s `t01`/`t02` is an unexplained magic number doing the
same job.

**Fix.** Once A2 lands, the hard ceiling is no longer required — sub-t₀ trials are absorbed by the
contaminant, so t₀ can be given a proper prior on `[floor, ∞)` rather than a data-dependent
sigmoid squeeze. If you want to keep the bounded form, switch the ceiling to the 3rd percentile so
it matches `DDM_fit.py`.

---

### A4. The hand model has no per-speed structure, and the shrinkage biases the speed effect *toward zero*

`Bayesian_HRT_fit.py` pools all 48 participant × speed units under one global prior
`z ~ N(μ_z, σ_z)`. Partial pooling therefore shrinks every unit toward the **grand mean across all
three speeds**. The speed effect is then read off the shrunken per-unit posterior means.

That is the wrong direction of conservatism: the reported 169.5 → 158.0 → 147.9 ms trend is an
**attenuated** estimate of the true speed effect, and there is no credible interval on any
between-speed contrast.

`kinarm_rt/models/hierarchical.py` already implements the correct structure (speed as a modelled
factor, participant random effects, group parameters per speed with CIs). It is currently filed
under "advanced analyses". **This should be the primary hand model.**

---

### A5. The dissociation is tested as two separate tests — this is the Gelman & Stern error

`stats_tests.dissociation_report()` runs Friedman + bootstrap + permutation separately for hand and
eye, and the conclusion is drawn from hand being significant (p ≈ 0.005) and eye not (p ≈ 0.74).

The difference between "significant" and "not significant" is not itself significant
(Gelman & Stern 2006, *Am. Stat.* 60:328). Three tests run in parallel on two effectors do not
constitute a test of the interaction. A methods reviewer will flag this.

There is a second problem: all three tests take **posterior means** as if they were raw data. That
discards the per-unit posterior uncertainty and ignores the shrinkage-induced correlation between
the estimates.

**Fix.** One joint model over both effectors with an `effector × speed` interaction on the t₀
linear predictor. The dissociation then becomes a single posterior contrast with a credible
interval — the claim you actually want to make, stated once, correctly. This also deletes
`stats_tests.py` from the critical path.

---

### A6. `express_mode` is the **mean**, not the mode **[verified]**

`DDM_fit.py`:

```python
express_mode = round((t0e + ae/ve)*1000)
reg_mode     = round((t0r + ar/vr)*1000)
```

`t₀ + a/v` is the mean of the shifted Wald. The mode is
`t₀ + μ·[√(1 + 9μ²/4λ²) − 3μ/2λ]` with `μ = a/v`, `λ = a²`.

Recomputed on real rows from `DDM_srt_fits.csv`:

| cell | reported "mode" | actual mode | error |
|---|---|---|---|
| CMT0012@150 express | 116 | 111 | 5.1 ms |
| CMT008@75 express | 175 | **148** | **27.0 ms** |
| CMT008@75 regular | 270 | 261 | 8.4 ms |
| CMT0012@150 regular | 214 | 205 | 9.9 ms |

Two downstream effects:

- The `≥ 30 ms` separation rule is applied to means. On CMT008@75 the mode separation is 113 ms vs
  94.5 ms by mean — a 20% difference in the quantity the rule tests.
- The app's "fast mode < 130 ms" express-territory column is computed from means. Since mean > mode
  for a skewed component, this **under-counts** genuine express cells. None of the four cells I
  checked flip, but CMT008@75 at 175 (mean) / 148 (mode) shows the margin is not always small.

**Fix.** Either rename the columns to `express_mean` / `reg_mean`, or compute the actual mode.
Given the express-saccade literature is about a modal peak near 100–130 ms, computing the mode is
the more useful choice. Whichever you pick, the separation rule and the < 130 ms test must use the
same quantity as the column name claims.

---

### A7. Boundary solutions at `v = 20` / `a = 2.5` are invisible in the output

From `DDM_srt_fits.csv`, cells sitting exactly on a cap: CMT0017@0 (`ve` = 20.0), CMT004@0
(`vr` = 20.0), CMT006@75 (`ve` = `vr` = 20.0), CMT004@75, CMT007@75, CMT010@150 (both = 20.0),
CMT001@75 (`a` = 2.5), CMT0012@75 (`v` = 20.0), CMT008@75 and CMT008@150 (`ar` = 2.5), CMT001@150
(`a` = 2.5). That is roughly a fifth of the SRT cells.

When `v` is at its cap the reported value is the bound, not an estimate — and because the Wald ties
`mean = a/v`, a capped `v` distorts `a` too. Any group-level mean of `v` that includes these cells
is biased, and the Method A vs Method B comparison inherits it.

The cap itself deserves scrutiny. Tran et al. (2020)'s `|v| ≲ 18.5` envelope is for the
**two-choice** DDM at s = 1. For a single-boundary Wald on saccades with a mean decision time near
100 ms you need `a/v ≈ 0.1`, so `v = 10a`; with `a ≈ 2` that is `v = 20` on the nose. The geometry
of fast saccadic cells demands drift values the borrowed envelope forbids.

**Fix.** (a) Add an `at_bound` flag to both fit CSVs and report the count. (b) Reparameterize as in
E2 below, which makes the constraint natural rather than a box corner. (c) Justify the cap for the
single-boundary case specifically, or widen it and document the change.

---

### A8. The Bayesian mixture uses a 40 ms floor while the singles use 70 ms

`Bayesian_SRT_fit.py`: `FLOOR = 0.040` for the mixture shifts, `FLOOR_PHYS = 0.070` for the
single-Wald t₀. So a mixture cell may place a component at 45 ms while the paper's central claim is
that 70 ms is the physiological minimum. That is internally inconsistent and hard to defend if
asked.

Separately, `bayesian_mixture()` has **no ordering constraint** — `pi ~ Beta(2,2)` is symmetric and
components 1 and 2 are exchangeable, so the posterior is genuinely label-switched. The frequentist
`_mix_nll` *does* enforce ordering (`if (t0e + ae/ve) >= (t0r + ar/vr): return 1e10`); the Bayesian
version does not. Post-hoc relabelling per draw is a workable patch but biases π toward 0.5 when
the components overlap, which is exactly the regime for the marginal cells.

**Fix.** Enforce ordering in the model — parameterize the regular component's mean as
`express_mean + positive_offset` — and use `FLOOR_PHYS` for both components unless there is a
stated reason not to.

---

### A9. LATER's headline r² is a Q-Q correlation, not a goodness-of-fit statistic

`LATER_analysis.py` fits by `linregress` on reciprobit **plotting positions** — OLS on order
statistics, not MLE. The reported median r² = 0.97 is then a Q-Q plot correlation, which is
extremely insensitive: badly non-normal data routinely exceed r² = 0.95 on a probit Q-Q plot
because the points are monotone by construction.

You already compute the right statistic in the same loop — `ks` against the fitted normal on
promptness — and its median is ≈ 0.12, a much less flattering and much more informative number.
But the figure caption and the abstract-level claim lead with the r².

**Fix.** (a) Fit LATER by MLE, which is closed form: `μ̂ = mean(1/RT)`, `σ̂ = sd(1/RT)`. (b) Report
a calibrated fit statistic — Shapiro–Wilk or Anderson–Darling on promptness, or the same
parametric-bootstrap KS p-value as A1. (c) Keep the reciprobit plot as the visual, which is what it
is good for. A reviewer who knows LATER will know the r² is decorative.

---

### A10. Method A reports no uncertainty at all — and the profile likelihood fixes that for free **[verified]**

`DDM_hrt_fits.csv` and `DDM_srt_fits.csv` report `v`, `a`, `t₀` as bare point estimates. Method A vs
Method B agreement is then assessed by eye against Method B's credible intervals.

Using the profile-likelihood machinery from Part 2, a 95% CI comes out of the same computation at
no extra cost (χ²₁ cutoff, Δ log-lik = 1.92). Simulated cells at realistic n:

| cell | true t₀ | fitted | 95% profile CI | width |
|---|---|---|---|---|
| hand, n = 160 | 165 | 142.1 | [130.0, 175.5] | 45.5 ms |
| hand, n = 160 | 162 | 169.7 | [130.0, 187.1] | 57.1 ms |
| eye, true 30 (below floor) | 30 | 70.0 | [70.0, 85.6] | 15.6 ms |
| eye, true 110 (above floor) | 110 | 111.9 | [70.0, 151.2] | 81.2 ms |

Two things worth internalising here. First, **per-cell frequentist t₀ carries ~±25 ms of
uncertainty even for the hand** — the point estimates in the CSVs are noisier than they look.
Second, that is precisely the argument for the hierarchical model: pooling across 48 cells is what
produces Method B's ~27 ms hand intervals. Reporting Method A CIs makes the hierarchical model's
contribution visible rather than assumed.

---

### A11. The identifiability sweep has no hand control **[verified]**

`SRT_identifiability_check.py` sweeps the floor over 40–90 ms for SRT cells and reports the slope
of fitted t₀ vs imposed floor. There is no equivalent run on HRT cells. The obvious reviewer
question — "does hand t₀ track *its* floor too?" — is currently unanswered by the diagnostic that
is meant to answer it.

Running the identical sweep on hand cells (floors 90–140 ms, bracketing the 130 ms floor):

| | slopes | median | tracking (slope > 0.7) |
|---|---|---|---|
| HRT cells | 0.00, −0.00, 0.00, 0.44, 0.00, −0.00 | **0.00** | **0 / 6** |
| SRT cells | 1.00, 0.19, 1.00, 0.00, 1.00, 0.85 | **0.93** | **4 / 6** |

This is a clean, decisive negative control and it is about ten lines of code. It converts a
one-sided observation into a dissociation. I would add it before anything else in this section.

---

### A12. Smaller accuracy items

- **`goodness_of_fit()` in `kinarm_rt/diagnostics.py` silently drops sub-t₀ trials** and recomputes
  the ECDF on the survivors (`ok = tau > 0`; `Fhi = arange(1, n+1)/n` with `n = ok.sum()`). That
  makes the fit look better than it is. Sub-t₀ trials should count as model failures (F = 0), not
  vanish. Never triggers in the original scripts because t₀ < min(RT) by construction, so this is
  an app-only divergence.

- **The RT-filter truncation is not in the likelihood** — you filter HRT to [150, 800] and SRT to
  [80, 600] ms, then fit an untruncated Wald. The correct likelihood divides by
  `F(hi − t₀) − F(lo − t₀)`. **[verified]** For typical cells this is numerically nil (Δt₀ < 0.05 ms).
  For a slow cell where the 600 ms cutoff actually clips the tail it moved t₀ by +7.9 ms and v by
  −0.20. So: not a crisis, but it is a one-line addition and a one-sentence robustness result you
  want in your pocket rather than discovered by a reviewer. The cells to check are the low-`v`,
  high-`a` ones (CMT008@75, CMT008@150).

- **`LATER_analysis.py` maps speed from `SpeedCode`** (`SPMAP = {1:0, 2:75, 3:150}`) while every
  other script reads `Speed_deg_per_s` directly. If those ever disagree in the data, LATER silently
  diverges from everything else. Pick one.

- **`warnings.filterwarnings("ignore")` at the top of nearly every script** hides exactly the
  overflow/divide warnings that would have surfaced A7. Suppress specific categories instead.

---

## Part 2 — Efficiency

### E1. Replace differential evolution with a 1-D profile likelihood — ~16× faster and *more* accurate **[verified]**

This is the single biggest win and it is not a tradeoff.

The current per-cell fit runs `differential_evolution` twice (seeds 42 and 7), maxiter 400,
popsize 12 over 3 parameters ≈ 29,000 NLL evaluations per cell, ~96 cells, plus 7-parameter
mixture fits at popsize 14.

But the Wald has a **closed-form MLE given t₀**. With `τᵢ = RTᵢ − t₀`:

```
μ̂ = mean(τ)                                  a = √λ̂
1/λ̂ = mean(1/τᵢ) − 1/μ̂                        v = a/μ̂
```

(Tweedie 1957 — verify the mapping: expanding `−(a − vτ)²/2τ` against the inverse-Gaussian
exponent gives `λ = a²`, `μ = a/v`, and the prefactors match.)

So the 3-D global search collapses to a **1-D search over t₀** on a smooth profile. The 5%
contamination has no closed form, but it yields to EM: E-step computes responsibilities, M-step is
the same closed form with weights. Nest the EM inside the 1-D search.

Verified against the current DE fitter on five simulated cells spanning HRT-like, SRT-like, and
fast-saccade parameters:

| cell | method | v | a | t₀ (ms) | NLL | sec |
|---|---|---|---|---|---|---|
| HRT-like | DE | 9.980 | 0.9619 | 155.6 | −325.414849 | 0.262 |
| | profile | 9.980 | 0.9619 | 155.6 | −325.414849 | **0.016** |
| SRT-like | DE | 12.752 | 1.6770 | 70.0 | −235.379822 | 0.352 |
| | profile | 12.752 | 1.6770 | 70.0 | −235.379822 | **0.023** |
| SRT-fast | DE | 15.884 | 1.5066 | 78.9 | −225.093571 | 0.246 |
| | profile | 15.884 | 1.5068 | 78.9 | −225.093571 | **0.015** |

Identical to 4 decimals on every cell; 16× faster overall. And it is *more* accurate in three ways:
the M-step is exact rather than a stochastic global search, the result is deterministic (no seed
dependence, so the dual-seed retry becomes unnecessary), and the profile curve is a byproduct.

**What the profile curve buys you, for free:**

- **Confidence intervals** for t₀ (A10) — Δ log-lik = 1.92.
- **The identifiability diagnostic done properly.** `SRT_identifiability_check.py` currently refits
  at 6 floors × ~34 cells × 2 seeds = 408 DE runs. The constrained optimum at *every* floor can be
  read off **one** profile curve per cell. The whole sweep becomes sub-second — and more to the
  point, the shape of the profile *is* the identifiability answer: a profile still descending at
  the floor means t₀ is unidentified, full stop, without a 0.7 slope heuristic.
- **`SRT_fixed_t0_analysis.py` needs no optimizer at all.** With t₀ fixed, `(v, a)` is the closed
  form. Instant.
- **The parametric bootstrap in A1 becomes affordable** — B = 200 refits per cell at 16 ms each.

A working implementation is in `wald_fast.py` alongside this document.

*Caveat, stated plainly:* your parity constraint means you cannot simply swap this in. Run it
side-by-side against the DE fitter on all 96 real cells first and confirm agreement to your
tolerance. Agreement was exact on simulated data, but real cells hitting the `v`/`a` caps may
differ, because the caps are applied inside the M-step here versus as box constraints in DE.

---

### E2. Reparameterize `(v, a)` → `(μ, λ)` — fixes the sampler *and* the boundary problem

This is the change I would make to the Bayesian models, and it is a pure reparameterization: the
likelihood is unchanged, so it is not a modelling decision you have to defend.

The Wald's mean is `μ = a/v` and shape is `λ = a²`. The data pin down the mean tightly and the
shape loosely, so the posterior has a strong ridge along `a/v = const`. In `(v, a)` coordinates that
ridge lies at 45°, so NUTS needs small step sizes and many leapfrog steps. Worse, the box
constraints `0.1 ≤ v ≤ 20` and `0.05 ≤ a ≤ 2.5` cut **across** the ridge — which is precisely why
a fifth of the SRT cells land in a corner (A7).

Sampling in `(log μ, log λ)` gives a near-spherical posterior, and the constraints become natural:
`μ ∈ (0, min_RT − floor)` is a real physiological statement, and `v ≤ 20` translates to
`√λ/μ ≤ 20`, which almost never binds.

Expect a meaningful reduction in leapfrog steps per draw, fewer divergences at `target_accept =
0.95`, and — the point — group-level `v` means that are not contaminated by corner solutions.
Verify parity by transforming the posterior back to `(v, a)` and comparing.

### E3. NUTS backend — and this may solve your Windows packaging problem

Two drop-in alternatives to PyMC's default sampler:

- `pm.sample(nuts_sampler="nutpie")` — Rust NUTS with better mass-matrix adaptation, typically
  2–10× on models of this shape. The adaptation quality matters *specifically* for the correlated
  ridge in E2.
- `pm.sample(nuts_sampler="numpyro", chain_method="vectorized")` — compiles the model to JAX and
  runs all chains as one vectorized computation. Well suited here because the likelihood is a
  single large vectorized `Potential` over ~5,000 trials.

The second point is the interesting one for you. Your model uses only `log`, `exp`, `sigmoid`,
`**2`, `logsumexp`, and `stack` — all JAX-compilable. Running under the JAX backend **bypasses
PyTensor's C backend entirely**, which means no C++ compiler. If that holds up in testing, a
pip-only `pymc + numpyro + jax[cpu]` install works on Windows, and the desktop build no longer
needs to ship a conda-pack environment with a compiler in it. Given how much effort that packaging
has absorbed, this is worth an afternoon of testing before anything else in this section.

I have not verified this one — no PyMC in this environment. Treat it as the highest-value
experiment, not a finding.

### E4. Vectorize the resampling tests

`stats_tests.permutation_test` calls `friedmanchisquare` 5,000 times inside a Python loop with
`np.apply_along_axis(rng.permutation, 1, mat)` on a 16×3 matrix. `bootstrap_speed_effect` does
5,000 separate `rng.choice` calls.

```python
# bootstrap: one line, ~100x
idx = rng.integers(0, n, size=(n_boot, n))
boot = diff[idx].mean(axis=1)
```

For the permutation test, permuting within rows of a 16×3 matrix is permuting the ranks {1,2,3}
per row — precompute the 6 permutations, index into them for all 5,000 draws at once, and compute
the Friedman statistic vectorized from the rank sums. Both become instant. Note this matters less
if you adopt A5, which removes these tests from the critical path.

### E5. If you keep differential evolution anywhere

- `vectorized=True` (SciPy ≥ 1.9) evaluates the whole population in one numpy call. Bigger win than
  multiprocessing for a cheap likelihood like this, since there is no process overhead.
- `workers=-1` parallelizes across cores.

**Both force `updating='deferred'`, which changes results relative to the current default
`updating='immediate'`.** That is a parity break, not a free speedup — the current results depend
on within-generation evaluation order. Flagging because it is an easy trap.

### E6. Minor

- `pipeline.py` and `app.py` both call `wald.mle_preview(kept, eff, ...)` again *after* a completed
  Bayesian fit (`res["preview"] = ...`). That is a full per-cell MLE pass duplicating one already
  run. Cache it.
- `LATER_analysis.py` and `why_saccadic_t0_floors.py` call `fm.findSystemFonts()` (a full
  filesystem scan, ~1–2 s) while `DDM_figures.py` and `vincentile_figures.py` check
  `fm.fontManager.ttflist`. Different methods can resolve to different fonts — see S2.
- `gaussian_kde` is recomputed across figures on the same pooled arrays. Cache per (speed,
  effector).

---

## Part 3 — Methods worth building for this specific problem

### N1. Profile likelihood as unified infrastructure

Covered in E1/A10/A11, but worth stating as a design principle rather than an optimization: for
this model, **fitting, identifiability, and interval estimation are the same computation**. One
profile curve per cell gives the MLE, the CI, the floor-tracking answer at any floor, and the
fixed-t₀ sensitivity. Four scripts currently compute overlapping subsets of this with four separate
optimizer setups. Consolidating them is both faster and less surface area for drift.

### N2. One joint model for the dissociation

Per A5. `effector × speed` interaction on t₀, participant random effects, both effectors in one
fit. The headline claim becomes one contrast with a credible interval instead of six tests and an
informal comparison. This is the change most likely to matter to a reviewer.

### N3. Hierarchical LATER as the primary saccade model

Your own `ISSUES_AND_IMPROVEMENTS.md` lists this under future work, and your findings argue for it:
LATER matches the Wald on raw goodness of fit, has no non-decision parameter to floor, and the
flooring is a shape property that more data cannot fix. Currently LATER is fit per-cell by OLS on
plotting positions with no pooling and no uncertainty.

A hierarchical LATER (participant-level rate and threshold, group means per speed) would be maybe
40 lines given the PyMC infrastructure already present, and it would let you report saccades in
their native model with credible intervals instead of reporting a floored parameter and explaining
why. Priors: median latency ≈ 200 ms implies a rate mean near 5 s⁻¹, SD 1–2 s⁻¹.

The two-component saccade cells fold in naturally as a LATER mixture (a second reciprocal-normal
population), which is closer to how the express-saccade literature models them than a second Wald.

### N4. Posterior predictive KS

The Bayesian analogue of A1, and cheaper because the draws already exist: for each posterior draw,
simulate a dataset, compute the KS, and locate the observed KS in that distribution. Self-
calibrating, no bootstrap refitting, and it gives a per-cell posterior predictive p-value that
slots straight into the existing goodness-of-fit table.

### N5. Simulation-based calibration

You have a parameter-recovery study, which is the right instinct. SBC (Talts et al. 2018) is the
rigorous form: draw θ from the prior, simulate, refit, check the rank of the true θ within the
posterior is uniform across replicates. It catches model/sampler bugs that point-recovery misses,
and it is becoming expected in computational modelling papers. Modest work given the recovery
harness already exists.

### N6. Trial-level covariates

Filed as low priority in your roadmap, but nearly free now that the hierarchical machinery exists,
and it pre-empts a specific reviewer question: *were the speed conditions blocked?* If so, "hand t₀
decreases with speed" and "hand t₀ decreases over the session" are confounded. A trial-index
regressor on t₀ (or on log v) settles it in one fit. You already carry a `trial` column for the
vincentile computation.

---

## Part 4 — Structural issues that create ongoing risk

### S1. Four independent copies of the Wald likelihood

`wald_pdf` / `wald_cdf` are defined separately in `DDM_fit.py`,
`SRT_identifiability_check.py`, `SRT_fixed_t0_analysis.py`, and `kinarm_rt/models/wald.py`.

Your parity constraint says the app must be byte-identical in statistical logic to the scripts.
Maintaining that across four copies means auditing forever; `tests/test_parity.py` currently
asserts that four implementations agree rather than testing one.

**Fix.** Extract a small `snlrt` package — likelihood, fitters, constants, filters — and have both
the scripts and the app import it. The scripts stay runnable standalone; parity stops being
something you re-audit and becomes something that cannot break. This is the change that makes the
rest of the roadmap sustainable.

### S2. Four different colour schemes across the figures

| file | 0 deg/s |
|---|---|
| `DDM_figures.py`, `SRT_fixed_t0_analysis.py` | `(0.30, 0.55, 0.20)` |
| `LATER_analysis.py`, `kinarm_rt/_speeds.py` | `(0.45, 0.68, 0.40)` |
| `vincentile_figures.py` | fill `#cfe8cf`, line `#4a7c59` |
| documented palette | RGB 191/230/191 with 0.55× for lines |

Four schemes, none matching the documented one. In a paper these figures sit next to each other and
the reader will notice. Same story for fonts (S2 in E6) — two different Arial-detection methods can
resolve differently on the same machine.

**Fix.** One `figstyle.py` with the palette, font setup, and rcParams, imported everywhere.

### S3. Resumability has no schema guard

`done(path)` reads `(pid, spd)` pairs from an existing CSV and skips them. If you change the output
columns between runs, a resume appends rows with a mismatched schema to the same file. Row-by-row
`mode="a"` append also means a crash mid-write leaves a partial line. Add a header/version check on
resume and write via a temp file.

---

## Suggested order of work

Roughly by (impact × confidence) ÷ effort:

1. **A11** — hand control on the identifiability sweep. Ten lines, verified, strengthens the central claim.
2. **A6** — fix or rename `express_mode`. Trivial, and it is currently wrong in a published CSV.
3. **E1 / N1** — profile-likelihood fitter, validated against DE on all 96 real cells. Unlocks 4, 5, 6.
4. **A1** — bootstrap KS p-values and the bootstrap LRT for mixture selection.
5. **A10** — profile CIs on the Method A tables.
6. **A2 + A3** — contamination in the Bayesian likelihood, drop the `min(RT)` ceiling.
7. **E3** — test the numpyro backend; if it works, the packaging problem gets much smaller.
8. **A4 + A5 / N2** — promote the per-speed model, joint dissociation model.
9. **S1** — extract the shared package.
10. **A7, A8, A9, A12, S2, S3** — cleanup.

Items 1, 2, 3, 5 and the S-series are low-risk. Items 6 and 8 change reported numbers, so each needs
a documented before/after.

---

## Honest limitations of this audit

- I read the scripts and the app modules through the project knowledge; I did not execute anything
  against the real `pooled_data.csv`. The verified findings were checked on simulated data with
  parameters taken from your published fit tables.
- The KS calibration (A1) and the profile-likelihood equivalence (E1) are the findings I am most
  confident in — both were checked directly, and both are properties of the model rather than of
  your data.
- E3 (numpyro/nutpie) is unverified and could fail on a PyTensor/JAX incompatibility. It is the
  highest-upside experiment here, not a result.
- A4/A5 are methodological judgements, not bugs. The current analysis is defensible; I am arguing
  the joint model is more defensible and more powerful. Reasonable people could disagree about
  whether it is worth the re-analysis.
- None of this overturns your results. The flooring diagnosis, the hand/eye dissociation, and the
  shape-based explanation all look sound. These are refinements to how tightly they are argued.
