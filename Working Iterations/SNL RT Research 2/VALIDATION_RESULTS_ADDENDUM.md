# Validation run — results read, corrections, and what to do next

Your run against the real `pooled_data.csv` (7,677 rows). Two things I got wrong are
corrected below. Re-run with the updated suite before acting on V2 or V4.

---

## 1. The 3 FAIL cells were my bug, not a benign artefact

Your diagnosis of the cause was right — they are cap cells. Your conclusion was not.
**The gate did not clear.**

v1's M-step solved the weighted MLE in closed form and then *clipped* the result to
`v ≤ 20` / `a ≤ 2.5`. Clipping gives a feasible point, not the constrained optimum.
When a cell's unconstrained drift exceeds the cap, the true constrained optimum sits
on the boundary face with the *other* parameter re-optimised — and clipping does not
do that. So v1 could genuinely land worse than differential evolution.

Head-to-head on cells with true drift beyond the cap:

| cell | DE | v1 (clip) | v2 (constrained) | v1 − DE | v2 − DE |
|---|---|---|---|---|---|
| v=22 | −344.9271 | −344.9219 | −344.9271 | **+0.0052** | −0.0000 |
| v=28 | −363.0823 | −362.6612 | −363.0823 | **+0.4211** | −0.0000 |
| v=25, a=2.4 | −299.1995 | −299.1893 | −299.1995 | **+0.0102** | −0.0000 |
| v=35 | −428.3954 | −427.5112 | −428.3954 | **+0.8841** | −0.0000 |
| a=2.6 | −226.4085 | −226.4080 | −226.4085 | **+0.0004** | −0.0000 |
| normal | −323.4239 | −323.4239 | −323.4239 | −0.0000 | −0.0000 |
| normal2 | −417.7280 | −417.7280 | −417.7280 | −0.0000 | −0.0000 |

**v1 failed on 5/7. v2 fails on 0/7 and matches DE exactly.**

The fix keeps everything closed form — on the active face the free parameter still
has an analytic solution:

- `a` at cap: `v = a·W / Σ(w·τ)`
- `v` at cap: `S·a² − W·v·a − W = 0` with `S = Σ(w/τ)`

`wald_fast.py` v2 evaluates each candidate face and takes the best. No iterative
optimizer, no speed cost.

**Action:** replace `wald_fast.py` with v2 and re-run `v1_parity_check.py`. Expect
zero FAIL. If any survive, stop and send me the rows — that would be a different
problem.

Two notes on your V1 numbers while you're there. Your counts sum to 77 classified +
3 FAIL = 80, so "80/80 as expected" was 77/80. And **"2 cells profile BEATS DE" is
almost certainly rounding**, not a real improvement: your CSVs store `v` to 3 dp and
`t₀` to whole ms, so re-evaluating the NLL at the published values is always
fractionally worse. The script only counts gaps beyond 0.5 log-lik as meaningful —
check whether CMT005@150 and CMT007@150 clear that bar. If they don't, drop the claim.

---

## 2. V2 is the real finding, but B=50 cannot support the headline number

**Your reported "SRT median p = 0.049" is not an achievable p-value at B=50.**

With B replicates, `p = (count+1)/(B+1)`, so at B=50 the only possible values are
0.020, 0.039, 0.059, 0.078, … With 48 cells the median is the mean of the 24th and
25th, and 0.049 is exactly (0.039 + 0.059)/2. **The median is straddling the
threshold and you cannot tell which side it falls on.**

This matters because "half of SRT cells are formally rejectable" is the single most
consequential claim in the audit, and it currently rests on a quantisation artefact.

The memory failure is fixed. Two causes, both addressed:

- `scipy.stats.kstest` with a callable was ~6× the cost of a direct ECDF computation
  and dominated the loop. Replaced with a direct max-deviation calculation and a
  closed-form normal-based Wald CDF (agrees with scipy to **1.2e-15**).
- Results are now written **per cell**, so a run is resumable rather than
  all-or-nothing. Interrupt it and re-run the same command; it picks up where it
  stopped.

Measured: **6.7 s/cell at B=200 → ~11 min for 96 cells**, flat memory.

```bash
python v2_ks_calibration.py --B 200
```

**What I expect to hold at B=200:** the direction of the finding. Median SRT KS
~0.083 against a calibrated crit95 of ~0.088 means a large fraction of cells sit
near the boundary, which is why the count is sensitive to resolution. The
qualitative claim — the 0.10 threshold is too lax and a substantial share of SRT
cells are formally rejectable — will survive. The exact "24/48" will move.

**Three things to check in the new output:**

- **`rejected AND v_at_cap`.** The new script flags these separately. If a cell is
  rejected *and* its drift is pinned at 20, the rejection may be the **cap**, not the
  model. Refit those with a wider bound before concluding the Wald fails (audit A7).
  Given how many SRT cells hit the cap, I'd expect real overlap here.
- **`pinned at the p resolution`.** Cells at the minimum achievable p need more
  replicates to resolve.
- **The 5 cells that pass `ks < 0.10` but fail calibration** (CMT0014@0, CMT007@0,
  CMT0016@75, CMT0011@150, CMT0016@150). These are the cleanest demonstration that
  the threshold was wrong — a fit your current rule accepts, correctly calibrated, is
  rejected. Put them in the supplement.

**On HRT 3/48 rejected** — I said HRT should be clean, so look at these individually
rather than absorbing them. CMT005@150 and CMT007@150 also appear in your "profile
beats DE" list, which is worth noting: a cell that is both hard to fit and hard to
optimise is usually telling you something about the data, not the method.

**On the recurring participants** (CMT0012 ×3, CMT009, CMT003, CMT004): this is the
structural-heterogeneity story, not cell-level noise. CMT0012 has `express_frac` ≈
0.82 in `LATER_fits.csv`. A single Wald cannot fit a distribution that is 82% express
saccades — **the rejection is correct and informative**. Cross-reference every
rejected cell against `express_frac` before treating any as a fit failure. This
strengthens the case for the mixture and for hierarchical LATER (N3).

---

## 3. My r > 0.9 threshold in V4 was wrong — withdraw it

Method B is hierarchical, so per-cell estimates are **shrunk toward the group mean**.
Method A's are not. Correlating a shrunk estimator against an unshrunk one is
attenuated *by construction*. Setting 0.9 was a mistake.

Simulating 48 hand cells with true t₀ SD = 17 ms, Method A SE = 11.9 ms (implied by
your observed 46.5 ms median CI width), Method B SE = 6.9 ms, and **no real
disagreement whatsoever**:

```
mean r = 0.755    90% range = [0.638, 0.846]
expected mean |A − B| from noise alone = 11.0 ms
```

Your observed **r = 0.644** sits inside that range (6th percentile). Your observed
**mean |A − B| = 12.7 ms** against an 11.0 ms noise expectation is essentially exact
agreement. And **47/48 intervals overlap**.

**Read in the right order, V4 says Methods A and B agree well.** The revised script
now reports interval overlap first, then |A−B| against its noise expectation, then
correlation last with an explicit warning not to over-read it.

Your contamination hypothesis (A2) is still worth testing — it's the most likely
source of any *systematic* offset — but check the **signed** mean, which the updated
script now prints. A systematic offset means contamination; symmetric scatter around
zero means noise. r = 0.644 alone doesn't distinguish them.

The SRT r = 0.829 being *higher* than HRT is also expected: Method B's SRT t₀ is
nearly constant at 70–73 ms, so there's little for shrinkage to attenuate.

---

## 4. V3A and V3B replicate cleanly — two small refinements

**V3A.** Confirmed as predicted, including the 27 ms gap at CMT008@75. The one flip
is **CMT003@150 at mean = 136 / mode = 130 — exactly on the boundary**. Decide
whether the express criterion is `< 130` or `≤ 130` and write it down; a cell landing
on 130.0 is otherwise a coin flip. Since express counts move 1 → 2 on a 16-cell base,
this is a real change in a reported number.

**V3B — the control holds, and it's stronger than it looks.** Your SRT tracking rate
of 9/17 is depressed by a selection effect I should have flagged: the sweep can only
run on cells whose *fastest* RT clears the highest candidate floor (92 ms). Cells
failing that are the **fastest** cells — precisely the ones most likely to floor. So
**the true SRT tracking rate is higher than 9/17**. The updated script now reports
how many cells were skipped.

Report the median slopes (0.00 vs 0.97) as the headline; they're unaffected by the
selection and are the cleaner statistic. Also glance at your HRT 5/48 tracking cells
— 10% is more than I'd expect, and they may be cells with an unusually fast trial
dragging the `min(RT)` ceiling (audit A3).

---

## What to re-run

```bash
cp wald_fast.py <your validation folder>/     # v2 — replaces v1
python v1_parity_check.py                     # expect ZERO fail
python v2_ks_calibration.py --B 200           # ~11 min, resumable
python v4_intervals_and_AvB.py                # revised interpretation
```

V3 needs no re-run — the findings stand. Re-run only if you want the skipped-cell
count in the output.

## Revised status

| Check | Status | Blocking? |
|---|---|---|
| V1 profile fitter | **Was NOT clear.** Fixed in v2; re-run to confirm | Yes, until re-run |
| V2 KS calibration | Direction solid, headline number needs B=200 | Yes, for the number |
| V3A express_mode | Confirmed; decide `<` vs `≤` at 130 | No |
| V3B floor control | Confirmed; tracking rate understated | No |
| V4 A-vs-B | **Agreement is good.** My threshold was wrong | No |

Nothing here changes the audit's priority order. V3B is still the cheapest high-value
change, and A6 is still a wrong value in a shipped CSV.
