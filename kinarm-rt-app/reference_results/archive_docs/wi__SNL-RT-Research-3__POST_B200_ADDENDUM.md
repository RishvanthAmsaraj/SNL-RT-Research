# Post-B=200 addendum — three things still open

The gate does clear and v2 is safe to adopt pipeline-wide. Three items below, in
descending order of how much they'd cost you if left alone.

---

## 1. The `-inf` cells are NOT rounding — and the explanation matters

I said rounding; the numbers say otherwise. Simulating 400 HRT-like cells, the number
where the rounded t₀ exceeded min(RT) was **0/400**. Rounding to whole ms cannot
normally produce this, so my addendum was wrong to file it as harmless.

`nll_at()` returns `inf` by exactly three routes:

| route | cause | fix |
|---|---|---|
| **A** | published t₀ ≥ min(RT) | store t₀ with more decimals |
| **B** | `v`, `a`, or `t₀` is NaN in the published row | that cell never fitted — drop or mark it |
| **C** | the loader and `DDM_fit.py` selected **different trials** | **stop and check every cell** |

Route C is the one that would hurt. If `DDM_fit.py` drops rows the validation loader
keeps — say it requires a paired non-NaN saccade for the same trial — then DE saw a
smaller, higher-`min(RT)` subset. Demonstration with a 4% trim:

```
DE's subset      n=153  min=193.97 ms  ->  NLL = 20.20   (feasible)
loader's fuller  n=160  min=181.90 ms  ->  NLL = inf     (infeasible)
```

Same published t₀, feasible for one trial set and not the other. That is the only
route that produces `-inf` on a cell whose published row is complete and whose t₀ sits
well inside the box — which is what CMT005@150 and CMT007@150 look like.

**If it is route C, your V1 result rests on an unchecked assumption for all 80 cells,
not just these two.** The discriminating test is one line:

```python
m = par.merge(pub[['pid','spd','n']], on=['pid','spd'], suffixes=('_loader','_pub'))
print(m[m.n_loader != m.n_pub])       # must be EMPTY
```

`v5_resolve.py` does this automatically and names the route.

**A detail worth chasing either way:** CMT005@150 and CMT007@150 are also two of your
three HRT rejections in V2. A cell whose t₀ is pushed hard against its upper bound is
a cell where the Wald cannot reach the fast trials — so **the ceiling behaviour and
the KS rejection are plausibly the same phenomenon**, not two coincidences. If route C
is ruled out, look at those two cells' leading edges directly. That is a finding about
your data, and it connects to audit A3.

---

## 2. 28/48 needs a multiple-comparison correction before it goes in a paper

48 cells at α = 0.05 gives ~2.4 expected false rejections under a true model, so 28 is
far beyond chance and the finding is real. But the **uncorrected count overstates**,
and this is the first thing a methods reviewer will ask for.

On a p-vector reconstructed to match your summary (48 cells, 28 below 0.05, median
0.032, 12 pinned):

```
uncorrected p < 0.05 : 28/48
BH FDR q = 0.05      : 17/48   (adaptive cutoff p <= 0.0175)
BH FDR q = 0.10      : 29/48   (adaptive cutoff p <= 0.0520)
```

Illustrative, not your actual numbers — `v5_resolve.py` computes it on your real
p-values. The claim survives either way; you just want the defensible number in the
manuscript rather than the raw count.

**One trap the script now guards against.** BH requires `p(k) ≤ (k/m)·q`, so the
tightest threshold is `q/m`. If your bootstrap p-floor sits above where cells actually
rank, BH can reject *nothing* no matter how strong the evidence — a resolution
artifact masquerading as a null result. At B=200, m=48, the floor is 0.005 and BH's
threshold at rank 12 is 0.0125, so you are fine. At lower B you would not be, and the
script warns rather than silently reporting zero.

---

## 3. Your proposed B=400 run would spend an hour and change nothing

Cells pinned at `p = 1/(B+1)` had **zero** of 200 null draws exceed the observed KS.
Simulated:

| true p | B | mean p | **P(p ≥ 0.05)** |
|---|---|---|---|
| 0.001 | 200 | 0.0060 | **0.0000** |
| 0.005 | 200 | 0.0099 | **0.0000** |
| 0.010 | 200 | 0.0147 | **0.0000** |

A pinned cell cannot cross 0.05 at any B. Refining it polishes a number already
decisively below threshold.

The cells that *do* need replicates are the ones **near** the threshold, where Monte
Carlo error is comparable to the distance from 0.05:

| true p | B=200 SE | P(misclassified vs 0.05) | B=2000 SE |
|---|---|---|---|
| 0.03 | 0.0119 | **7.7%** | 0.0038 |
| 0.08 | 0.0191 | **3.7%** | 0.0060 |

Your median SRT p is 0.032 — right in that band — so a meaningful number of cells sit
where the verdict is genuinely uncertain.

```bash
python v5_resolve.py --refine --B 2000        # p in [0.015, 0.12] only
```

At your measured ~15.6 s/cell at B=200, that's ~2.6 min/cell at B=2000; for ~20
borderline cells, roughly 50 minutes. Same budget as your proposal, spent where it can
actually move a decision. (A cell whose true p is exactly 0.05 stays a coin flip at
any B — that's inherent, not a resolution problem.)

---

## 4. Two smaller corrections to my own previous notes

**The SRT +3.8 ms offset is the right sign, but "systematic" overstates it.**
Simulating the A/B contamination mismatch: mean signed difference **+5.58 ms**, but
median **0.00** and only **41% of cells positive**. The mean is driven by a minority
with large shifts, not a uniform displacement. So the contamination fix (A2) will
likely move a *subset* of cells rather than shifting everything by ~4 ms. Check the
distribution of `AB_diff_ms`, not just its mean.

**Your SRT A-vs-B agreement is weaker evidence than the HRT agreement, not stronger.**
When true t₀ is below the floor, simulation gives: A floors 103/120, B floors 100/120,
**both floor 96/120**. Where both are pinned at 70 ms, "intervals overlap" is
guaranteed and carries no information — you are comparing two constants. So your
32/32 SRT overlap and r = 0.824 are largely structural, and r = 0.824 is driven by
whichever minority escapes the floor. **HRT's 47/48 overlap with r = 0.641 is the
stronger result**, despite the lower correlation. Before presenting SRT convergent
validity, report how many cells have both estimates at their floor.

---

## On `<` vs `≤` at 130 ms

Check the **unrounded** mode for CMT003@150 first — if it is 129.6 or 130.4 the
question dissolves. If it really is ~130.0, don't settle it by picking an inequality.
A hard cut deciding a cell that sits exactly on it is arbitrary in either direction,
and a reviewer can see that. Bootstrap a CI on the mode and report the cell as
`130 [CI]`, borderline. That is both more honest and more defensible than a
convention you'd have to justify.

---

## What to run

```bash
python v5_resolve.py                       # diagnose + BH, seconds
python v5_resolve.py --refine --B 2000     # borderline cells only, ~50 min
```

## Revised status

| Item | Status |
|---|---|
| V1 profile fitter | Clear **if** the `-inf` cells are route A or B. Route C reopens it. |
| V2 direction | Solid. 58% raw; report the BH count as the headline. |
| V2 borderline cells | ~20 cells genuinely uncertain at B=200 |
| V3A / V3B | Confirmed. Handle the 130 ms cell with a CI, not an inequality. |
| V4 HRT | Good agreement — the stronger of the two |
| V4 SRT | Largely vacuous while both methods floor; quantify before claiming it |
