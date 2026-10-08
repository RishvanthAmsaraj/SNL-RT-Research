# snl-rt-nextsteps

Response to the professor's two directives, built on the audited pipeline.

**Read `PROFESSOR_MEMO.md` first** — it explains what each analysis answers and where
the draft reply needs adjusting.

## Setup

```bash
cp /path/to/pooled_data.csv analyses/
python run_all.py
```

Requires numpy, scipy, pandas, matplotlib. **No PyMC.** Nothing here modifies your
existing pipeline; every script reads `pooled_data.csv` and writes new files into
`analyses/`.

## Layout

```
core/
  wald_fast.py    profile-likelihood shifted-Wald fitter (validated 80/80 cells)
  later.py        LATER family: plain (2p), shifted (3p), truncated; MLE + nested shift test
  ddm2.py         two-boundary DDM: Navarro-Fuss density, fitter, power analysis
  compare.py      fair non-nested comparison: Vuong, CV log-lik, per-model bootstrap KS

analyses/
  q1_direction_audit.py          is a two-boundary model possible at all?  [GATE]
  q2_later_vs_wald.py            the professor's directive, done fairly
  q3_twoboundary_feasibility.py  power analysis at your error rate
  q4_figures.py                  reciprobit, Q-Q, and the shift-LRT dissociation
```

## Order

q1 is a gate — it tells you whether a directional outcome exists and what the error
rate is. Feed that rate to q3. q4 needs q2's output.

```bash
python run_all.py                                # q1, q2, q4
python run_all.py --error-rate 0.06 --n 110      # adds q3
python run_all.py --quick                        # B=50, for a first look only
```

## Two things to know before reading the output

**A tie is the most likely result of the LATER-vs-Wald comparison.** On data generated
from a shifted Wald the two models are indistinguishable (Vuong p ≈ 0.8). Decide what
a tie means before you see the numbers.

**The per-cell shift LRT is underpowered at n ≈ 110.** Report the effector-level
Mann-Whitney and the median D, not the count of significant cells. On mock data the
counts looked weak (4/18 vs 2/18) while the effector-level test gave p = 0.00016.

## Validation

- Wald fitter: 80/80 real cells against the published differential-evolution fits,
  zero failures
- DDM density: lower/upper masses 0.0832 / 0.9168 against analytic choice
  probabilities, total 1.0000
- LATER MLE: closed form, checked against numerical optimisation
- Nested shift test: recovers t₀ > 0 on hand-like simulations, t₀ = 0 on saccade-like
