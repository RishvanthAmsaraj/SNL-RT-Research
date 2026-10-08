# Migration Notes: Deprecated Ver 1 → Ver 2

## Overview

This document explains the transition from the initial PyDDM-based prototypes
to the first production-ready pipeline (Ver 2). Ver 1 served as a proof of
concept; Ver 2 introduced the core methodology still used today.

---

## 1. From PyDDM to Native Implementation

### Ver 1 Approach
```python
from pyddm import Model, Fittable, Sample
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant

model = Model(
    name="DDM with Mixture",
    drift=DriftConstant(drift=Fittable(minval=-3, maxval=3)),
    noise=NoiseConstant(noise=1),
    bound=BoundConstant(B=Fittable(minval=0.3, maxval=3)),
    overlay=OverlayChain(overlays=[
        OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
        OverlayPoissonMixture(pmixturecoef=Fittable(minval=0.0, maxval=0.2))
    ])
)
model.fit(sample, lossfunction=LossRobustBIC)
```

**Problems:**
1. **External dependency:** PyDDM must be installed; version conflicts possible
2. **Limited customization:** Cannot easily modify likelihood for specific needs
3. **Black box:** Internal optimization opaque; hard to debug failures
4. **Performance:** Slower than native scipy on large datasets
5. **Two-boundary model:** Designed for choice tasks, not interception

### Ver 2 Solution
```python
from scipy.optimize import differential_evolution

def neg_ll(params, rts, T_range):
    v, a, t0 = params
    adj = rts - t0
    wald = (a / np.sqrt(2*np.pi*adj**3)) * np.exp(-(a - v*adj)**2 / (2*adj))
    mixture = (1 - P_CONTAM) * wald + P_CONTAM / T_range
    return -np.sum(np.log(mixture))

res = differential_evolution(neg_ll, bounds, seed=42, maxiter=1500)
```

**Advantages:**
- No external dependencies beyond scipy/numpy
- Full control over likelihood (contamination, bounds, constraints)
- Transparent optimization (can inspect intermediate results)
- Faster execution
- Single-boundary Wald model appropriate for interception

---

## 2. From Synthetic to Real Data

### Ver 1 Approach
```python
# Synthetic data generation
num_trials = 2000
drift_rate = 0.50
upper_boundary = 1.0
lower_boundary = -1.0
# ... simulate random walk
```

**Problems:**
- No connection to actual KINARM data
- Fixed parameters; no validation against ground truth
- Cannot assess model fit quality on real distributions

### Ver 2 Solution
```python
# Real data loading
df = pd.read_csv('pooled_data.csv')
df_i = df[df['BlockType'] == 'I'].copy()
rts = df_i[(df_i['Participant'] == pid) & 
           (df_i['Speed_deg_per_s'] == spd)]['HandRT_ms'].values / 1000.0
```

**Advantages:**
- Direct analysis of experimental data
- Empirical validation of model fit (KS statistics)
- Participant-level parameter variation observable

---

## 3. Model Structure: Two-Boundary vs Single-Boundary

### Ver 1 Approach
Two-boundary DDM (correct/incorrect choices):
```
Evidence accumulates from z toward +a (correct) or -a (incorrect)
```

**Problem:** Interception tasks involve a single decision ("go"), not a binary
choice. The two-boundary model is inappropriate and overparameterized.

### Ver 2 Solution
Single-boundary Wald (first-passage time):
```
Evidence accumulates from 0 toward +a (response threshold)
RT = t₀ + first-passage time
```

**Advantages:**
- Correct model for go/no-go and interception paradigms
- One fewer parameter (no starting point z)
- Directly interpretable: drift = evidence accumulation rate

---

## 4. From Basic to Publication-Quality Visualization

### Ver 1 Output
- Simple histograms with matplotlib defaults
- Trajectory plots with basic styling
- No vector output (PDF)

### Ver 2 Output
- Vector PDFs with editable text (`pdf.fonttype=42`)
- Condition-specific color scheme (green/red/blue)
- Arial font throughout
- Proper figure sizing for publications
- Annotated conceptual schematics

---

## 5. Statistical Validation

### Ver 1: None
No goodness-of-fit metrics. Model fit assumed valid if optimization converged.

### Ver 2: Comprehensive
- **KS statistic:** Kolmogorov-Smirnov test for distribution match
- **Quality labels:** Good (<0.05), OK (<0.10), Borderline (<0.12), Poor (>0.12)
- **Per-participant diagnostics:** Heatmap showing exactly who is hard to fit
- **Mixture validation:** Separate PDFs for express vs regular components

---

## 6. Code Organization

### Ver 1 Structure
```
Deprecated Ver 1/
├── DualChoice(PyDDM).py          # PyDDM dual choice
├── DualChoice(NonPyDDM).py       # Native dual choice
├── DualChoice(SyntheticPyDDM).py # Synthetic data + PyDDM
├── SingleChoice(PyDDM).py        # PyDDM single choice
├── SingleChoice(NonPyDDM).py     # Native single choice
├── SingleChoice(SyntheticPyDDM).py
├── DualChoiceDataGen.py          # Synthetic generator
└── SingleChoiceDataGen.py        # Synthetic generator
```

**Problems:**
- Multiple versions of same concept (PyDDM vs native)
- Synthetic and real data mixed
- No clear pipeline flow

### Ver 2 Structure
```
Deprecated Ver 2/
├── Bayesian Model/       # Bayesian implementations
├── DDM Model/            # Core DDM fitting
├── NDT Code/             # NDT analysis
├── Vincentile Code/      # Distribution analysis
└── Verification Code/    # Diagnostics
```

**Advantages:**
- Clear separation by analysis type
- Each folder self-contained
- Logical progression from fitting → visualization

---

## Summary Table: Ver 1 vs Ver 2

| Aspect | Ver 1 | Ver 2 |
|--------|-------|-------|
| **Library** | PyDDM | Native scipy |
| **Data** | Synthetic only | Real KINARM data |
| **Model** | Two-boundary (choice) | Single-boundary (Wald) |
| **Validation** | None | KS statistics |
| **Visualization** | Basic | Publication-quality PDF |
| **Dependencies** | pyddm | scipy, numpy, pandas |
| **Organization** | Multiple prototypes | Pipeline structure |

---

## Recommendation

Ver 1 is preserved for historical reference only. The synthetic data generators
may be useful for:
- Teaching DDM concepts
- Parameter recovery simulations
- Testing new fitting algorithms

**For any real analysis:** Use Ver 2 or (preferably) the Current Pipeline.
