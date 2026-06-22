
You cannot run any of the code other than the synthetic ones, as you need to change the file path according to your device on the non-synthetic DDM files, which are typically located near the bottom of the code. 

# PyDDM Overview

## What is PyDDM

PyDDM (Python Drift Diffusion Model) is a Python library for building, simulating, and fitting drift-diffusion models (DDMs).
A DDM is a type of sequential sampling model used in cognitive science, neuroscience, and decision-making research to model how agents make binary decisions over time. It assumes that noisy evidence accumulates until a decision threshold is reached.

---

## How It Works

1. **Model Parameters**

   * Drift rate (v): speed of evidence accumulation
   * Boundary separation (a): amount of evidence required to make a decision
   * Starting point (z): initial bias toward one choice
   * Non-decision time (t0): perceptual or motor delays not part of the decision process

2. **Simulation**

   * PyDDM generates trajectories of evidence accumulation leading to choices and response times.

3. **Fitting**

   * Models are fit to experimental data (choice and reaction time distributions) using likelihood-based methods.

4. **Extensions**

   * Supports custom models, such as time-varying drift rates, collapsing bounds, or multiple decision boundaries.

---

## Pros

* Flexible and extensible: custom components are easy to define
* Python-native: integrates with NumPy, SciPy, and pandas
* Visualization tools: trajectories, distributions, parameter recovery
* Research-grade: designed for psychology and neuroscience experiments
* Supports complex models beyond the standard DDM

---

## Cons

* Requires background in drift-diffusion theory to use effectively
* Slower than optimized C/C++ implementations on large datasets
* Smaller community and fewer tutorials than mainstream ML libraries
* Specialized for cognitive modeling rather than general-purpose ML

---

## Alternatives

* **HDDM**: Bayesian hierarchical DDM in Python, built on PyMC/Stan. Good for group-level modeling, but slower due to sampling.
* **fast-dm**: C++ implementation with Python interface. Very fast for standard DDMs but less flexible.
* **rtdists (R)**: R package for fitting response time distributions, including DDMs.
* **Stan / PyMC**: General-purpose Bayesian frameworks where you can build custom DDMs from scratch.
