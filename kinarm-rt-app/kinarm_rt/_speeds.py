"""
Shared, dependency-free constants for the KINARM RT pipeline.

Values are taken directly from the repository's CODE_REFERENCE.md so the app
reproduces the published pipeline rather than approximating it.
"""

# Canonical target-speed conditions.
SPEEDS = (0.0, 75.0, 150.0)
EFFECTORS = ("hand", "eye")

# SpeedCode -> deg/s (the data carries both SpeedCode 1/2/3 and Speed_deg_per_s).
SPEED_CODE_MAP = {1: 0.0, 2: 75.0, 3: 150.0}

# Physiological non-decision-time floors (seconds).
#   hand: reach-preparation minimum, Haith et al. 2016
#   eye : saccadic afferent+efferent dead time, Bompas & Sumner 2011; Ludwig et al. 2007
PHYSIO_FLOOR = {"hand": 0.130, "eye": 0.070}

# Data inclusion windows (seconds).
#   hand: anticipation removal (Whelan 2008) + lapse removal (Luce 1986)
#   eye : anticipation threshold (Fischer & Weber 1993) + lapse removal (Luce 1986)
FILTER_WINDOWS = {"hand": (0.150, 0.800), "eye": (0.080, 0.600)}

# Parameter caps (Tran et al. 2020 systematic-review envelope at s = 1).
V_MAX = 20.0
A_MAX = 2.5

# Mixture-shift validity floor for express/regular saccade cells (seconds).
MIX_SHIFT_FLOOR = 0.040

# Default uniform-contamination share (used by the frequentist fit; optional in
# the Bayesian fit, which by default matches the repo's pure-Wald likelihood).
P_CONTAM = 0.05

# Per-speed colour palette (RGB 0-1), matching the repository figures.
SPEED_RGB = {0: (0.45, 0.68, 0.40), 75: (0.85, 0.55, 0.55), 150: (0.50, 0.62, 0.82)}
