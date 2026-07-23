"""
Data loading, validation, and tidying for the KINARM RT pipeline.

The repository's canonical file (`pooled_data.csv`) is WIDE: one row per trial
with both HandRT_ms and GazeSRT_ms, plus Participant, Speed_deg_per_s (and/or
SpeedCode 1/2/3), and BlockType (interception trials are BlockType == "I").

The rest of the pipeline works on a TIDY (long) table with one row per trial per
effector:

    participant : str
    effector    : "hand" or "eye"
    condition   : int (0/1/2, see SPEEDS)
    speed       : float (0, 75, 150)
    rt          : float, reaction time in SECONDS

`load_trials` accepts either the wide repository format (map both RT columns) or
a long format (one RT column plus an effector column), converts milliseconds to
seconds, and optionally filters by BlockType.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ._speeds import SPEEDS, EFFECTORS, SPEED_CODE_MAP


# --------------------------------------------------------------------------- #
# Reading raw files robustly (encoding / delimiter)
# --------------------------------------------------------------------------- #
def read_table(source) -> pd.DataFrame:
    """Read a CSV/TSV path, bytes, or file-like into a DataFrame, robustly."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    # Let pandas sniff the separator; fall back through common encodings.
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source)


def _to_seconds(rt: pd.Series, units: str) -> pd.Series:
    rt = pd.to_numeric(rt, errors="coerce")
    if units == "s":
        return rt
    if units == "ms":
        return rt / 1000.0
    med = np.nanmedian(rt.values)
    return rt / 1000.0 if med > 20 else rt


def _normalize_effector(x) -> str:
    s = str(x).strip().lower()
    if s in ("hand", "manual", "reach", "h", "hrt", "handrt", "handrt_ms"):
        return "hand"
    if s in ("eye", "saccade", "saccadic", "gaze", "e", "srt", "gazesrt", "gazesrt_ms"):
        return "eye"
    return s


def _nearest_condition(speed) -> int | float:
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        return np.nan
    return int(np.argmin([abs(speed - s) for s in SPEEDS]))


def _condition_from(df, condition_col, speed_col, speedcode_col):
    """Return (condition Series 0/1/2, speed Series) from whichever column is given."""
    if speedcode_col and speedcode_col in df.columns:
        speed = pd.to_numeric(df[speedcode_col], errors="coerce").map(SPEED_CODE_MAP)
    elif speed_col and speed_col in df.columns:
        speed = pd.to_numeric(df[speed_col], errors="coerce")
    elif condition_col and condition_col in df.columns:
        cond = pd.to_numeric(df[condition_col], errors="coerce").astype("Int64")
        speed = cond.map(lambda c: SPEEDS[int(c)] if pd.notna(c) and int(c) < len(SPEEDS) else np.nan)
        return cond, speed
    else:
        raise ValueError("Provide a condition, speed, or SpeedCode column.")
    cond = speed.map(_nearest_condition).astype("Int64")
    return cond, speed


def load_trials(
    source,
    participant_col: str,
    *,
    hand_rt_col: str | None = None,
    eye_rt_col: str | None = None,
    rt_col: str | None = None,
    effector_col: str | None = None,
    effector_value: str | None = None,
    condition_col: str | None = None,
    speed_col: str | None = None,
    speedcode_col: str | None = None,
    blocktype_col: str | None = None,
    blocktype_keep: str | None = "I",
    rt_units: str = "auto",
) -> pd.DataFrame:
    """
    Load into the tidy schema.

    Wide (repository) format: pass `hand_rt_col` and/or `eye_rt_col`.
    Long format: pass `rt_col` plus either `effector_col` or `effector_value`.
    """
    df = read_table(source)

    if blocktype_col and blocktype_col in df.columns and blocktype_keep not in (None, "", "— all —"):
        df = df[df[blocktype_col].astype(str) == str(blocktype_keep)]

    cond, speed = _condition_from(df, condition_col, speed_col, speedcode_col)
    part = df[participant_col].astype(str)

    frames = []
    if hand_rt_col or eye_rt_col:                      # wide format
        for eff, col in [("hand", hand_rt_col), ("eye", eye_rt_col)]:
            if col and col in df.columns:
                f = pd.DataFrame({"participant": part, "effector": eff,
                                  "condition": cond, "speed": speed,
                                  "rt": _to_seconds(df[col], rt_units),
                                  # the source row is the trial: the hand and eye
                                  # measurements on one row come from the same trial,
                                  # and keeping that link is what lets paired analyses
                                  # (the vincentile differences) match the pipeline.
                                  # A positional counter is used rather than the index
                                  # so the link holds even if the caller hands over a
                                  # frame whose index repeats.
                                  "trial": np.arange(len(df))})
                frames.append(f)
    else:                                              # long format
        if effector_col and effector_col in df.columns:
            eff_series = df[effector_col].map(_normalize_effector)
        elif effector_value is not None:
            eff_series = pd.Series(_normalize_effector(effector_value), index=df.index)
        else:
            raise ValueError("Long format needs effector_col or effector_value.")
        frames.append(pd.DataFrame({"participant": part, "effector": eff_series,
                                    "condition": cond, "speed": speed,
                                    "rt": _to_seconds(df[rt_col], rt_units),
                                    "trial": np.arange(len(df))}))

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["rt", "condition"]).reset_index(drop=True)
    out["condition"] = out["condition"].astype(int)
    return out[["participant", "effector", "condition", "speed", "rt", "trial"]]


# --------------------------------------------------------------------------- #
# Validation & summaries
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, min_trials_per_cell: int = 15) -> list[dict]:
    issues: list[dict] = []
    if df.empty:
        return [{"level": "error", "message": "No usable trials after loading."}]
    for eff in df["effector"].unique():
        if eff not in EFFECTORS:
            issues.append({"level": "warning",
                           "message": f"Unrecognised effector '{eff}' (expected hand/eye)."})
    counts = cell_counts(df)
    thin = counts[counts["n"] < min_trials_per_cell]
    for _, r in thin.iterrows():
        issues.append({"level": "warning",
                       "message": f"{r['effector']} {r['participant']} cond {r['condition']}: "
                                  f"{int(r['n'])} trials (<{min_trials_per_cell}); this cell will be skipped."})
    if (df["rt"] <= 0).any():
        issues.append({"level": "error", "message": "Some RTs are <= 0 after unit conversion."})
    if df["rt"].median() > 5:
        issues.append({"level": "warning",
                       "message": "Median RT > 5 s; check the RT unit setting."})
    return issues


def cell_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["effector", "participant", "condition"])
              .size().rename("n").reset_index())


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import skew
    rows = []
    for (eff, cond), g in df.groupby(["effector", "condition"]):
        rt_ms = g["rt"].values * 1000.0
        cv = np.std(rt_ms) / np.mean(rt_ms)
        rows.append({"effector": eff, "condition": cond, "speed": SPEEDS[cond],
                     "n": len(g), "mean_ms": np.mean(rt_ms), "sd_ms": np.std(rt_ms),
                     "cv": cv, "skew": skew(rt_ms),
                     "skew_over_cv": skew(rt_ms) / cv if cv > 0 else np.nan})
    return pd.DataFrame(rows).sort_values(["effector", "condition"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic data (example dataset + test fixture), in the repository's WIDE shape
# --------------------------------------------------------------------------- #
def simulate_dataset(n_participants: int = 16, trials_per_cell: int = 110, seed: int = 7,
                     express_fraction_eye: float = 0.18, wide: bool = True) -> pd.DataFrame:
    """
    Realistic synthetic data with the repository's columns and phenomena:
    right-skewed hand RTs whose t0 is identified above the 130 ms floor, and
    near-symmetric saccadic RTs whose t0 floors at 70 ms, with a subset of
    express-dominant participants (a fast saccade mode).
    """
    rng = np.random.default_rng(seed)
    # Shifted-Wald (v, a) parameters -> draw inverse Gaussian mean a/v, shape a^2.
    hand_t0 = {0: 0.170, 1: 0.158, 2: 0.148}
    hand_mean, hand_shape = 0.100, 0.60        # a~0.78, v~7.8; right-skewed
    eye_t0_true = 0.030                          # below the 70 ms floor -> floors
    eye_mean, eye_shape = 0.130, 1.80            # a~1.34, v~10.3; near-symmetric

    rows = []
    for p in range(n_participants):
        pid = f"P{p+1:02d}"
        p_slow = rng.normal(0, 0.010)
        express = p < int(round(express_fraction_eye * n_participants))
        for c in range(3):
            speed = SPEEDS[c]; scode = c + 1
            hrt = hand_t0[c] + p_slow + rng.wald(hand_mean + max(p_slow * 0.4, -0.03), hand_shape,
                                                 size=trials_per_cell)
            ert = eye_t0_true + p_slow * 0.5 + rng.wald(eye_mean + max(p_slow * 0.3, -0.05), eye_shape,
                                                        size=trials_per_cell)
            if express:
                k = int(0.30 * trials_per_cell)
                idx = rng.choice(trials_per_cell, size=k, replace=False)
                ert[idx] = 0.090 + rng.normal(0.010, 0.008, size=k)
            for h, e in zip(hrt, ert):
                rows.append((pid, speed, scode, "I", float(h) * 1000.0, float(max(e, 0.03)) * 1000.0))

    df = pd.DataFrame(rows, columns=["Participant", "Speed_deg_per_s", "SpeedCode",
                                     "BlockType", "HandRT_ms", "GazeSRT_ms"])
    # a few contaminants
    n_bad = int(0.006 * len(df))
    for col in ("HandRT_ms", "GazeSRT_ms"):
        bad = rng.choice(len(df), size=n_bad, replace=False)
        df.loc[bad, col] = rng.uniform(30, 1200, size=n_bad)

    if not wide:
        return load_trials(df, "Participant", hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                           speed_col="Speed_deg_per_s", blocktype_col="BlockType")
    return df


if __name__ == "__main__":
    d = simulate_dataset()
    d.to_csv("sample_data/example_pooled_data.csv", index=False)
    print("wrote sample_data/example_pooled_data.csv", d.shape)
    tidy = load_trials(d, "Participant", hand_rt_col="HandRT_ms", eye_rt_col="GazeSRT_ms",
                       speed_col="Speed_deg_per_s", blocktype_col="BlockType")
    from .filters import apply_windows
    kept, _ = apply_windows(tidy)
    print(cell_summary(kept).round(2).to_string(index=False))
