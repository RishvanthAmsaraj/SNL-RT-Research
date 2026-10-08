"""
build_pooled_data_P2.py  --  Paradigm 2 (CIR) input builder

Concatenates the 16 per-participant CIR*_TRIAL_Summary_v0_1_36.csv files into ONE
canonical input, pooled_data_P2.csv, in the same long format the Paradigm 1 pipeline
reads from pooled_data.csv:

  * one row per trial, every original column kept unchanged (nothing is dropped,
    renamed or recomputed);
  * a `Participant` column is added (a copy of `ParticipantID`), because every
    pipeline script keys on `Participant`, as in Paradigm 1;
  * rows are sorted by Participant, TrialNum.

No trial is filtered here. The pipeline scripts apply exactly the Paradigm 1 rules:
BlockType filter (here "P2" -- the only block type in these files), then the RT
windows (hand 150-800 ms, saccade 80-600 ms).

Integrity checks (the build aborts if any fails):
  - all files share one identical header;
  - (Participant, TrialNum) and TrialKey are unique;
  - speeds are exactly {75, 100, 125, 150} and BlockType is exactly {"P2"};
  - HandRT_ms / GazeSRT_ms in the output are bit-identical to the source files.

Run: python build_pooled_data_P2.py  [folder containing the CIR*_TRIAL_Summary files]
     (default: the folder this script sits in)
Output: pooled_data_P2.csv  +  pooled_data_P2_audit.txt
"""
import os, sys, glob, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else HERE
PATTERN = "CIR*_TRIAL_Summary_v0_1_36.csv"
OUT = os.path.join(HERE, "pooled_data_P2.csv")
AUDIT = os.path.join(HERE, "pooled_data_P2_audit.txt")
EXPECTED_SPEEDS = {75, 100, 125, 150}
EXPECTED_BLOCK = {"P2"}


def main():
    files = sorted(glob.glob(os.path.join(SRC, PATTERN)))
    if not files:
        sys.exit(f"ERROR: no {PATTERN} files found in {SRC}")
    parts, header, log = [], None, []
    for f in files:
        d = pd.read_csv(f)
        if header is None:
            header = list(d.columns)
        elif list(d.columns) != header:
            sys.exit(f"ERROR: column mismatch in {os.path.basename(f)}")
        parts.append(d)
        log.append(f"  {os.path.basename(f)}: {len(d)} rows, sha1 {hashlib.sha1(open(f,'rb').read()).hexdigest()[:12]}")
    df = pd.concat(parts, ignore_index=True).copy()
    df.insert(0, "Participant", df["ParticipantID"])
    df = df.sort_values(["Participant", "TrialNum"], kind="mergesort").reset_index(drop=True)

    # ---------------- integrity checks
    assert not df.duplicated(["Participant", "TrialNum"]).any(), "duplicate (Participant, TrialNum)"
    assert df["TrialKey"].is_unique, "duplicate TrialKey"
    assert set(df["Speed_deg_per_s"].unique()) == EXPECTED_SPEEDS, f"speeds {sorted(df.Speed_deg_per_s.unique())}"
    assert set(df["BlockType"].unique()) == EXPECTED_BLOCK, f"block types {df.BlockType.unique()}"
    src = pd.concat(parts, ignore_index=True).set_index(["ParticipantID", "TrialNum"]).sort_index()
    chk = df.set_index(["ParticipantID", "TrialNum"]).sort_index()
    for c in ["HandRT_ms", "GazeSRT_ms", "Speed_deg_per_s", "Direction"]:
        a, b = src[c], chk[c]
        same = (a == b) | (a.isna() & b.isna())
        assert bool(same.all()), f"{c} changed during pooling"
    df.to_csv(OUT, index=False)

    # ---------------- audit report (what the pipeline will see)
    h, g = df["HandRT_ms"], df["GazeSRT_ms"]
    hw, gw = h.between(150, 800), g.between(80, 600)
    lines = ["pooled_data_P2.csv -- build audit", "=" * 60,
             f"source files: {len(files)}", *log,
             f"rows: {len(df)}   columns: {df.shape[1]}   participants: {df.Participant.nunique()}",
             f"BlockType: {df.BlockType.value_counts().to_dict()}",
             "", "trials per participant x speed (all rows):",
             df.groupby(["Participant", "Speed_deg_per_s"]).size().unstack().to_string(),
             "", f"HandRT_ms non-missing {h.notna().sum()}; in 150-800 ms window {hw.sum()} "
                 f"(<150: {(h < 150).sum()}, >800: {(h > 800).sum()})",
             f"GazeSRT_ms non-missing {g.notna().sum()}; in 80-600 ms window {gw.sum()} "
                 f"(<80: {(g < 80).sum()}, >600: {(g > 600).sum()})",
             "", "hand trials kept per cell (150-800 ms):",
             df[hw].groupby(["Participant", "Speed_deg_per_s"]).size().unstack().to_string(),
             "", "saccade trials kept per cell (80-600 ms):",
             df[gw].groupby(["Participant", "Speed_deg_per_s"]).size().unstack().to_string(),
             "", "Extraction QA flags vs the Paradigm 1 RT-window rule:",
             f"  hand: in-window trials with IncludeInHandAnalysis==0: {int((hw & (df.IncludeInHandAnalysis == 0)).sum())}",
             f"  eye : in-window trials with IncludeInEyeAnalysis==0 : {int((gw & (df.IncludeInEyeAnalysis == 0)).sum())}"
             "  (whole-trial exclusions because HandRT <= 100 ms; kept by the P1 rule,"
             " see SRT_QA_flag_sensitivity.py)"]
    open(AUDIT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    print(f"saved {OUT}\nsaved {AUDIT}")


if __name__ == "__main__":
    main()
