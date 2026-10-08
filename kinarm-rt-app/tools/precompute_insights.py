#!/usr/bin/env python3
"""
precompute_insights.py -- compute the aggregate numbers the app's figure explanations and
comparison notes show, from the per-participant fit tables, and store them as
pipelines/insights_summary.json.

The per-participant tables are participant data and are not shipped, so the app cannot compute
these numbers at runtime. Run this script where the tables exist (a local copy of the lab
results, for example an earlier app bundle) and commit the summary; the app ships the summary
and falls back to it whenever a table is absent. "Your run" results still compute live from the
tables the scripts just wrote.

    python tools/precompute_insights.py <E1_dir> <E2_dir> <E1_later_dir> <E2_later_dir> <E1_two_boundary_dir>

Only aggregate numbers (group means, counts, p-values) end up in the summary; the script refuses
to write if a participant identifier shows up in the output.
"""

from __future__ import annotations
import json, os, re, sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from kinarm_rt.engine import results
from kinarm_rt import insights

SPEEDS = {"E1": (0, 75, 150), "E2": (75, 100, 125, 150)}
SCHEMATIC_TABLES = {"bayes_hrt": "Bayesian_hrt_fits.csv", "bayes_srt": "Bayesian_srt_fits.csv",
                    "ddm_hrt": "DDM_hrt_fits.csv", "ddm_srt": "DDM_srt_fits.csv"}
PARTICIPANT_ID = re.compile(r"\b(CMT[0-9]{3,4}|CIR[0-9]{3})\b")


def build(label: str, folders: list, speeds: tuple) -> dict:
    rs = results.ResultSet(label, "reference", tuple(folders))
    out = {
        "headline": results.headline(rs, speeds),
        "method_a": insights.method_a(rs, speeds),
        "saccade_models": insights.saccade_models(rs),
        "later": insights.later(rs),
        "schematic": {},
    }
    for key, table in SCHEMATIC_TABLES.items():
        for s in speeds:
            g = insights.schematic(rs, table, s)
            if g:
                out["schematic"][f"{key}_{s}"] = g
    if "two_boundary_readiness_hand.csv" in rs.tables():
        out["readiness"] = insights.readiness(rs)
    return out


def main() -> int:
    dirs = sys.argv[1:]
    if len(dirs) < 5:
        sys.exit(__doc__)
    summary = {
        "E1": build("E1", dirs[:1] + dirs[2:3] + dirs[4:5], SPEEDS["E1"]),
        "E2": build("E2", dirs[1:2] + dirs[3:4], SPEEDS["E2"]),
    }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if PARTICIPANT_ID.search(text):
        sys.exit("refusing to write: participant identifiers found in the computed summary")
    out_path = os.path.join(APP, "pipelines", "insights_summary.json")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"wrote {out_path} ({len(text)} chars)")
    print("E1 hand_t0:", summary["E1"]["headline"].get("hand_t0"))
    print("E2 hand_t0:", summary["E2"]["headline"].get("hand_t0"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
