"""
intake.py -- check a dataset before anything runs, and (Experiment 2) build the pooled file from the raw
per-participant files with the repository's own builder script.
"""
from __future__ import annotations
import io, os, shutil, subprocess, tempfile
from dataclasses import dataclass, field
import pandas as pd

from .registry import Experiment, steps_for
from .runner import PIPELINES, python_exe, _child_env

REQUIRED = ("Participant", "BlockType", "Speed_deg_per_s", "HandRT_ms", "GazeSRT_ms")
HAND_WINDOW, EYE_WINDOW = (150, 800), (80, 600)          # the pipeline's RT windows, for the preview counts only


@dataclass
class Report:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    columns: set = field(default_factory=set)
    per_cell: pd.DataFrame | None = None


def read_table(src) -> pd.DataFrame:
    if isinstance(src, (bytes, bytearray)): return pd.read_csv(io.BytesIO(src), low_memory=False)
    return pd.read_csv(src, low_memory=False)


def validate(df: pd.DataFrame, exp: Experiment) -> Report:
    r = Report(ok=True, columns=set(df.columns))
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        r.ok = False; r.errors.append("Missing required column" + ("s " if len(miss) > 1 else " ") + ", ".join(miss) + ".")
        return r
    d = df[df.BlockType.astype(str) == exp.block]
    if d.empty:
        r.ok = False
        r.errors.append(f"No trials with BlockType '{exp.block}'. {exp.name} analyses only {exp.block} trials; "
                        f"this file has {', '.join(sorted(map(str, df.BlockType.unique()))[:6])}.")
        return r
    speeds = set(pd.to_numeric(d.Speed_deg_per_s, errors="coerce").dropna().astype(int).unique())
    want = set(exp.speeds)
    if want - speeds:
        r.ok = False; r.errors.append(f"{exp.name} needs speeds {sorted(want)} deg/s; the file is missing {sorted(want - speeds)}.")
    if speeds - want:
        r.warnings.append(f"Speeds {sorted(speeds - want)} deg/s are not part of {exp.name} and will be ignored by the scripts.")
    h = pd.to_numeric(d.HandRT_ms, errors="coerce"); g = pd.to_numeric(d.GazeSRT_ms, errors="coerce")
    cells = d.assign(_h=h.between(*HAND_WINDOW), _g=g.between(*EYE_WINDOW)).groupby(["Participant", "Speed_deg_per_s"]).agg(
        trials=("_h", "size"), hand_kept=("_h", "sum"), saccade_kept=("_g", "sum")).reset_index()
    r.per_cell = cells
    r.facts = {"participants": int(d.Participant.nunique()), "trials": int(len(d)),
               "hand kept": int(h.between(*HAND_WINDOW).sum()), "saccade kept": int(g.between(*EYE_WINDOW).sum()),
               "fewest hand trials in a cell": int(cells.hand_kept.min()) if len(cells) else 0}
    if len(cells) and cells.hand_kept.min() < 15:
        r.warnings.append("Some participant × speed cells have fewer than 15 usable hand trials; the scripts skip those cells.")
    by_step = {}
    for st in steps_for(exp).values():
        for c in st.columns:
            if c not in df.columns: by_step.setdefault(c, []).append(st.label)
    for c, labels in by_step.items():
        r.warnings.append(f"No {c} column, so these steps will be skipped: {', '.join(labels)}.")
    return r


def build_e2_from_raw(files: list[tuple[str, bytes]], exp: Experiment) -> tuple[str, str]:
    """Run build_pooled_data_P2.py on the uploaded per-participant files. Returns (pooled csv path, audit text)."""
    work = tempfile.mkdtemp(prefix="kinarm_build_"); raw = os.path.join(work, "raw"); os.makedirs(raw)
    for name, data in files:
        with open(os.path.join(raw, os.path.basename(name)), "wb") as f: f.write(data)
    shutil.copy2(os.path.join(PIPELINES, exp.pipeline, exp.raw_builder), work)
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}
    p = subprocess.run([python_exe(), exp.raw_builder, raw], cwd=work, capture_output=True, text=True, env=_child_env(), **kw)
    out = os.path.join(work, exp.data_file)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError((p.stdout + "\n" + p.stderr).strip()[-1500:] or "the builder did not write a pooled file")
    audit = os.path.join(work, "pooled_data_P2_audit.txt")
    return out, (open(audit, encoding="utf-8").read() if os.path.exists(audit) else p.stdout)
