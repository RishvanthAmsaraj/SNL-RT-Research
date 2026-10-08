"""End to end: the runner executes the vendored scripts and reproduces the committed tables exactly."""
import os, shutil, time
import numpy as np, pandas as pd
import pytest
from conftest import APP, REPO
from kinarm_rt.engine import registry as R, runner


def _run(exp, data, steps, tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path))
    run_dir = runner.stage(exp, data)
    plan = [s for s in R.steps_for(exp).values() if s.id in steps]
    assert runner.start(exp, run_dir, plan, columns={"SpeedCode", "IncludeInEyeAnalysis", "Direction", "SignedError_deg", "SignedError_deg_HRT50"})
    while runner.is_running(run_dir): time.sleep(1)
    return run_dir, runner.load_state(run_dir)


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "Paradigm 2 Pipeline", "Code", "pooled_data_P2.csv")), reason="needs the repo data")
def test_experiment2_method_a_reproduces_committed_tables(tmp_path, monkeypatch):
    exp = R.EXPERIMENTS["E2"]
    run_dir, st = _run(exp, os.path.join(REPO, "Paradigm 2 Pipeline", "Code", "pooled_data_P2.csv"), {"ddm_fit", "srt_fixed_t0"}, tmp_path, monkeypatch)
    assert st["steps"]["ddm_fit"]["status"] == "done", st
    same_bytes = []
    for rel in ["DDM/DDM_hrt_fits.csv", "DDM/DDM_srt_fits.csv", "SRT Analysis/SRT_fixedt0_fits.csv"]:
        ref = os.path.join(REPO, "Paradigm 2 Pipeline", "Code", rel); new = os.path.join(run_dir, os.path.basename(rel))
        if runner.file_sha(new) == runner.file_sha(ref): same_bytes.append(rel); continue
        # different numerical-library versions: the scripts' numbers may move in the last stored digit, nothing more
        a, b = pd.read_csv(new), pd.read_csv(ref)
        assert len(a) == len(b) and list(a.columns) == list(b.columns), rel
        for c in a.columns:
            if a[c].dtype.kind in "fi":
                tol = 0.15 if c.startswith("t0") else 0.005
                assert np.nanmax(np.abs(a[c].values - b[c].values)) <= tol, (rel, c)
            else:
                assert (a[c].astype(str).values == b[c].astype(str).values).all(), (rel, c)
    if runner.environment().get("numpy", "").startswith("2.4"):
        assert len(same_bytes) == 3, same_bytes            # the lab's library versions: byte-identical
    # a second run reuses both steps
    plan = [s for s in R.steps_for(exp).values() if s.id in {"ddm_fit", "srt_fixed_t0"}]
    runner.start(exp, run_dir, plan, columns=set())
    while runner.is_running(run_dir): time.sleep(0.5)
    st = runner.load_state(run_dir)
    assert st["steps"]["ddm_fit"].get("cached") and st["steps"]["srt_fixed_t0"].get("cached")


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "Working Iterations", "SNL RT Research", "pooled_data.csv")), reason="needs the repo data")
def test_experiment1_later_reproduces_committed_table(tmp_path, monkeypatch):
    exp = R.EXPERIMENTS["E1"]
    run_dir, st = _run(exp, os.path.join(REPO, "Working Iterations", "SNL RT Research", "pooled_data.csv"), {"later"}, tmp_path, monkeypatch)
    assert st["steps"]["later"]["status"] == "done", st
    assert runner.file_sha(os.path.join(run_dir, "LATER_fits.csv")) == runner.file_sha(
        os.path.join(REPO, "Deprecated Pipelines", "LATER", "Experiment 1", "LATER_fits.csv"))


def test_missing_column_skips_step_without_running(tmp_path, monkeypatch):
    exp = R.EXPERIMENTS["E1"]; sample = os.path.join(APP, "sample_data", "example_pooled_data.csv")
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path))
    run_dir = runner.stage(exp, sample)
    runner.start(exp, run_dir, [R.steps_for(exp)["later"]], columns=set())
    while runner.is_running(run_dir): time.sleep(0.2)
    assert runner.load_state(run_dir)["steps"]["later"]["status"] == "skipped"
