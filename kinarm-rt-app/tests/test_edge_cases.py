"""Bad input and bad luck: the app rejects or survives them, it never hangs or crashes."""
import io, os, json, time
import pandas as pd, pytest
from conftest import APP
from kinarm_rt.engine import registry as R, runner, intake

E1, E2 = R.EXPERIMENTS["E1"], R.EXPERIMENTS["E2"]
SAMPLE = os.path.join(APP, "sample_data", "example_pooled_data.csv")


def _wait(run_dir, seconds=90):
    for _ in range(seconds * 5):
        if not runner.is_running(run_dir): return
        time.sleep(0.2)


# ---------------------------------------------------------------- data intake
def test_header_only_file_is_rejected():
    df = pd.read_csv(io.StringIO("Participant,BlockType,Speed_deg_per_s,HandRT_ms,GazeSRT_ms\n"))
    r = intake.validate(df, E1); assert not r.ok and r.errors


def test_missing_required_columns():
    r = intake.validate(pd.DataFrame({"a": [1]}), E1); assert not r.ok and "Missing required column" in r.errors[0]


def test_file_for_the_other_experiment_is_rejected():
    r = intake.validate(pd.read_csv(SAMPLE), E2); assert not r.ok and "BlockType" in r.errors[0]


def test_missing_speed_is_an_error_and_extra_speed_a_warning():
    df = pd.read_csv(SAMPLE)
    r = intake.validate(df[df.Speed_deg_per_s != 150], E1); assert not r.ok and "[150]" in " ".join(r.errors)
    r = intake.validate(pd.concat([df, df.head(5).assign(Speed_deg_per_s=999)]), E1); assert r.ok and any("999" in w for w in r.warnings)


def test_non_numeric_reaction_times_do_not_crash():
    df = pd.read_csv(SAMPLE).astype({"HandRT_ms": object}); df.loc[:20, "HandRT_ms"] = "n/a"
    assert intake.validate(df, E1).ok


def test_missing_optional_column_warns():
    r = intake.validate(pd.read_csv(SAMPLE).drop(columns=["SpeedCode"]), E1)
    assert r.ok and any("SpeedCode" in w for w in r.warnings)


def test_garbage_bytes_are_rejected_or_raise_cleanly():
    try:
        r = intake.validate(intake.read_table(b"\x00\x01 not,a\ncsv \xff\xfe"), E1); assert not r.ok
    except Exception as exc:                       # the app catches this and says the file could not be read
        assert isinstance(exc, (ValueError, UnicodeDecodeError, pd.errors.ParserError))


# ---------------------------------------------------------------- runner
def test_failing_step_marks_failed_and_blocks_what_depends_on_it(tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path)); run_dir = runner.stage(E1, SAMPLE)
    bad = R.Step("boom", "Broken step", "does_not_exist.py", ("never.csv",))
    child = R.Step("after", "Depends on it", "DDM_fit.py", ("DDM_hrt_fits.csv",), needs=("boom",))
    assert runner.start(E1, run_dir, [bad, child], columns=set()); _wait(run_dir)
    st = runner.load_state(run_dir)
    assert st["steps"]["boom"]["status"] == "failed" and st["steps"]["after"]["status"] == "blocked"
    assert st["status"] == "finished with problems" and "not in the run folder" in st["steps"]["boom"]["detail"]


def test_interpreter_that_cannot_start_fails_the_step(tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path)); run_dir = runner.stage(E1, SAMPLE)
    monkeypatch.setattr(runner, "python_exe", lambda: os.path.join(str(tmp_path), "no_such_python"))
    runner.start(E1, run_dir, [R.steps_for(E1)["vincentile"]], columns=set()); _wait(run_dir)
    rec = runner.load_state(run_dir)["steps"]["vincentile"]
    assert rec["status"] == "failed" and "could not start Python" in rec["detail"]


def test_cancel_stops_a_running_step_and_a_second_start_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path)); run_dir = runner.stage(E1, SAMPLE)
    step = [R.steps_for(E1)["ddm_fit"]]
    assert runner.start(E1, run_dir, step, columns=set())
    assert runner.start(E1, run_dir, step, columns=set()) is False
    time.sleep(3); runner.cancel(run_dir); _wait(run_dir, 30)
    st = runner.load_state(run_dir)
    assert st["steps"]["ddm_fit"]["status"] == "cancelled" and st["status"] == "cancelled"


def test_a_run_left_running_by_a_closed_app_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path)); run_dir = runner.stage(E1, SAMPLE)
    st = runner.load_state(run_dir); st["status"] = "running"
    json.dump(st, open(os.path.join(run_dir, "run_state.json"), "w"))
    assert runner.interrupted(run_dir)


def test_staging_the_same_data_twice_reuses_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("KINARM_RT_RUNS", str(tmp_path))
    assert runner.stage(E1, SAMPLE) == runner.stage(E1, SAMPLE)


# ---------------------------------------------------------------- interface
def test_nonsense_deep_link_falls_back_quietly():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(APP, "app.py"), default_timeout=120)
    for k, v in {"exp": "nonsense", "view": "Nope", "cview": "x", "fig": "no_such_figure"}.items(): at.query_params[k] = v
    at.run(); assert not at.exception
