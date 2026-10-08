"""
runner.py -- runs pipeline scripts, unchanged, in an isolated folder per experiment and dataset.

Each step is `python <script>` started in the run folder, exactly as when the script is run by hand: same file,
same working directory layout, same interpreter as the app. Runs happen in a background thread so the interface
stays responsive, survive page refreshes, can be cancelled, and resume where they stopped (a step whose outputs
exist for the same data and the same script is not run again).
"""
from __future__ import annotations
import hashlib, json, os, shutil, signal, subprocess, sys, threading, time
from dataclasses import dataclass, field

from .registry import Experiment, Step, steps_for

APP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINES = os.path.join(APP, "pipelines")


def runs_root() -> str:
    if os.environ.get("KINARM_RT_RUNS"): return os.environ["KINARM_RT_RUNS"]
    if os.name == "nt": base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin": base = os.path.expanduser("~/Library/Application Support")
    else: base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "kinarm-rt", "runs")


def python_exe() -> str:
    exe = sys.executable
    if os.name == "nt" and os.path.basename(exe).lower() == "pythonw.exe":
        alt = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(alt): return alt
    return exe


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Interpreter and numerical-library versions. The scripts' numbers match the committed tables exactly when these
    match the lab's environment; other versions can move Method A fits in the last stored digit."""
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "pymc", "arviz", "diptest"):
        try: out[pkg] = metadata.version(pkg)
        except Exception: pass
    return out


REQUIRED_MODULES = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib",
                    "sklearn": "scikit-learn", "diptest": "diptest"}


def missing_modules() -> list[str]:
    """Packages the scripts import. scikit-learn and diptest are optional inside DDM_fit.py -- without them it still runs
    but leaves the dip-test column empty -- so the app treats them as required: a run must match the lab's pipeline."""
    import importlib.util
    return [pip for mod, pip in REQUIRED_MODULES.items() if importlib.util.find_spec(mod) is None]


def bayes_ready() -> tuple[bool, str]:
    """PyMC importable and a C++ compiler present (without one PyMC runs ~9x slower and lands elsewhere)."""
    try:
        import importlib.util
        if importlib.util.find_spec("pymc") is None: return False, "PyMC is not installed"
    except Exception as exc:
        return False, f"PyMC check failed ({exc})"
    comp = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    return (True, "PyMC ready") if comp else (False, "no C++ compiler found — Bayesian fits would be far slower and not reproducible")


# --------------------------------------------------------------------------- state on disk
def _state_path(run_dir): return os.path.join(run_dir, "run_state.json")


def load_state(run_dir: str) -> dict:
    try:
        with open(_state_path(run_dir)) as f: return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(run_dir: str, state: dict) -> None:
    tmp = _state_path(run_dir) + ".tmp"
    with open(tmp, "w") as f: json.dump(state, f, indent=1)
    os.replace(tmp, _state_path(run_dir))


def stage(exp: Experiment, pooled_csv: str) -> str:
    """Create (or reuse) the run folder for this experiment + dataset and put the data and scripts in it."""
    dsha = file_sha(pooled_csv)
    run_dir = os.path.join(runs_root(), exp.id, dsha[:16]); os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    dst = os.path.join(run_dir, exp.data_file)
    if not os.path.exists(dst) or file_sha(dst) != dsha: shutil.copy2(pooled_csv, dst)
    for pipe in (exp.pipeline, exp.later_pipeline):
        for f in os.listdir(os.path.join(PIPELINES, pipe)):
            if f.endswith(".py"): shutil.copy2(os.path.join(PIPELINES, pipe, f), os.path.join(run_dir, f))
    st = load_state(run_dir)
    if not st:
        st = {"experiment": exp.id, "data_sha": dsha, "created": time.time(), "status": "idle", "steps": {}}
        _save_state(run_dir, st)
    return run_dir


def step_signature(run_dir: str, step: Step, data_sha: str) -> str:
    path = os.path.join(run_dir, step.script)
    if not os.path.exists(path): return ""
    return hashlib.sha256((file_sha(path) + data_sha).encode()).hexdigest()[:20]


def outputs_present(run_dir: str, step: Step) -> bool:
    return all(os.path.exists(os.path.join(run_dir, o)) for o in step.outputs)


# --------------------------------------------------------------------------- execution
@dataclass
class _Handle:
    thread: threading.Thread
    cancel: threading.Event = field(default_factory=threading.Event)
    proc: subprocess.Popen | None = None


_ACTIVE: dict[str, _Handle] = {}
_LOCK = threading.Lock()


def is_running(run_dir: str) -> bool:
    h = _ACTIVE.get(run_dir)
    return bool(h and h.thread.is_alive())


def start(exp: Experiment, run_dir: str, steps: list[Step], columns: set[str]) -> bool:
    """Start the steps in the background; returns False if a run is already active in this folder."""
    with _LOCK:
        if is_running(run_dir): return False
        handle = _Handle(threading.Thread(target=_execute, args=(exp, run_dir, steps, columns), daemon=True))
        _ACTIVE[run_dir] = handle
        handle.thread.start()
        return True


def cancel(run_dir: str) -> None:
    h = _ACTIVE.get(run_dir)
    if not h: return
    h.cancel.set()
    if h.proc and h.proc.poll() is None: _kill_tree(h.proc)


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(50):
                if proc.poll() is not None: return
                time.sleep(0.1)
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _child_env() -> dict:
    env = dict(os.environ)
    env.update({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    return env


def _execute(exp: Experiment, run_dir: str, steps: list[Step], columns: set[str]) -> None:
    """Run the steps; whatever goes wrong, the state file ends in a final status (never stuck at 'running')."""
    try:
        _execute_steps(exp, run_dir, steps, columns)
    except Exception as exc:                                    # disk full, folder removed, ... -- record it, do not hang
        st = load_state(run_dir) or {"steps": {}}
        for rec in st.get("steps", {}).values():
            if rec.get("status") == "running": rec.update(status="failed", detail=f"stopped: {exc}")
        st.update(status="finished with problems", ended=time.time(), error=f"{type(exc).__name__}: {exc}")
        try: _save_state(run_dir, st)
        except Exception: pass


def _execute_steps(exp: Experiment, run_dir: str, steps: list[Step], columns: set[str]) -> None:
    st = load_state(run_dir); data_sha = st.get("data_sha", "")
    st.update(status="running", started=time.time(), plan=[s.id for s in steps], environment=environment()); _save_state(run_dir, st)
    ok_bayes, why_bayes = bayes_ready()
    handle = _ACTIVE[run_dir]; failed = set()
    for step in steps:
        rec = st["steps"].setdefault(step.id, {})
        sig = step_signature(run_dir, step, data_sha)
        if handle.cancel.is_set():
            rec.update(status="cancelled"); continue
        blocked = [d for d in step.needs if d in failed]
        missing_cols = [c for c in step.columns if c not in columns]
        if rec.get("status") == "done" and rec.get("signature") == sig and outputs_present(run_dir, step):
            rec.update(cached=True); _save_state(run_dir, st); continue
        if blocked:
            rec.update(status="blocked", detail="needs " + ", ".join(blocked)); failed.add(step.id); _save_state(run_dir, st); continue
        if missing_cols:
            rec.update(status="skipped", detail="the data have no " + ", ".join(missing_cols) + " column"); failed.add(step.id)
            _save_state(run_dir, st); continue
        if step.bayes and not ok_bayes:
            rec.update(status="skipped", detail=why_bayes); failed.add(step.id); _save_state(run_dir, st); continue
        if not os.path.exists(os.path.join(run_dir, step.script)):
            rec.update(status="failed", detail=f"{step.script} is not in the run folder"); failed.add(step.id)
            _save_state(run_dir, st); continue
        log_path = os.path.join(run_dir, "logs", f"{step.id}.log")
        rec.update(status="running", started=time.time(), log=os.path.relpath(log_path, run_dir), cached=False, detail="")
        _save_state(run_dir, st)
        kwargs = {"cwd": run_dir, "env": _child_env(), "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            try:
                proc = subprocess.Popen([python_exe(), step.script, *step.args], stdout=log, stderr=subprocess.STDOUT, **kwargs)
            except OSError as exc:                              # interpreter missing or not executable
                rec.update(status="failed", ended=time.time(), detail=f"could not start Python: {exc}"); failed.add(step.id)
                _save_state(run_dir, st); continue
            handle.proc = proc
            while proc.poll() is None:
                if handle.cancel.is_set(): _kill_tree(proc); break
                time.sleep(0.5)
        rc = proc.wait(); handle.proc = None
        if handle.cancel.is_set():
            rec.update(status="cancelled", ended=time.time(), rc=rc)
        elif rc == 0 and outputs_present(run_dir, step):
            rec.update(status="done", ended=time.time(), rc=rc, signature=sig)
        else:
            missing = [o for o in step.outputs if not os.path.exists(os.path.join(run_dir, o))]
            rec.update(status="failed", ended=time.time(), rc=rc,
                       detail=(f"exit code {rc}" if rc else "finished without writing " + ", ".join(missing[:3])))
            failed.add(step.id)
        _save_state(run_dir, st)
    st["status"] = "cancelled" if handle.cancel.is_set() else ("finished with problems" if failed else "finished")
    st["ended"] = time.time(); _save_state(run_dir, st)


def log_tail(run_dir: str, step_id: str, n: int = 25) -> str:
    """Last lines of a step's log, with runs of identical lines collapsed (some scripts repeat font warnings)."""
    path = os.path.join(run_dir, "logs", f"{step_id}.log")
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return ""
    out, last, rep = [], None, 0
    for ln in lines:
        if ln == last: rep += 1; continue
        if rep: out.append(f"   … repeated {rep} more time{'s' if rep > 1 else ''}")
        out.append(ln); last, rep = ln, 0
    if rep: out.append(f"   … repeated {rep} more time{'s' if rep > 1 else ''}")
    return "\n".join(out[-n:])


def interrupted(run_dir: str) -> bool:
    """A state file that says 'running' with no live thread means the app was closed mid-run."""
    return load_state(run_dir).get("status") == "running" and not is_running(run_dir)
