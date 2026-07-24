"""
Desktop entry point for the KINARM RT app.

The app itself is a Streamlit server, so this wraps it to behave like an ordinary
desktop program: double-click to open, close the window to quit. The server listens
on the loopback interface only and picks a free port at startup, so nothing is
exposed to the network and several copies can run side by side.

Two processes are used, and that is deliberate rather than incidental. Streamlit
installs signal handlers when it starts, which only works on a process's main
thread; on macOS the native window toolkit also has to own the main thread. Both
cannot have it, so the launcher re-runs itself with a hidden flag: the child serves
the app, the parent shows the window and shuts the child down on close.

A native window is used when pywebview is present. If it is not, the app opens in
the default browser, which is a fully working fallback rather than a failure.

Run it from a source checkout with:
    python desktop/launcher.py
"""

from __future__ import annotations

import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

APP_NAME = "KINARM RT"
HOST = "127.0.0.1"
SERVE_FLAG = "--kx-serve"      # internal: marks the child process
STARTUP_TIMEOUT = 120.0        # first import of SciPy and friends is slow on a cold disk
WINDOW_SIZE = (1440, 940)
MIN_WINDOW = (1024, 700)


def resource_root() -> str:
    """
    Where app.py and the package live.

    PyInstaller unpacks a bundle into a temporary directory and records it in
    sys._MEIPASS. From a source checkout the files sit one level above this script.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    return bundled or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port() -> int:
    """Ask the operating system for a port nothing else is using."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


def server_is_up(port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Runtime environment
# --------------------------------------------------------------------------- #
def prepare_pytensor_env() -> None:
    """
    Make sure PyMC can find a C++ compiler.

    PyTensor generates C++ for the model's log-likelihood and compiles it on first
    use. If it cannot find a compiler it silently drops to a pure-Python evaluation
    path -- measured at roughly nine times slower, and the sampler does not land on
    the same estimates, so a fit run that way is neither as fast nor as reproducible
    as one from the conda environment. The packaged app therefore carries its own
    toolchain, and this points PyTensor at it.

    Nothing is forced: if a compiler is already on PATH, that one is used.
    """
    root = os.path.dirname(sys.executable)          # the packed environment's root
    candidates = []
    if os.name == "nt":
        # conda-forge's m2w64 toolchain
        candidates += [
            os.path.join(root, "Library", "mingw-w64", "bin"),
            os.path.join(root, "Library", "usr", "bin"),
            os.path.join(root, "Library", "bin"),
        ]
    else:
        candidates += [os.path.join(root, "bin")]

    existing = os.environ.get("PATH", "")
    added = [d for d in candidates if os.path.isdir(d) and d not in existing]
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [existing])

    # Keep the compile cache beside the user's other application data rather than
    # inside the bundle, which may sit in a read-only location.
    if "PYTENSOR_FLAGS" not in os.environ:
        cache = os.path.join(
            os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache"),
            "kinarm-rt", "pytensor")
        try:
            os.makedirs(cache, exist_ok=True)
            os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={cache}"
        except OSError:
            pass                                    # fall back to PyTensor's default


def bayesian_available() -> tuple[bool, str]:
    """Whether the hierarchical fit can run here, and why not if it cannot."""
    try:
        import pymc  # noqa: F401
    except Exception as exc:
        return False, f"PyMC is not installed ({type(exc).__name__})"
    import shutil
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if not compiler:
        return False, "no C++ compiler found; sampling would be far slower"
    return True, f"PyMC ready, compiling with {os.path.basename(compiler)}"


# --------------------------------------------------------------------------- #
# Child role: serve the app
# --------------------------------------------------------------------------- #
def serve(port: int) -> int:
    """Run the Streamlit server in this process's main thread, and block."""
    from streamlit import config as st_config
    from streamlit.web import bootstrap

    prepare_pytensor_env()
    root = resource_root()
    script = os.path.join(root, "app.py")
    if not os.path.isfile(script):
        print(f"[{APP_NAME}] app.py not found at {script}", file=sys.stderr)
        return 1
    os.chdir(root)          # Streamlit and the bundled sample data use relative paths

    st_config.set_option("server.port", port)
    st_config.set_option("server.address", HOST)
    st_config.set_option("server.headless", True)           # the parent opens the window
    # kept consistent with XSRF protection: leaving these to differ makes Streamlit
    # override one of them and print a warning on every start
    st_config.set_option("server.enableCORS", True)
    st_config.set_option("server.enableXsrfProtection", True)
    st_config.set_option("browser.gatherUsageStats", False)
    st_config.set_option("server.fileWatcherType", "none")  # nothing to watch in a bundle
    st_config.set_option("server.maxUploadSize", 500)       # trial files can be large
    st_config.set_option("global.developmentMode", False)
    st_config.set_option("theme.base", "light")

    bootstrap.run(script, False, [], {})
    return 0


# --------------------------------------------------------------------------- #
# Parent role: start the child, show the window
# --------------------------------------------------------------------------- #
def spawn_server(port: int) -> subprocess.Popen:
    """Re-run this program in its serving role."""
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, SERVE_FLAG, str(port)]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), SERVE_FLAG, str(port)]
    kwargs = {}
    if os.name == "nt":
        # no console window for the background process
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(cmd, **kwargs)


def wait_for_server(proc: subprocess.Popen, port: int) -> bool:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if server_is_up(port):
            return True
        if proc.poll() is not None:        # child exited before it was ready
            return False
        time.sleep(0.25)
    return False


def open_window(url: str, on_close) -> bool:
    """Show the app in a native window. Returns False if that is not possible."""
    try:
        import webview
    except Exception:
        return False
    try:
        window = webview.create_window(APP_NAME, url,
                                       width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
                                       min_size=MIN_WINDOW, confirm_close=False)
        try:
            window.events.closed += on_close
        except Exception:
            pass                            # older pywebview: shutdown still runs below
        webview.start()
        return True
    except Exception as exc:                # headless session, missing GTK/WebKit, etc.
        print(f"[{APP_NAME}] Could not open a native window ({exc}).")
        return False


def main() -> int:
    multiprocessing.freeze_support()

    # child role
    if SERVE_FLAG in sys.argv:
        i = sys.argv.index(SERVE_FLAG)
        try:
            port = int(sys.argv[i + 1])
        except (IndexError, ValueError):
            print(f"[{APP_NAME}] {SERVE_FLAG} needs a port number", file=sys.stderr)
            return 2
        return serve(port)

    # parent role
    prepare_pytensor_env()
    ok, why = bayesian_available()
    print(f"[{APP_NAME}] {why}")
    if not ok:
        print(f"[{APP_NAME}] The hierarchical Bayesian fit will be unavailable; "
              f"everything else works.")

    port = free_port()
    url = f"http://{HOST}:{port}"
    print(f"[{APP_NAME}] Starting…")
    proc = spawn_server(port)

    stopping = threading.Event()

    def shutdown(*_args) -> None:
        if stopping.is_set():
            return
        stopping.set()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        if not wait_for_server(proc, port):
            if proc.poll() is not None:
                print(f"[{APP_NAME}] The app failed to start "
                      f"(exit code {proc.returncode}).", file=sys.stderr)
            else:
                print(f"[{APP_NAME}] The app did not start within "
                      f"{STARTUP_TIMEOUT:.0f} seconds.", file=sys.stderr)
            return 1

        print(f"[{APP_NAME}] Ready at {url}")
        if not open_window(url, shutdown):
            print(f"[{APP_NAME}] Opening in your browser instead. "
                  f"Keep this window open while you work — closing it stops the app.")
            webbrowser.open(url)
            try:
                while proc.poll() is None:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        return 0
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
