"""
Check that a freshly built bundle actually starts and serves the app.

A build can succeed and still produce something that dies on launch -- a missing
hidden import only shows up when the module is first needed, which is at runtime.
This runs the built executable, waits for it to report its address, fetches the
page and looks for markup the app is known to emit, then shuts it down. It runs on
the build machine as part of CI, so a bundle that cannot start fails the build
instead of being published.

    python desktop/smoke_test.py "dist/KINARM RT/KINARM RT.exe"
    python desktop/smoke_test.py "staging/env/bin/python" --script app/desktop/launcher.py

The second form is what the packaged build uses: the app ships a real Python rather
than a frozen executable, so the interpreter and the script are given separately.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

STARTUP_TIMEOUT = 180.0     # cold start on a CI runner is slow
RENDER_TIMEOUT = 120.0
URL_PATTERN = re.compile(r"http://127\.0\.0\.1:\d+")


def fetch(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "kinarm-smoke-test"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: smoke_test.py <executable> [--script <launcher.py>]",
              file=sys.stderr)
        return 2
    exe = sys.argv[1]
    if not os.path.exists(exe):
        print(f"FAIL: nothing at {exe}", file=sys.stderr)
        return 1

    cmd = [exe]
    if "--script" in sys.argv:
        script = sys.argv[sys.argv.index("--script") + 1]
        if not os.path.exists(script):
            print(f"FAIL: no launcher script at {script}", file=sys.stderr)
            return 1
        cmd.append(script)

    print(f"launching {' '.join(cmd)}")
    env = dict(os.environ)
    env.pop("DISPLAY", None)        # force the browser fallback: CI has no desktop
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)

    url = None
    bayesian_line = None          # True/False once the launcher reports it
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        print("   ", line.rstrip())
        if "PyMC ready" in line:
            bayesian_line = True
        elif "Bayesian fit will be unavailable" in line:
            bayesian_line = False
        match = URL_PATTERN.search(line)
        if match:
            url = match.group(0)
            break

    try:
        if not url:
            print(f"FAIL: the app did not report an address within "
                  f"{STARTUP_TIMEOUT:.0f}s", file=sys.stderr)
            return 1

        # Streamlit serves the shell immediately but runs the script a moment later,
        # so poll until the app's own markup appears rather than trusting the first
        # response.
        deadline = time.time() + RENDER_TIMEOUT
        last = ""
        while time.time() < deadline:
            try:
                last = fetch(url)
                if "streamlit" in last.lower():
                    print(f"PASS: {url} served {len(last)} bytes")
                    if bayesian_line is False:
                        print("FAIL: the app reported that the Bayesian fit is "
                              "unavailable; the build is missing PyMC or a compiler",
                              file=sys.stderr)
                        return 1
                    return 0
            except (urllib.error.URLError, OSError) as exc:
                last = f"({exc})"
            time.sleep(2.0)

        print(f"FAIL: {url} never served a usable page. Last response: "
              f"{last[:200]}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
