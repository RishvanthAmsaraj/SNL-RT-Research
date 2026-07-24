"""
The double-clickable front door for the packaged app.

The app ships as a complete Python environment rather than a frozen executable,
because PyMC's compiler has to be real -- see desktop/README.md. That environment
cannot itself be double-clicked, so this tiny program sits beside it, finds it, and
starts the app with it.

It is deliberately dependency-free: it is compiled on its own and must not drag the
scientific stack into a second copy.
"""

from __future__ import annotations

import os
import subprocess
import sys


def bundle_root() -> str:
    """The folder holding this program, whether it is frozen or a plain script."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    root = bundle_root()
    if os.name == "nt":
        # pythonw runs without opening a console window
        interpreter = os.path.join(root, "env", "pythonw.exe")
        if not os.path.isfile(interpreter):
            interpreter = os.path.join(root, "env", "python.exe")
    else:
        interpreter = os.path.join(root, "env", "bin", "python")

    launcher = os.path.join(root, "app", "desktop", "launcher.py")

    for path, what in ((interpreter, "bundled Python"), (launcher, "application")):
        if not os.path.exists(path):
            sys.stderr.write(f"KINARM RT: could not find the {what} at {path}.\n"
                             f"The download may be incomplete -- try unzipping it again.\n")
            return 1

    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.call([interpreter, launcher], cwd=root, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
