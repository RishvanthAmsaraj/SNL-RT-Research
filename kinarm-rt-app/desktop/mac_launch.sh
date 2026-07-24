#!/bin/bash
# Entry point inside KINARM RT.app/Contents/MacOS/.
# Resolves the bundle's Resources folder and starts the app with the Python that
# ships inside it.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(cd "$HERE/../Resources" && pwd)"
exec "$RES/env/bin/python" "$RES/app/desktop/launcher.py"
