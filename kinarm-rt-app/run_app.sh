#!/usr/bin/env bash
# Start the KINARM RT app on macOS / Linux. Opens in your web browser.
cd "$(dirname "$0")" || exit 1

if command -v streamlit >/dev/null 2>&1; then
  echo "Starting the app... a browser tab should open. Close this window to stop it."
  exec streamlit run app.py
fi

echo "Streamlit was not found on your PATH."
echo
echo "Set it up once, then re-run this script:"
echo "  Option A (conda, recommended):"
echo "    conda env create -f environment.yml && conda activate kinarm-rt"
echo "  Option B (pip):"
echo "    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
echo
read -r -p "Press Enter to close."
