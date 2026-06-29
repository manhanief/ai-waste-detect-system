#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv311/bin/python ]; then
  exec .venv311/bin/python gui.py
fi

echo "ERROR: .venv311 virtual environment not found."
echo "Create it with Python 3.11, then install dependencies:"
echo "  /opt/homebrew/bin/python3.11 -m venv .venv311"
echo "  .venv311/bin/python -m pip install -r requirements.txt"
exit 1
