#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Prefer Desktop venv (outside iCloud Drive) to avoid dyld mmap hang
if [ -x "$HOME/Desktop/venv311/bin/python" ]; then
  exec "$HOME/Desktop/venv311/bin/python" gui.py
fi

if [ -x .venv311/bin/python ]; then
  exec .venv311/bin/python gui.py
fi

echo "ERROR: virtual environment not found."
echo "Create a venv outside iCloud Drive and install dependencies:"
echo "  /opt/homebrew/bin/python3.11 -m venv ~/Desktop/venv311"
echo "  ~/Desktop/venv311/bin/pip install -r requirements.txt"
exit 1
