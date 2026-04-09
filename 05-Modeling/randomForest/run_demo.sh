#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/sklearn-env/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec "$PYTHON_BIN" "$SCRIPT_DIR/demo_app.py" "$@"
