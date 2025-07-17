#!/bin/bash
# Simple helper to install dependencies and launch the Gradio UI
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Install dependencies
if [ -f "$SCRIPT_DIR/scripts/setup.sh" ]; then
    bash "$SCRIPT_DIR/scripts/setup.sh"
fi

# Launch the UI
python "$SCRIPT_DIR/app.py" "$@"
