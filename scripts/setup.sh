#!/bin/bash
set -e

# Install base Python dependencies
pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.nvidia.com/

# Build xformers from source if it isn't already installed. This avoids issues
# with pre-built wheels not matching the current CUDA or PyTorch versions.
if ! python -c "import xformers" >/dev/null 2>&1; then
    TEMP_DIR=$(mktemp -d)
    git clone --recursive https://github.com/facebookresearch/xformers.git "$TEMP_DIR/xformers"
    pushd "$TEMP_DIR/xformers"
    pip install --no-cache-dir -v -e .
    popd
    rm -rf "$TEMP_DIR"
fi
