#!/usr/bin/env bash
set -euo pipefail

# Use docker buildx to build multi-arch images
if ! docker buildx inspect multiarch-builder > /dev/null 2>&1; then
    docker buildx create --name multiarch-builder --use
else
    docker buildx use multiarch-builder
fi

docker buildx build --platform linux/amd64,linux/arm64 -t sparktimize:latest --push .
