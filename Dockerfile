# Multi-arch Dockerfile for amd64 and arm64
FROM --platform=$TARGETPLATFORM python:3.10-slim

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Display build architecture
RUN echo "Building for $TARGETPLATFORM on $BUILDPLATFORM"

# Install nvidia packages using the NVIDIA PyPI index
RUN pip install -U pip \
    && pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-pyindex

CMD ["python3", "--version"]
