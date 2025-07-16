ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app
COPY requirements.txt ./
COPY scripts/setup.sh ./scripts/setup.sh
RUN ./scripts/setup.sh
COPY . .
CMD ["python3", "app.py"]
