FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.nvidia.com/
COPY . .
CMD ["python3", "app.py"]
