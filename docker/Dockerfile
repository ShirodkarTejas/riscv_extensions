# CUDA 12.1 base with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Optional GPU deps (CUDA 12.1 wheels)
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install triton

# Default command: run tests
CMD ["pytest", "-q"]


