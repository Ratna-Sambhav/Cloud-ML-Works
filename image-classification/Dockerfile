FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR / 

COPY trainer /trainer 

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git wget curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with GPU support (from official wheels)
RUN pip install ultralytics

ENTRYPOINT ["python3", "/trainer/train.py"]