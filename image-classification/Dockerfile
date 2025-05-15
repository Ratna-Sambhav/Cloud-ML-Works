# FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

WORKDIR / 

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 python3-pip python3-dev python3-venv \
#     git wget curl ca-certificates \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Install PyTorch with GPU support (from official wheels)
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install ultralytics
RUN apt-get update && apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install numpy

COPY trainer /trainer

ENTRYPOINT ["python3", "./trainer/train.py"]