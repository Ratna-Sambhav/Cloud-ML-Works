FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

WORKDIR / 

RUN pip install ultralytics
RUN apt-get update && apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install numpy

COPY trainer /trainer

ENTRYPOINT ["python3", "./trainer/train.py"]