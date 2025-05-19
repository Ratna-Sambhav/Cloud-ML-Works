FROM us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest

WORKDIR / 

RUN pip install ultralytics
RUN apt-get update && apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install numpy
RUN pip install fastapi uvicorn pillow

COPY deploy.py /

ENTRYPOINT ["python3", "./deploy.py"]