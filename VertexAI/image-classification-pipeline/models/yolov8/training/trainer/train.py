from ultralytics import YOLO
import torch
import os

# Check device
print("******************", "Torch availability: ", torch.cuda.is_available())

# Load yolo8 nano class model
model = YOLO("yolov8n-cls.pt")

# Train the model
data_uri = os.getenv("DATA_DIR") #"/gcs/my-projects-123/data"
model_output_dir = os.getenv("MODEL_OUTPUT_DIR") #"/gcs/my-projects-123/models/yolov8_isic"
epochs = int(os.getenv("EPOCH", 1))

results = model.train(data=data_uri, epochs=epochs, imgsz=224, 
                      project=model_output_dir, device = "cuda")