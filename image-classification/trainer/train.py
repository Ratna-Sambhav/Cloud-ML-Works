from ultralytics import YOLO

# Load yolo8 nano class model
model = YOLO("yolov8n-cls.pt")

# Train the model
data_uri = "/gcs/my-projects-123/data"
model_output_dir = "/gcs/my-projects-123/models/yolov8_isic"
results = model.train(data=data_uri, epochs=5, imgsz=224, 
                      project=model_output_dir)