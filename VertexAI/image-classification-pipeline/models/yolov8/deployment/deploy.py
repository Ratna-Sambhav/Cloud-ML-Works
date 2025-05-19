print("Importing libraries")

import io
import base64
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
import uvicorn
import numpy as np
import os 

print("Libraries imported")

model_dir = os.getenv("MODEL_DIR")

print(f"Env var: {model_dir}")

# Load your YOLOv8 classification model
model = YOLO(f"{model_dir}/best.pt")

print("Model loaded")

app = FastAPI()


class Instance(BaseModel):
    content: str  # base64-encoded image


class PredictRequest(BaseModel):
    instances: List[Instance]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictRequest):
    results = []

    for instance in request.instances:
        try:
            # Decode base64 image
            image_data = base64.b64decode(instance.content)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Run prediction
            pred = model(image)[0]
            probs = pred.probs.data.tolist()
            classes = pred.names

            # Get top prediction
            pred_class = int(np.argmax(probs))
            result = {
                "label": classes[pred_class],
                "confidence": round(probs[pred_class], 4),
                "all_probs": {classes[i]: round(probs[i], 4) for i in range(len(probs))}
            }
            results.append(result)

        except Exception as e:
            results.append({"error": str(e)})

    return {"predictions": results}


if __name__ == "__main__":

    print("Deploying the model")
    uvicorn.run(app, host="0.0.0.0", port=8080)
