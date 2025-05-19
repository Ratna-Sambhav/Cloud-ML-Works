from google.cloud import aiplatform

artifact_uri = "gs://my-projects-123/models/yolov8_isic/train/weights"
serving_container = "us-central1-docker.pkg.dev/winter-agility-425909-q9/my-vai-docker-repo/deploy-yolov8:latest"
aiplatform.init(
    project="winter-agility-425909-q9",
    location="us-central1",
    staging_bucket="gs://my-projects-123",
)

# Upload the model using a custom serving container
model = aiplatform.Model.upload(
    display_name="yolov8-isic-model-deployment",
    artifact_uri=artifact_uri,
    serving_container_image_uri=serving_container,
    serving_container_environment_variables={
        "MODEL_DIR": artifact_uri,
    }
)

# Deploy the model with custom environment variables
endpoint = model.deploy(
    machine_type="n1-standard-4",
    # accelerator_type=None,
    # accelerator_count=None,
)
