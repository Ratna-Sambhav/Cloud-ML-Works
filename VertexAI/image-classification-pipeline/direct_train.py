from google.cloud import aiplatform

project_id = "winter-agility-425909-q9"
location = "us-central1"
BUCKET_URI = "gs://my-projects-123"
TRAIN_IMAGE = "us-central1-docker.pkg.dev/winter-agility-425909-q9/my-vai-docker-repo/train-yolov8:latest"

aiplatform.init(project=project_id, location=location, staging_bucket=BUCKET_URI)

JOB_DISPLAY_NAME = "isic-yolov8-1epoch"
REPLICA_COUNT = 1
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_COUNT = 1
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
envs = {
        "DATA_DIR": "/gcs/my-projects-123/data/",
        "MODEL_OUTPUT_DIR": "/gcs/my-projects-123/models/yolov8_isic/",
        "EPOCH": "1",
    }

# Create the custom container training job
custom_container_training_job = aiplatform.CustomContainerTrainingJob(
    display_name=JOB_DISPLAY_NAME,
    container_uri=TRAIN_IMAGE,
)

custom_container_training_job.run(
    args=None,
    # base_output_dir="gcs/my-projects-123/models/yolov8_isic/", #AIP_MODEL_DIR
    replica_count=REPLICA_COUNT,
    machine_type=MACHINE_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    accelerator_type=ACCELERATOR_TYPE,
    sync=True,
    environment_variables=envs
)