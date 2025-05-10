from google.cloud import aiplatform

# Initialise the ai platform
PROJECT_ID = "winter-agility-425909-q9"
LOCATION = "asia-south1"  # Or the region you used
BUCKET_URI = "gs://my-projects-123/cleaned_churn_dataset.csv"
aiplatform.init(project=PROJECT_ID, location=LOCATION)


# Load the model previously trained
model = aiplatform.Model("projects/1001559767519/locations/asia-south1/models/2883860669982048256")

# Deploy to endpoint
endpoint = model.deploy(
    deployed_model_display_name="churn-model-endpoint",
    traffic_split={"0": 100}
)

# Get the endpoint resource name
print("Deployed at endpoint:", endpoint.resource_name)
