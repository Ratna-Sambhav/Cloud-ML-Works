from google.cloud import aiplatform

# Initialise the ai platform
PROJECT_ID = "winter-agility-425909-q9"
LOCATION = "asia-south1"  # Or the region you used
BUCKET_URI = "gs://my-projects/cleaned_churn_dataset.csv"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Create the dataset using the cleaned dataset uploaded at the provided bucket
my_dataset = aiplatform.TabularDataset.create(
    display_name="my-churn-dataset", gcs_source=[BUCKET_URI])

# Get dataset resource name for further use
print("Here is the uploaded dataset: ", my_dataset.resource_name)