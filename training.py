from google.cloud import aiplatform

# Initialise the ai platform
PROJECT_ID = "winter-agility-425909-q9"
LOCATION = "asia-south1"  # Or the region you used
BUCKET_URI = "gs://my-projects-123/cleaned_churn_dataset.csv"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Get the dataset
dataset = aiplatform.TabularDataset('projects/1001559767519/locations/asia-south1/datasets/2883860669982048256')

# Defin an automl tabular training job
job = aiplatform.AutoMLTabularTrainingJob(
  display_name="train-automl-churn-prediction",
  optimization_prediction_type="classification",
  optimization_objective="maximize-au-roc",
)

# Start model training
model = job.run(
    dataset=dataset,
    target_column="Churn",
    training_fraction_split=0.6,
    validation_fraction_split=0.2,
    test_fraction_split=0.2,
    budget_milli_node_hours=1000,
    model_display_name="train-automl-churn-prediction",
    disable_early_stopping=False,
    sync=True
)

# Get model resource number for future use
print("\n\nModel training done. Here is the model reference id: ", model.resource_name)