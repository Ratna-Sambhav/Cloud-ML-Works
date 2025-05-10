from google.cloud import aiplatform

# Init Vertex AI
aiplatform.init(project="winter-agility-425909-q9", location="asia-south1")

# Delete ednpoint
endpoint = aiplatform.Endpoint("projects/1001559767519/locations/asia-south1/endpoints/2283860669982048256")
endpoint.delete()