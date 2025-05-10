import pandas as pd
import numpy as np
from google.cloud import storage
import os

## Download kaggle dataset
os.system("curl -L -o telco-customer-churn.zip https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn")
os.system("unzip telco-customer-churn.zip")

## Load the dataset as pandas dataframe
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

## TotalCharges column contains numeric data but the data is in the form of string. Converting to numeric type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

## SeniorCitizen column contains numeric data but it is supposed to be categorical type
df['SeniorCitizen'] = ['Yes' if i == 1 else 'No' for i in df['SeniorCitizen'].values]
df.drop('customerID', axis=1, inplace=True)

## Take a look at unique values in each column to see if they have enough variance for prediction class
for col in df.columns:
    print(col, np.unique(df[col], return_counts=True))

## Exporting to csv format
df.to_csv("cleaned_churn_dataset.csv", index=False)

## Upload the data to gcp bucket and retrieve uri
gcs_client = storage.Client.from_service_account_json("winter-agility-425909-q9-d81ee40df75a.json")
bucket_name = "my-projects-123"
bucket = gcs_client.bucket(bucket_name)

file = open("cleaned_churn_dataset.csv", "rb")
blob = bucket.blob("cleaned_churn_dataset.csv")
blob.upload_from_file(file, content_type="text/csv")
file.close()

gcs_uri = f"gs://{bucket_name}/{blob.name}"
print(f"Data was downloaded, extracted, cleaned, prepared and uploaded to bucket with uril {gcs_uri}")