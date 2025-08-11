import pandas as pd
from sklearn.datasets import load_iris
import os

# Create folders if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Save raw data
raw_path = "data/raw/iris_raw.csv"
df.to_csv(raw_path, index=False)
print(f"Raw data saved to {raw_path}")

# For "processed" data, let's just normalize feature columns as an example
features = iris.feature_names
df[features] = (df[features] - df[features].mean()) / df[features].std()

processed_path = "data/processed/iris_processed.csv"
df.to_csv(processed_path, index=False)
print(f"Processed data saved to {processed_path}")