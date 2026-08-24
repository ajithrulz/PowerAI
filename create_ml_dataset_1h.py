import pandas as pd
import os

from src.features.forecasting_features_1h import (
    create_1h_forecasting_features
)


INPUT_FILE = (
    "data/processed/master/"
    "DE_LU_master.csv"
)

OUTPUT_FILE = (
    "data/processed/ml/"
    "DE_LU_features_1h.csv"
)


print("Reading master dataset...")

df = pd.read_csv(
    INPUT_FILE
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

print(
    f"Original rows: {len(df)}"
)


# ==================================================
# Create features
# ==================================================

df = create_1h_forecasting_features(
    df
)


# ==================================================
# Remove rows required for training
# ==================================================

required_columns = [
    "target_price",
    "price_lag_1h",
    "price_lag_24h",
    "price_lag_7d"
]

before = len(df)

df = df.dropna(
    subset=required_columns
)

after = len(df)

print(
    f"Removed rows: {before - after}"
)

print(
    f"Final rows: {after}"
)


# ==================================================
# Save
# ==================================================

os.makedirs(
    "data/processed/ml",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    
    "\nShape:"
)

print(
    df.shape
)