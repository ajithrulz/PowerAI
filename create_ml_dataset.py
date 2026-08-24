import pandas as pd

from src.features.forecasting_features import (
    create_forecasting_features
)


INPUT_FILE = (
    "data/processed/master/"
    "DE_LU_master.csv"
)

OUTPUT_FILE = (
    "data/processed/ml/"
    "DE_LU_features.csv"
)


# ==================================================
# Read master dataset
# ==================================================

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

df = create_forecasting_features(
    df
)


# ==================================================
# Remove rows with missing values
# ==================================================

before = len(df)

df = df.dropna(
    subset=[
        "target_price",
        "price_lag_1h",
        "price_lag_24h",
        "price_lag_168h"
    ]
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

import os

os.makedirs(
    "data/processed/ml",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nML dataset saved to:"
    f"\n{OUTPUT_FILE}"
)


print(
    "\nFeature dataset shape:"
)

print(
    df.shape
)