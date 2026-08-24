import pandas as pd

from src.models.baseline import (
    naive_prediction,
    evaluate_predictions
)


FILE = (
    "data/processed/ml/"
    "DE_LU_features.csv"
)


# ==================================================
# Load data
# ==================================================

df = pd.read_csv(
    FILE
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ==================================================
# Time-series split
# ==================================================

split_index = int(
    len(df) * 0.8
)

train = df.iloc[
    :split_index
].copy()

test = df.iloc[
    split_index:
].copy()


print(
    "Training period:"
)

print(
    train["timestamp"].min()
)

print(
    train["timestamp"].max()
)


print(
    "\nTesting period:"
)

print(
    test["timestamp"].min()
)

print(
    test["timestamp"].max()
)


# ==================================================
# Baseline prediction
# ==================================================

predicted = naive_prediction(
    test
)

actual = test[
    "target_price"
]


# ==================================================
# Metrics
# ==================================================

metrics = evaluate_predictions(
    actual,
    predicted
)


print(
    "\nNAIVE BASELINE RESULTS"
)

print(
    "=" * 40
)

for name, value in metrics.items():

    print(
        f"{name}: {value:.4f}"
    )