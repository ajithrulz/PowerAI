import pandas as pd
import numpy as np

from src.models.random_forest import (
    create_random_forest
)

from src.models.baseline import (
    evaluate_predictions
)


FILE = (
    "data/processed/ml/"
    "DE_LU_features.csv"
)


# ==================================================
# 1. Load dataset
# ==================================================

print("Reading feature dataset...")

df = pd.read_csv(
    FILE
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    f"Dataset rows: {len(df)}"
)


# ==================================================
# 2. Define features
# ==================================================

features = [

    # ------------------------------
    # Time
    # ------------------------------

    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",

    # ------------------------------
    # Historical price
    # ------------------------------

    "price_lag_1h",
    "price_lag_2h",
    "price_lag_3h",
    "price_lag_6h",

    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",

    # ------------------------------
    # Price statistics
    # ------------------------------

    "price_mean_6h",
    "price_mean_24h",
    "price_std_24h",
    "price_min_24h",
    "price_max_24h",
    "price_mean_168h",

    # ------------------------------
    # Load
    # ------------------------------

    "load_lag_1h",
    "load_lag_24h",
    "load_mean_24h",

    # ------------------------------
    # Renewable
    # ------------------------------

    "renewable_lag_1h",
    "renewable_lag_24h",

    # ------------------------------
    # Solar
    # ------------------------------

    "solar_lag_1h",
    "solar_lag_24h",

    # ------------------------------
    # Wind
    # ------------------------------

    "wind_lag_1h",
    "wind_lag_24h"
]


# ==================================================
# 3. Keep only columns that exist
# ==================================================

features = [
    feature
    for feature in features
    if feature in df.columns
]


print(
    "\nFeatures used by Random Forest:"
)

for feature in features:

    print(
        f"  {feature}"
    )


# ==================================================
# 4. Prepare model dataset
# ==================================================

model_df = df[
    features + ["target_price"]
].copy()

model_df = model_df.dropna()


print(
    "\nRows after removing missing values:",
    len(model_df)
)


# ==================================================
# 5. Time-based train/test split
# ==================================================

split_index = int(
    len(model_df) * 0.80
)

train = model_df.iloc[
    :split_index
]

test = model_df.iloc[
    split_index:
]


X_train = train[
    features
]

y_train = train[
    "target_price"
]

X_test = test[
    features
]

y_test = test[
    "target_price"
]


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==================================================
# 6. Train Random Forest
# ==================================================

print(
    "\nTraining Random Forest..."
)

model = create_random_forest()

model.fit(
    X_train,
    y_train
)


# ==================================================
# 7. Predict
# ==================================================

print(
    "Generating predictions..."
)

predictions = model.predict(
    X_test
)


# ==================================================
# 8. Evaluate
# ==================================================

metrics = evaluate_predictions(
    y_test.to_numpy(),
    predictions
)


print(
    "\nRANDOM FOREST RESULTS"
)

print(
    "=" * 45
)

for name, value in metrics.items():

    print(
        f"{name}: {value:.4f}"
    )


# ==================================================
# 9. Compare with Linear Regression
# ==================================================

print(
    "\nMODEL COMPARISON"
)

print(
    "=" * 45
)

print(
    "Random Forest MAE:",
    f"{metrics['MAE']:.4f}"
)

print(
    "Random Forest RMSE:",
    f"{metrics['RMSE']:.4f}"
)

print(
    "Random Forest R2:",
    f"{metrics['R2']:.4f}"
)


# ==================================================
# 10. Feature importance
# ==================================================

importance = pd.DataFrame(
    {
        "feature": features,
        "importance":
            model.feature_importances_
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nTOP 15 FEATURES"
)

print(
    "=" * 45
)

print(
    importance.head(15)
)


# ==================================================
# 11. Save feature importance
# ==================================================

importance.to_csv(
    "data/processed/ml/"
    "random_forest_feature_importance.csv",
    index=False
)


print(
    "\nFeature importance saved."
)

import matplotlib.pyplot as plt


# ==================================================
# Actual vs predicted
# ==================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    y_test.to_numpy(),
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.title(
    "Random Forest: Actual vs Predicted "
    "Electricity Price"
)

plt.xlabel(
    "Test Observation"
)

plt.ylabel(
    "Price (EUR/MWh)"
)

plt.legend()

plt.tight_layout()

plt.show()


top_features = (
    importance
    .head(15)
    .sort_values("importance")
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.show()