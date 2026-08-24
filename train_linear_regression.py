import pandas as pd
import numpy as np

from src.models.linear_regression import (
    create_linear_regression
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


# ==================================================
# 2. Define features
# ==================================================

features = [

    # Time
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",

    # Historical price
    "price_lag_1h",
    "price_lag_2h",
    "price_lag_3h",
    "price_lag_6h",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_7d",

    # Price statistics
    "price_mean_1h",
    "price_mean_6h",
    "price_mean_24h",
    "price_std_24h",
    "price_min_24h",
    "price_max_24h",
    "price_mean_7d",

    # Load
    "load_lag_1h",
    "load_lag_24h",
    "load_mean_24h",

    # Renewable
    "renewable_lag_1h",
    "renewable_lag_24h",

    # Solar
    "solar_lag_1h",
    "solar_lag_24h",

    # Wind
    "wind_lag_1h",
    "wind_lag_24h"
]


# ==================================================
# 3. Keep only available features
# ==================================================

features = [
    feature
    for feature in features
    if feature in df.columns
]


print(
    "\nFeatures used:"
)

for feature in features:
    print(
        f"  {feature}"
    )


# ==================================================
# 4. Remove missing values
# ==================================================

model_df = df[
    features + ["target_price"]
].dropna()


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
# 6. Train Linear Regression
# ==================================================

model = create_linear_regression()

print(
    "\nTraining Linear Regression..."
)

model.fit(
    X_train,
    y_train
)


# ==================================================
# 7. Predict
# ==================================================

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
    "\nLINEAR REGRESSION RESULTS"
)

print(
    "=" * 45
)

for name, value in metrics.items():

    print(
        f"{name}: {value:.4f}"
    )


# ==================================================
# 9. Compare with Naive
# ==================================================

naive_predictions = test[
    "price_lag_1h"
]

naive_metrics = evaluate_predictions(
    y_test.to_numpy(),
    naive_predictions.to_numpy()
)


print(
    "\nNAIVE BASELINE"
)

print(
    "=" * 45
)

for name, value in naive_metrics.items():

    print(
        f"{name}: {value:.4f}"
    )


# ==================================================
# 10. Improvement
# ==================================================

print(
    "\nIMPROVEMENT OVER NAIVE"
)

print(
    "=" * 45
)

mae_improvement = (
    (
        naive_metrics["MAE"]
        -
        metrics["MAE"]
    )
    /
    naive_metrics["MAE"]
) * 100

rmse_improvement = (
    (
        naive_metrics["RMSE"]
        -
        metrics["RMSE"]
    )
    /
    naive_metrics["RMSE"]
) * 100

print(
    f"MAE improvement: "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)