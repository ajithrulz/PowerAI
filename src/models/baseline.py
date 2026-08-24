import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_predictions(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    # Avoid division by zero
    non_zero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (
                    actual[non_zero]
                    -
                    predicted[non_zero]
                )
                /
                actual[non_zero]
            )
        )
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def naive_prediction(
    df
):

    return df["price_lag_1h"]