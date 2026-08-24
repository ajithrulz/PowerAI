import pandas as pd
import numpy as np


def create_forecasting_features(df):

    df = df.copy()

    # ==================================================
    # 1. Sort chronologically
    # ==================================================

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ==================================================
    # 2. Time features
    # ==================================================

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ==================================================
    # 3. Cyclic hour encoding
    # ==================================================

    df["hour_sin"] = np.sin(
        2 * np.pi *
        df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi *
        df["hour"] / 24
    )

    # ==================================================
    # 4. Cyclic day-of-week encoding
    # ==================================================

    df["day_sin"] = np.sin(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    # ==================================================
    # 5. Historical price features
    # ==================================================

    df["price_lag_1h"] = (
        df["price_eur_mwh"]
        .shift(1)
    )

    df["price_lag_2h"] = (
        df["price_eur_mwh"]
        .shift(2)
    )

    df["price_lag_3h"] = (
        df["price_eur_mwh"]
        .shift(3)
    )

    df["price_lag_6h"] = (
        df["price_eur_mwh"]
        .shift(6)
    )

    df["price_lag_24h"] = (
        df["price_eur_mwh"]
        .shift(24)
    )

    df["price_lag_48h"] = (
        df["price_eur_mwh"]
        .shift(48)
    )

    df["price_lag_168h"] = (
        df["price_eur_mwh"]
        .shift(168)
    )

    # ==================================================
    # 6. Rolling price statistics
    # ==================================================

    previous_price = (
        df["price_eur_mwh"]
        .shift(1)
    )

    df["price_mean_6h"] = (
        previous_price
        .rolling(6)
        .mean()
    )

    df["price_mean_24h"] = (
        previous_price
        .rolling(24)
        .mean()
    )

    df["price_std_24h"] = (
        previous_price
        .rolling(24)
        .std()
    )

    df["price_min_24h"] = (
        previous_price
        .rolling(24)
        .min()
    )

    df["price_max_24h"] = (
        previous_price
        .rolling(24)
        .max()
    )

    df["price_mean_168h"] = (
        previous_price
        .rolling(168)
        .mean()
    )

    # ==================================================
    # 7. Historical load
    # ==================================================

    if "load_mw" in df.columns:

        df["load_lag_1h"] = (
            df["load_mw"]
            .shift(1)
        )

        df["load_lag_24h"] = (
            df["load_mw"]
            .shift(24)
        )

        df["load_mean_24h"] = (
            df["load_mw"]
            .shift(1)
            .rolling(24)
            .mean()
        )

    # ==================================================
    # 8. Historical renewable generation
    # ==================================================

    if "renewable_generation_mw" in df.columns:

        df["renewable_lag_1h"] = (
            df["renewable_generation_mw"]
            .shift(1)
        )

        df["renewable_lag_24h"] = (
            df["renewable_generation_mw"]
            .shift(24)
        )

    # ==================================================
    # 9. Historical solar
    # ==================================================

    if "solar_generation_mw" in df.columns:

        df["solar_lag_1h"] = (
            df["solar_generation_mw"]
            .shift(1)
        )

        df["solar_lag_24h"] = (
            df["solar_generation_mw"]
            .shift(24)
        )

    # ==================================================
    # 10. Historical wind
    # ==================================================

    if "wind_generation_mw" in df.columns:

        df["wind_lag_1h"] = (
            df["wind_generation_mw"]
            .shift(1)
        )

        df["wind_lag_24h"] = (
            df["wind_generation_mw"]
            .shift(24)
        )

    # ==================================================
    # 11. Target
    # ==================================================

    df["target_price"] = (
        df["price_eur_mwh"]
        .shift(-1)
    )

    return df