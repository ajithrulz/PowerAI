import pandas as pd
import numpy as np


def create_1h_forecasting_features(df):

    df = df.copy()

    # ==================================================
    # 1. Sort chronologically
    # ==================================================

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

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
    # 3. Cyclic features
    # ==================================================

    df["hour_sin"] = np.sin(
        2 * np.pi *
        df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi *
        df["hour"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    # ==================================================
    # 4. Price lags
    #
    # 15-minute data:
    # 4 observations = 1 hour
    # ==================================================

    df["price_lag_1h"] = (
        df["price_eur_mwh"].shift(4)
    )

    df["price_lag_2h"] = (
        df["price_eur_mwh"].shift(8)
    )

    df["price_lag_3h"] = (
        df["price_eur_mwh"].shift(12)
    )

    df["price_lag_6h"] = (
        df["price_eur_mwh"].shift(24)
    )

    df["price_lag_24h"] = (
        df["price_eur_mwh"].shift(96)
    )

    df["price_lag_48h"] = (
        df["price_eur_mwh"].shift(192)
    )

    df["price_lag_7d"] = (
        df["price_eur_mwh"].shift(672)
    )

    # ==================================================
    # 5. Rolling statistics
    # ==================================================

    previous_price = (
        df["price_eur_mwh"]
        .shift(1)
    )

    # Last 1 hour
    df["price_mean_1h"] = (
        previous_price
        .rolling(4)
        .mean()
    )

    # Last 6 hours
    df["price_mean_6h"] = (
        previous_price
        .rolling(24)
        .mean()
    )

    # Last 24 hours
    df["price_mean_24h"] = (
        previous_price
        .rolling(96)
        .mean()
    )

    df["price_std_24h"] = (
        previous_price
        .rolling(96)
        .std()
    )

    df["price_min_24h"] = (
        previous_price
        .rolling(96)
        .min()
    )

    df["price_max_24h"] = (
        previous_price
        .rolling(96)
        .max()
    )

    # Last 7 days
    df["price_mean_7d"] = (
        previous_price
        .rolling(672)
        .mean()
    )

    # ==================================================
    # 6. Load features
    # ==================================================

    if "load_mw" in df.columns:

        df["load_lag_1h"] = (
            df["load_mw"].shift(4)
        )

        df["load_lag_24h"] = (
            df["load_mw"].shift(96)
        )

        df["load_mean_24h"] = (
            df["load_mw"]
            .shift(1)
            .rolling(96)
            .mean()
        )

    # ==================================================
    # 7. Renewable generation
    # ==================================================

    if "renewable_generation_mw" in df.columns:

        df["renewable_lag_1h"] = (
            df[
                "renewable_generation_mw"
            ].shift(4)
        )

        df["renewable_lag_24h"] = (
            df[
                "renewable_generation_mw"
            ].shift(96)
        )

    # ==================================================
    # 8. Solar
    # ==================================================

    if "solar_generation_mw" in df.columns:

        df["solar_lag_1h"] = (
            df[
                "solar_generation_mw"
            ].shift(4)
        )

        df["solar_lag_24h"] = (
            df[
                "solar_generation_mw"
            ].shift(96)
        )

    # ==================================================
    # 9. Wind
    # ==================================================

    if "wind_generation_mw" in df.columns:

        df["wind_lag_1h"] = (
            df[
                "wind_generation_mw"
            ].shift(4)
        )

        df["wind_lag_24h"] = (
            df[
                "wind_generation_mw"
            ].shift(96)
        )

    # ==================================================
    # 10. Target
    #
    # 4 x 15 minutes = 1 hour
    # ==================================================

    df["target_price"] = (
        df["price_eur_mwh"]
        .shift(-4)
    )

    return df