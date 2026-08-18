import pandas as pd
import numpy as np


class PowerDataPreprocessor:

    def __init__(self, df):

        self.df = df.copy()

    # ------------------------------------------------
    # Timestamp processing
    # ------------------------------------------------

    def process_timestamp(self):

        self.df["timestamp"] = pd.to_datetime(
            self.df["timestamp"],
            utc=True
        )

        self.df = self.df.sort_values(
            "timestamp"
        )

        self.df = self.df.drop_duplicates(
            subset=["timestamp"]
        )

        return self

    # ------------------------------------------------
    # Missing values
    # ------------------------------------------------

    def handle_missing_values(self):

        numeric_columns = (
            self.df
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        for column in numeric_columns:

            self.df[column] = (
                self.df[column]
                .interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )

        return self

    # ------------------------------------------------
    # Outlier handling
    # ------------------------------------------------

    def handle_price_outliers(
        self,
        column="price_eur_mwh"
    ):

        if column not in self.df.columns:
            return self

        # European electricity prices can legitimately
        # be negative or very high, so we should NOT
        # blindly remove outliers.

        # Instead, use robust percentile clipping only
        # as an optional preprocessing operation.

        lower = self.df[column].quantile(0.01)
        upper = self.df[column].quantile(0.99)

        self.df[f"{column}_clipped"] = (
            self.df[column]
            .clip(
                lower=lower,
                upper=upper
            )
        )

        return self

    # ------------------------------------------------
    # Calendar features
    # ------------------------------------------------

    def create_calendar_features(self):

        timestamp = self.df["timestamp"]

        self.df["hour"] = timestamp.dt.hour

        self.df["day"] = timestamp.dt.day

        self.df["day_of_week"] = (
            timestamp.dt.dayofweek
        )

        self.df["month"] = (
            timestamp.dt.month
        )

        self.df["quarter"] = (
            timestamp.dt.quarter
        )

        self.df["year"] = (
            timestamp.dt.year
        )

        self.df["is_weekend"] = (
            timestamp.dt.dayofweek >= 5
        ).astype(int)

        return self

    # ------------------------------------------------
    # Cyclical features
    # ------------------------------------------------

    def create_cyclical_features(self):

        self.df["hour_sin"] = np.sin(
            2 * np.pi *
            self.df["hour"] / 24
        )

        self.df["hour_cos"] = np.cos(
            2 * np.pi *
            self.df["hour"] / 24
        )

        self.df["day_sin"] = np.sin(
            2 * np.pi *
            self.df["day_of_week"] / 7
        )

        self.df["day_cos"] = np.cos(
            2 * np.pi *
            self.df["day_of_week"] / 7
        )

        self.df["month_sin"] = np.sin(
            2 * np.pi *
            self.df["month"] / 12
        )

        self.df["month_cos"] = np.cos(
            2 * np.pi *
            self.df["month"] / 12
        )

        return self

    # ------------------------------------------------
    # Lag features
    # ------------------------------------------------

    def create_price_lags(
        self,
        column="price_eur_mwh"
    ):

        if column not in self.df.columns:
            return self

        self.df["price_lag_1h"] = (
            self.df[column].shift(1)
        )

        self.df["price_lag_2h"] = (
            self.df[column].shift(2)
        )

        self.df["price_lag_24h"] = (
            self.df[column].shift(24)
        )

        self.df["price_lag_48h"] = (
            self.df[column].shift(48)
        )

        self.df["price_lag_7d"] = (
            self.df[column].shift(24 * 7)
        )

        return self

    # ------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------

    def create_rolling_features(
        self,
        column="price_eur_mwh"
    ):

        if column not in self.df.columns:
            return self

        self.df["price_rolling_mean_24h"] = (
            self.df[column]
            .shift(1)
            .rolling(24)
            .mean()
        )

        self.df["price_rolling_std_24h"] = (
            self.df[column]
            .shift(1)
            .rolling(24)
            .std()
        )

        self.df["price_rolling_mean_7d"] = (
            self.df[column]
            .shift(1)
            .rolling(24 * 7)
            .mean()
        )

        return self

    # ------------------------------------------------
    # Final cleanup
    # ------------------------------------------------

    def final_cleanup(self):

        self.df = self.df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return self

    # ------------------------------------------------
    # Return DataFrame
    # ------------------------------------------------

    def get_data(self):

        return self.df