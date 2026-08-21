import pandas as pd
from pathlib import Path

from src.features.generation_features import (
    process_generation
)


def prepare_timestamp(df):

    df = df.copy()

    if "timestamp" not in df.columns:

        raise ValueError(
            "Dataset does not contain a timestamp column."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


def build_master_dataset(
    price_file,
    load_file,
    load_forecast_file,
    generation_file,
    output_file
):

    print("=" * 60)
    print("BUILDING MASTER DATASET")
    print("=" * 60)

    # ==================================================
    # 1. Read raw datasets
    # ==================================================

    print("\nReading price data...")

    prices = pd.read_csv(
        price_file
    )

    print(
        f"Price rows: {len(prices)}"
    )

    print("\nReading actual load...")

    load = pd.read_csv(
        load_file
    )

    print(
        f"Load rows: {len(load)}"
    )

    print("\nReading load forecast...")

    load_forecast = pd.read_csv(
        load_forecast_file
    )

    print(
        f"Load forecast rows: "
        f"{len(load_forecast)}"
    )

    print("\nReading generation...")

    generation = pd.read_csv(
        generation_file
    )

    print(
        f"Generation rows: "
        f"{len(generation)}"
    )

    # ==================================================
    # 2. Prepare timestamps
    # ==================================================

    prices = prepare_timestamp(
        prices
    )

    load = prepare_timestamp(
        load
    )

    load_forecast = prepare_timestamp(
        load_forecast
    )

    generation = prepare_timestamp(
        generation
    )

    # ==================================================
    # 3. Prepare price
    # ==================================================

    prices = prices[
        [
            "timestamp",
            "price_eur_mwh"
        ]
    ]

    prices = (
        prices
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
    )

    # ==================================================
    # 4. Prepare actual load
    # ==================================================

    load = load[
        [
            "timestamp",
            "load_mw"
        ]
    ]

    load = (
        load
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
    )

    # ==================================================
    # 5. Prepare load forecast
    # ==================================================

    load_forecast = load_forecast[
        [
            "timestamp",
            "load_forecast_mw"
        ]
    ]

    load_forecast = (
        load_forecast
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
    )

    # ==================================================
    # 6. Process generation
    # ==================================================

    print("\nProcessing generation data...")

    generation = process_generation(
        generation
    )

    print("\nProcessed generation columns:")

    print(
        generation.columns.tolist()
    )

    # ==================================================
    # 7. Merge price + load
    # ==================================================

    master = prices.merge(
        load,
        on="timestamp",
        how="left"
    )

    # ==================================================
    # 8. Merge load forecast
    # ==================================================

    master = master.merge(
        load_forecast,
        on="timestamp",
        how="left"
    )

    # ==================================================
    # 9. Merge generation
    # ==================================================

    master = master.merge(
        generation,
        on="timestamp",
        how="left"
    )

    # ==================================================
    # 10. Sort
    # ==================================================

    master = (
        master
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ==================================================
    # 11. Load forecast error
    # ==================================================

    master["load_forecast_error"] = (
        master["load_mw"]
        -
        master["load_forecast_mw"]
    )

    master["load_forecast_error_pct"] = (
        master["load_forecast_error"]
        /
        master["load_forecast_mw"]
        .replace(0, pd.NA)
    ) * 100

    # ==================================================
    # 12. Calendar features
    # ==================================================

    master["hour"] = (
        master["timestamp"].dt.hour
    )

    master["day_of_week"] = (
        master["timestamp"].dt.dayofweek
    )

    master["month"] = (
        master["timestamp"].dt.month
    )

    master["year"] = (
        master["timestamp"].dt.year
    )

    master["is_weekend"] = (
        master["day_of_week"] >= 5
    ).astype(int)

    # ==================================================
    # 13. Cyclical time features
    # ==================================================

    import numpy as np

    master["hour_sin"] = np.sin(
        2 * np.pi *
        master["hour"] / 24
    )

    master["hour_cos"] = np.cos(
        2 * np.pi *
        master["hour"] / 24
    )

    master["day_sin"] = np.sin(
        2 * np.pi *
        master["day_of_week"] / 7
    )

    master["day_cos"] = np.cos(
        2 * np.pi *
        master["day_of_week"] / 7
    )

    # ==================================================
    # 14. Price lag features
    # ==================================================

    master["price_lag_1h"] = (
        master["price_eur_mwh"]
        .shift(1)
    )

    master["price_lag_2h"] = (
        master["price_eur_mwh"]
        .shift(2)
    )

    master["price_lag_24h"] = (
        master["price_eur_mwh"]
        .shift(24)
    )

    master["price_lag_48h"] = (
        master["price_eur_mwh"]
        .shift(48)
    )

    master["price_lag_7d"] = (
        master["price_eur_mwh"]
        .shift(24 * 7)
    )

    # ==================================================
    # 15. Rolling price statistics
    # ==================================================

    master["price_rolling_mean_24h"] = (
        master["price_eur_mwh"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    master["price_rolling_std_24h"] = (
        master["price_eur_mwh"]
        .shift(1)
        .rolling(24)
        .std()
    )

    master["price_rolling_mean_7d"] = (
        master["price_eur_mwh"]
        .shift(1)
        .rolling(24 * 7)
        .mean()
    )

    # ==================================================
    # 16. Save master dataset
    # ==================================================

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        output_file,
        index=False
    )

    # ==================================================
    # 17. Validation
    # ==================================================

    print("\n" + "=" * 60)
    print("MASTER DATASET VALIDATION")
    print("=" * 60)

    print("\nShape:")
    print(master.shape)

    print("\nImportant columns:")

    important_columns = [
        "price_eur_mwh",
        "load_mw",
        "load_forecast_mw",
        "wind_generation_mw",
        "solar_generation_mw",
        "renewable_generation_mw",
        "total_generation_mw",
        "renewable_share"
    ]

    for column in important_columns:

        if column in master.columns:

            print(
                f"✓ {column}"
            )

        else:

            print(
                f"✗ {column} MISSING"
            )

    print("\nFirst rows:")

    print(
        master[
            [
                column
                for column in important_columns
                if column in master.columns
            ]
        ].head()
    )

    print(
        f"\nSaved master dataset:"
        f"\n{output_file}"
    )

    return master