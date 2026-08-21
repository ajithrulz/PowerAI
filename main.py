import pandas as pd

from src.data.entsoe_client import ENTSOEDataClient
from src.config import PRICE_DIR, LOAD_DIR
from src.config import GENERATION_DIR


def main():

    client = ENTSOEDataClient()

    # ------------------------------------------------
    # Configuration
    # ------------------------------------------------

    zone = "DE_LU"

    start_date = "2025-01-01"
    end_date = "2025-01-08"

    print("=" * 60)
    print("EUROPEAN POWER MARKET DATA COLLECTION")
    print("=" * 60)

    # ------------------------------------------------
    # Day-ahead prices
    # ------------------------------------------------

    print("\nDownloading day-ahead prices...")

    prices = client.get_day_ahead_prices(
        zone,
        start_date,
        end_date
    )

    print(prices.head())
    print(f"Rows: {len(prices)}")

    price_file = (
        PRICE_DIR /
        f"{zone}_day_ahead_prices.csv"
    )

    prices.to_csv(
        price_file,
        index=False
    )

    print(f"Saved: {price_file}")

    # ------------------------------------------------
    # Actual load
    # ------------------------------------------------

    print("\nDownloading actual load...")

    load = client.get_actual_load(
        zone,
        start_date,
        end_date
    )

    print(load.head())
    print(f"Rows: {len(load)}")

    load_file = (
        LOAD_DIR /
        f"{zone}_actual_load.csv"
    )

    load.to_csv(
        load_file,
        index=False
    )

    print(f"Saved: {load_file}")

    print("\nData collection completed.")


    load_forecast = client.get_load_forecast(
        zone,
        start_date,
        end_date
    )

    output_file = (
        LOAD_DIR /
        f"{zone}_load_forecast.csv"
    )

    load_forecast.to_csv(
        output_file,
        index=False
    )

    print(load_forecast.head())
    print(f"Saved: {output_file}")


    generation = client.get_generation(
        zone,
        start_date,
        end_date
    )

    output_file = (
        GENERATION_DIR /
        f"{zone}_generation.csv"
    )

    generation.to_csv(
        output_file,
        index=False
    )

    print(generation.head())



if __name__ == "__main__":
    main()