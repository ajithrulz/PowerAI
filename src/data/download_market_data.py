from pathlib import Path

from src.config import (
    BIDDING_ZONES,
    PRICE_DIR,
    LOAD_DIR
)

from src.data.entsoe_client import ENTSOEDataClient


START_DATE = "2025-01-01"
END_DATE = "2025-02-01"


def download_zone_data():

    client = ENTSOEDataClient()

    for zone, zone_name in BIDDING_ZONES.items():

        print("\n" + "=" * 60)
        print(f"Downloading: {zone_name} ({zone})")
        print("=" * 60)

        try:

            # ------------------------------
            # Price
            # ------------------------------

            prices = client.get_day_ahead_prices(
                zone,
                START_DATE,
                END_DATE
            )

            price_file = (
                PRICE_DIR /
                f"{zone}_prices.csv"
            )

            prices.to_csv(
                price_file,
                index=False
            )

            print(
                f"Price data saved: {price_file}"
            )

            # ------------------------------
            # Load
            # ------------------------------

            load = client.get_actual_load(
                zone,
                START_DATE,
                END_DATE
            )

            load_file = (
                LOAD_DIR /
                f"{zone}_load.csv"
            )

            load.to_csv(
                load_file,
                index=False
            )

            print(
                f"Load data saved: {load_file}"
            )

        except Exception as e:

            print(
                f"ERROR for {zone}: {e}"
            )


if __name__ == "__main__":
    download_zone_data()