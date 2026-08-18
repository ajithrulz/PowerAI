import pandas as pd

from pathlib import Path


def build_master_dataset(
    price_file,
    load_file,
    load_forecast_file,
    generation_file,
    output_file
):

    # ----------------------------------------
    # Read data
    # ----------------------------------------

    prices = pd.read_csv(
        price_file
    )

    load = pd.read_csv(
        load_file
    )

    load_forecast = pd.read_csv(
        load_forecast_file
    )

    generation = pd.read_csv(
        generation_file
    )

    # ----------------------------------------
    # Timestamp conversion
    # ----------------------------------------

    for df in [
        prices,
        load,
        load_forecast,
        generation
    ]:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

    # ----------------------------------------
    # Select required columns
    # ----------------------------------------

    prices = prices[
        [
            "timestamp",
            "price_eur_mwh"
        ]
    ]

    load = load[
        [
            "timestamp",
            "load_mw"
        ]
    ]

    load_forecast = load_forecast[
        [
            "timestamp",
            "load_forecast_mw"
        ]
    ]

    # ----------------------------------------
    # Merge price + load
    # ----------------------------------------

    master = prices.merge(
        load,
        on="timestamp",
        how="left"
    )

    master = master.merge(
        load_forecast,
        on="timestamp",
        how="left"
    )

    # ----------------------------------------
    # Merge generation
    # ----------------------------------------

    master = master.merge(
        generation,
        on="timestamp",
        how="left"
    )

    # ----------------------------------------
    # Sort
    # ----------------------------------------

    master = master.sort_values(
        "timestamp"
    )

    # ----------------------------------------
    # Derived variables
    # ----------------------------------------

    if (
        "wind_generation_mw" in master.columns
        and
        "solar_generation_mw" in master.columns
    ):

        master["renewable_generation_mw"] = (
            master["wind_generation_mw"]
            +
            master["solar_generation_mw"]
        )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        output_file,
        index=False
    )

    print(
        f"Master dataset saved: {output_file}"
    )

    print(
        f"Rows: {len(master)}"
    )

    print(
        f"Columns: {len(master.columns)}"
    )

    return master