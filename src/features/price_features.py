import pandas as pd


def prepare_prices(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df[
        [
            "timestamp",
            "price_eur_mwh"
        ]
    ]

    return df