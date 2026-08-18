import pandas as pd


def prepare_load(
    actual_load,
    forecast_load
):

    actual = actual_load.copy()
    forecast = forecast_load.copy()

    actual["timestamp"] = pd.to_datetime(
        actual["timestamp"],
        utc=True
    )

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"],
        utc=True
    )

    actual = actual[
        [
            "timestamp",
            "load_mw"
        ]
    ]

    forecast = forecast[
        [
            "timestamp",
            "load_forecast_mw"
        ]
    ]

    df = actual.merge(
        forecast,
        on="timestamp",
        how="outer"
    )

    # Load forecast error

    df["load_forecast_error"] = (
        df["load_mw"]
        -
        df["load_forecast_mw"]
    )

    df["load_forecast_error_pct"] = (
        df["load_forecast_error"]
        /
        df["load_forecast_mw"].replace(
            0,
            pd.NA
        )
    ) * 100

    return df