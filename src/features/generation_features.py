import pandas as pd
import numpy as np


def process_generation(generation_df):

    df = generation_df.copy()

    df.index = pd.to_datetime(
        df.index,
        utc=True
    )

    # Convert MultiIndex columns into
    # simple strings where necessary.

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [
            "_".join(
                str(x)
                for x in col
                if str(x) != "nan"
            )
            for col in df.columns
        ]

    df = df.reset_index()

    # Identify generation columns
    # containing common technologies.

    columns = df.columns.tolist()

    wind_columns = [
        c for c in columns
        if "Wind" in c
    ]

    solar_columns = [
        c for c in columns
        if "Solar" in c
    ]

    nuclear_columns = [
        c for c in columns
        if "Nuclear" in c
    ]

    hydro_columns = [
        c for c in columns
        if "Hydro" in c
    ]

    gas_columns = [
        c for c in columns
        if "Gas" in c
    ]

    coal_columns = [
        c for c in columns
        if "Coal" in c
    ]

    def safe_sum(columns):

        if not columns:
            return 0

        return df[columns].sum(
            axis=1,
            min_count=1
        )

    df["wind_generation_mw"] = (
        safe_sum(wind_columns)
    )

    df["solar_generation_mw"] = (
        safe_sum(solar_columns)
    )

    df["nuclear_generation_mw"] = (
        safe_sum(nuclear_columns)
    )

    df["hydro_generation_mw"] = (
        safe_sum(hydro_columns)
    )

    df["gas_generation_mw"] = (
        safe_sum(gas_columns)
    )

    df["coal_generation_mw"] = (
        safe_sum(coal_columns)
    )

    df["renewable_generation_mw"] = (
        df["wind_generation_mw"]
        + df["solar_generation_mw"]
        + df["hydro_generation_mw"]
    )

    df["renewable_share"] = (
        df["renewable_generation_mw"]
        /
        df["total_generation_mw"]
    )

    return df