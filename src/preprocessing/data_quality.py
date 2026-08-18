import pandas as pd


def check_data_quality(df):

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
        .head(20)
    )

    print("\nDuplicate rows:")
    print(
        df.duplicated().sum()
    )

    if "timestamp" in df.columns:

        timestamps = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        print("\nTimestamp range:")
        print(
            timestamps.min(),
            "to",
            timestamps.max()
        )

        print("\nTimestamp frequency:")
        print(
            timestamps
            .diff()
            .value_counts()
            .head()
        )

    if "price_eur_mwh" in df.columns:

        print("\nPrice statistics:")

        print(
            df["price_eur_mwh"]
            .describe()
        )

        print("\nNegative prices:")

        negative_count = (
            df["price_eur_mwh"] < 0
        ).sum()

        print(negative_count)

    print("\n" + "=" * 60)