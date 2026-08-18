import pandas as pd

from src.config import (
    PRICE_DIR,
    FEATURE_DIR
)

from src.preprocessing.preprocess import (
    PowerDataPreprocessor
)


def preprocess_prices():

    input_file = (
        PRICE_DIR /
        "DE_LU_prices.csv"
    )

    output_file = (
        FEATURE_DIR /
        "DE_LU_price_features.csv"
    )

    print("Reading:")
    print(input_file)

    df = pd.read_csv(
        input_file
    )

    print(
        f"Original rows: {len(df)}"
    )

    processor = (
        PowerDataPreprocessor(df)

        .process_timestamp()

        .handle_missing_values()

        .handle_price_outliers()

        .create_calendar_features()

        .create_cyclical_features()

        .create_price_lags()

        .create_rolling_features()

        .final_cleanup()
    )

    processed_df = (
        processor
        .get_data()
    )

    # Remove rows where lag values
    # are not available.
    processed_df = (
        processed_df
        .dropna()
        .reset_index(drop=True)
    )

    processed_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Processed rows: {len(processed_df)}"
    )

    print(
        f"Saved: {output_file}"
    )

    print("\nColumns:")
    print(
        processed_df.columns.tolist()
    )


if __name__ == "__main__":
    preprocess_prices()