import pandas as pd

from src.eda.analysis import (
    price_summary,
    plot_price,
    plot_load,
    plot_price_vs_load,
    plot_hourly_price,
    plot_price_correlation,
    plot_wind_vs_price,
    plot_solar_vs_price
)


# --------------------------------------------------
# Load master dataset
# --------------------------------------------------

file = (
    "data/processed/master/"
    "DE_LU_master.csv"
)

print("Reading:")
print(file)

df = pd.read_csv(file)


# --------------------------------------------------
# Timestamp
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)


# --------------------------------------------------
# Create hour if it doesn't exist
# --------------------------------------------------

if "hour" not in df.columns:

    df["hour"] = (
        df["timestamp"]
        .dt.hour
    )


# --------------------------------------------------
# Sort by timestamp
# --------------------------------------------------

df = df.sort_values(
    "timestamp"
)


# --------------------------------------------------
# Display basic information
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 1. Price summary
# --------------------------------------------------

price_summary(df)


# --------------------------------------------------
# 2. Price over time
# --------------------------------------------------

plot_price(df)


# --------------------------------------------------
# 3. Load over time
# --------------------------------------------------

if "load_mw" in df.columns:

    plot_load(df)


# --------------------------------------------------
# 4. Price vs Load
# --------------------------------------------------

if "load_mw" in df.columns:

    plot_price_vs_load(df)


# --------------------------------------------------
# 5. Hourly price behavior
# --------------------------------------------------

plot_hourly_price(df)


# --------------------------------------------------
# 6. Correlation analysis
# --------------------------------------------------

plot_price_correlation(df)


# --------------------------------------------------
# 7. Wind impact
# --------------------------------------------------

plot_wind_vs_price(df)


# --------------------------------------------------
# 8. Solar impact
# --------------------------------------------------

plot_solar_vs_price(df)