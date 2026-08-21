import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def price_summary(df):

    print("\nPRICE SUMMARY")
    print("=" * 50)

    print(
        df["price_eur_mwh"]
        .describe()
    )

    print("\nNegative prices:")

    print(
        (
            df["price_eur_mwh"] < 0
        ).sum()
    )

    print("\nMissing prices:")

    print(
        df["price_eur_mwh"]
        .isna()
        .sum()
    )


def plot_price(df):

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        df["timestamp"],
        df["price_eur_mwh"]
    )

    plt.title(
        "European Day-Ahead Electricity Price"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Price (€ / MWh)"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.show()


def plot_load(df):

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        df["timestamp"],
        df["load_mw"]
    )

    plt.title(
        "Electricity Load"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Load (MW)"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.show()


def plot_price_vs_load(df):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        df["load_mw"],
        df["price_eur_mwh"],
        alpha=0.5
    )

    plt.title(
        "Electricity Price vs Load"
    )

    plt.xlabel(
        "Load (MW)"
    )

    plt.ylabel(
        "Price (€ / MWh)"
    )

    plt.tight_layout()

    plt.show()


def plot_hourly_price(df):

    hourly_price = (
        df.groupby("hour")[
            "price_eur_mwh"
        ]
        .mean()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        hourly_price.index,
        hourly_price.values,
        marker="o"
    )

    plt.title(
        "Average Electricity Price by Hour"
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Average Price (€ / MWh)")

    plt.xticks(range(24))

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

def plot_price_correlation(df):

    numeric_columns = [
        "price_eur_mwh",
        "load_mw",
        "load_forecast_mw",
        "wind_generation_mw",
        "solar_generation_mw",
        "renewable_generation_mw"
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    correlation = (
        df[available_columns]
        .corr()
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Correlation Matrix"
    )

    plt.tight_layout()
    plt.show()


def plot_wind_vs_price(df):

    if "wind_generation_mw" not in df.columns:
        print(
            "wind_generation_mw not available."
        )
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["wind_generation_mw"],
        df["price_eur_mwh"],
        alpha=0.4
    )

    plt.xlabel(
        "Wind Generation (MW)"
    )

    plt.ylabel(
        "Price (€ / MWh)"
    )

    plt.title(
        "Wind Generation vs Electricity Price"
    )

    plt.tight_layout()
    plt.show()


def plot_solar_vs_price(df):

    if "solar_generation_mw" not in df.columns:
        print(
            "solar_generation_mw not available."
        )
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["solar_generation_mw"],
        df["price_eur_mwh"],
        alpha=0.4
    )

    plt.xlabel(
        "Solar Generation (MW)"
    )

    plt.ylabel(
        "Price (€ / MWh)"
    )

    plt.title(
        "Solar Generation vs Electricity Price"
    )

    plt.tight_layout()
    plt.show()