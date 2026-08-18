from src.config import (
    PRICE_DIR,
    LOAD_DIR,
    GENERATION_DIR,
    MASTER_DATA_DIR
)

from src.features.master_dataset import (
    build_master_dataset
)


zone = "DE_LU"


master = build_master_dataset(

    price_file=(
        PRICE_DIR /
        f"{zone}_prices.csv"
    ),

    load_file=(
        LOAD_DIR /
        f"{zone}_load.csv"
    ),

    load_forecast_file=(
        LOAD_DIR /
        f"{zone}_load_forecast.csv"
    ),

    generation_file=(
        GENERATION_DIR /
        f"{zone}_generation.csv"
    ),

    output_file=(
        MASTER_DATA_DIR /
        f"{zone}_master.csv"
    )
)


print("\nMaster dataset:")
print(master.head())

print("\nDataset information:")
print(master.info())