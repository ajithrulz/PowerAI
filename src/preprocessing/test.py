import pandas as pd

from src.preprocessing.data_quality import (
    check_data_quality
)

df = pd.read_csv(
    "data/raw/prices/DE_LU_prices.csv"
)

check_data_quality(df)