from pathlib import Path
import os

from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# ENTSO-E API
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

if not ENTSOE_API_KEY:
    raise ValueError(
        "ENTSOE_API_KEY not found. "
        "Please create a .env file and add your ENTSO-E API token."
    )


# Directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PRICE_DIR = RAW_DATA_DIR / "prices"
LOAD_DIR = RAW_DATA_DIR / "load"
GENERATION_DIR = RAW_DATA_DIR / "generation"
TRANSMISSION_DIR = RAW_DATA_DIR / "transmission"
WEATHER_DIR = RAW_DATA_DIR / "weather"

FEATURE_DIR = PROCESSED_DATA_DIR / "features"
MASTER_DATA_DIR = PROCESSED_DATA_DIR / "master"

MODEL_DIR = PROJECT_ROOT / "models"


# Create directories
for directory in [
    PRICE_DIR,
    LOAD_DIR,
    GENERATION_DIR,
    TRANSMISSION_DIR,
    WEATHER_DIR,
    FEATURE_DIR,
    MASTER_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# European bidding zones
BIDDING_ZONES = {
    "FR": "France",
    "DE_LU": "Germany-Luxembourg",
    "NL": "Netherlands",
    "BE": "Belgium",
    "ES": "Spain",
    "PT": "Portugal",
    "IT_NORTH": "Italy North",
    "AT": "Austria",
    "CH": "Switzerland",
    "PL": "Poland",
    "CZ": "Czech Republic",
    "HU": "Hungary",
    "DK_1": "Denmark West",
    "DK_2": "Denmark East",
    "SE_1": "Sweden North",
    "SE_2": "Sweden Central",
    "SE_3": "Sweden South",
    "SE_4": "Sweden South",
    "NO_1": "Norway South East",
    "NO_2": "Norway South West",
}