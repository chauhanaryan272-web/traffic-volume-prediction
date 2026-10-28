"""
prepare_data.py — Data preparation script for the Traffic Volume Prediction project.

This script:
  1. Generates a synthetic dataset modeled on the UCI Metro Interstate Traffic Volume
     dataset (or loads an existing local copy).
  2. Cleans the data (duplicates, missing values, datetime conversion, sorting).
  3. Engineers temporal and cyclical features.
  4. Saves the cleaned dataset to data/cleaned_traffic_data.csv.
  5. Prints summary statistics.

Dataset reference:
  https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
"""

import os
import sys
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "Metro_Interstate_Traffic_Volume.csv")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_traffic_data.csv")

RANDOM_SEED = 42

# US holidays that appear in the original dataset
US_HOLIDAYS = [
    "New Years Day",
    "Martin Luther King Jr Day",
    "Washington's Birthday",
    "Memorial Day",
    "Independence Day",
    "Labor Day",
    "Columbus Day",
    "Veterans Day",
    "Thanksgiving Day",
    "Christmas Day",
    "State Fair",
]

WEATHER_OPTIONS = ["Clear", "Clouds", "Rain", "Snow", "Fog", "Mist"]

# Detailed weather descriptions keyed by weather_main
WEATHER_DESCRIPTIONS = {
    "Clear": ["sky is clear"],
    "Clouds": [
        "few clouds",
        "scattered clouds",
        "broken clouds",
        "overcast clouds",
    ],
    "Rain": ["light rain", "moderate rain", "heavy intensity rain"],
    "Snow": ["light snow", "snow", "heavy snow"],
    "Fog": ["fog", "freezing fog"],
    "Mist": ["mist", "haze"],
}


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic dataset that mirrors the structure and realistic
    statistical properties of the UCI Metro Interstate Traffic Volume dataset.

    Returns:
        pd.DataFrame with the same columns as the original dataset.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Date range — hourly from 2012-10-02 to 2018-09-30
    # ------------------------------------------------------------------
    full_range = pd.date_range(start="2012-10-02", end="2018-09-30 23:00:00", freq="h")

    # Sample exactly 48,204 rows (original dataset size) from the range
    n_rows = 48_204
    if len(full_range) >= n_rows:
        indices = np.sort(rng.choice(len(full_range), size=n_rows, replace=False))
        date_times = full_range[indices]
    else:
        # If the range is shorter (shouldn't happen), allow duplicates
        indices = np.sort(rng.choice(len(full_range), size=n_rows, replace=True))
        date_times = full_range[indices]

    hours = date_times.hour
    day_of_week = getattr(date_times, "day_of_week", date_times.dayofweek)  # 0=Monday … 6=Sunday
    months = date_times.month

    # ------------------------------------------------------------------
    # 2. Temperature (Kelvin) — normal around 280 K, clipped to [220, 320]
    # ------------------------------------------------------------------
    temp = rng.normal(loc=280.0, scale=20.0, size=n_rows)
    temp = np.clip(temp, 220.0, 320.0)

    # ------------------------------------------------------------------
    # 3. Rain (mm in last hour) — mostly 0, some up to 50 mm
    # ------------------------------------------------------------------
    rain_indicator = rng.random(size=n_rows) < 0.10  # ~10 % chance of rain
    rain_1h = np.where(
        rain_indicator,
        np.clip(rng.exponential(scale=5.0, size=n_rows), 0, 50),
        0.0,
    )

    # ------------------------------------------------------------------
    # 4. Snow (mm in last hour) — mostly 0, rare values up to 5 mm
    # ------------------------------------------------------------------
    snow_indicator = rng.random(size=n_rows) < 0.03  # ~3 % chance of snow
    snow_1h = np.where(
        snow_indicator,
        np.clip(rng.exponential(scale=1.0, size=n_rows), 0, 5),
        0.0,
    )

    # ------------------------------------------------------------------
    # 5. Cloud cover (%) — uniform 0–100
    # ------------------------------------------------------------------
    clouds_all = rng.integers(0, 101, size=n_rows)

    # ------------------------------------------------------------------
    # 6. Holiday — "None" for most rows, ~2 % assigned a random US holiday
    # ------------------------------------------------------------------
    holiday_mask = rng.random(size=n_rows) < 0.02
    holiday = np.where(
        holiday_mask,
        rng.choice(US_HOLIDAYS, size=n_rows),
        "None",
    )

    # ------------------------------------------------------------------
    # 7. Weather — random choice from main categories + descriptions
    # ------------------------------------------------------------------
    weather_main = rng.choice(WEATHER_OPTIONS, size=n_rows)
    weather_description = np.array(
        [rng.choice(WEATHER_DESCRIPTIONS[w]) for w in weather_main]
    )

    # ------------------------------------------------------------------
    # 8. Traffic volume — realistic hourly pattern
    #    Peak hours: 7–9 AM and 4–6 PM (indices 7,8,9 and 16,17,18)
    #    Low hours:  2–5 AM
    #    Weekends are ~40 % lower than weekdays
    # ------------------------------------------------------------------
    # Base volume curve by hour (0–23)
    hourly_base = np.array(
        [
            800,   # 0
            500,   # 1
            300,   # 2
            200,   # 3
            250,   # 4
            600,   # 5
            2000,  # 6
            4500,  # 7
            5500,  # 8
            4000,  # 9
            3200,  # 10
            3400,  # 11
            3600,  # 12
            3500,  # 13
            3800,  # 14
            4500,  # 15
            5800,  # 16
            5500,  # 17
            4200,  # 18
            3200,  # 19
            2400,  # 20
            1800,  # 21
            1400,  # 22
            1000,  # 23
        ],
        dtype=float,
    )

    base = hourly_base[hours]

    # Weekend factor
    is_weekend = np.isin(day_of_week, [5, 6])
    base = np.where(is_weekend, base * 0.60, base)

    # Weather penalty
    weather_factor = np.ones(n_rows)
    weather_factor[weather_main == "Rain"] = 0.90
    weather_factor[weather_main == "Snow"] = 0.80
    weather_factor[weather_main == "Fog"] = 0.85
    base = base * weather_factor

    # Add Gaussian noise (std ~10 % of base)
    noise = rng.normal(loc=0, scale=0.10, size=n_rows)
    traffic_volume = base * (1 + noise)
    traffic_volume = np.clip(traffic_volume, 0, 7280).astype(int)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "holiday": holiday,
            "temp": np.round(temp, 2),
            "rain_1h": np.round(rain_1h, 2),
            "snow_1h": np.round(snow_1h, 2),
            "clouds_all": clouds_all,
            "weather_main": weather_main,
            "weather_description": weather_description,
            "date_time": date_times,
            "traffic_volume": traffic_volume,
        }
    )

    return df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw traffic volume DataFrame.

    Steps:
      1. Drop duplicate rows.
      2. Drop rows with any NaN values.
      3. Convert ``date_time`` column to datetime.
      4. Sort by ``date_time``.

    Args:
        df: Raw DataFrame loaded from the CSV.

    Returns:
        Cleaned DataFrame.
    """
    initial_shape = df.shape
    print(f"  Raw dataset shape: {initial_shape}")

    # 1. Drop duplicates
    df = df.drop_duplicates()
    print(f"  After dropping duplicates: {df.shape}")

    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    print(f"  Dropped {missing_before} NaN cells → shape: {df.shape}")

    # 3. Convert date_time to datetime
    df["date_time"] = pd.to_datetime(df["date_time"])

    # 4. Sort by date_time
    df = df.sort_values("date_time").reset_index(drop=True)
    print("  Sorted by date_time.")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, cyclical, and categorical features.

    New columns:
      - hour, day_of_week, month, is_weekend
      - hour_sin, hour_cos  (cyclical encoding of hour)
      - month_sin, month_cos (cyclical encoding of month)
      - dow_sin, dow_cos     (cyclical encoding of day_of_week)
      - is_holiday           (1 if holiday != "None", else 0)

    Numeric columns kept as-is: temp, rain_1h, snow_1h, clouds_all.

    Args:
        df: Cleaned DataFrame with a ``date_time`` column.

    Returns:
        DataFrame augmented with engineered features.
    """
    # Temporal extraction
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.day_of_week if hasattr(df["date_time"].dt, "day_of_week") else df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding — hour (period 24)
    df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)

    # Cyclical encoding — month (period 12)
    df["month_sin"] = np.sin(2 * math.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * math.pi * df["month"] / 12)

    # Cyclical encoding — day_of_week (period 7)
    df["dow_sin"] = np.sin(2 * math.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * math.pi * df["day_of_week"] / 7)

    # Holiday indicator
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full data-preparation pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1 — Generate (or load) the raw dataset
    if os.path.exists(RAW_CSV_PATH):
        print(f"[INFO] Loading existing raw CSV from {RAW_CSV_PATH}")
        df = pd.read_csv(RAW_CSV_PATH, keep_default_na=False)
    else:
        print("[INFO] Generating synthetic Metro Interstate Traffic Volume dataset …")
        df = generate_synthetic_dataset()
        df.to_csv(RAW_CSV_PATH, index=False)
        print(f"[INFO] Saved synthetic CSV to {RAW_CSV_PATH}")

    # Step 2 — Clean the data
    print("\n[STEP] Cleaning data …")
    df = clean_data(df)

    # Step 3 — Feature engineering
    print("\n[STEP] Engineering features …")
    df = engineer_features(df)

    # Step 4 — Save cleaned dataset
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"\n[INFO] Saved cleaned dataset to {CLEANED_CSV_PATH}")

    # Step 5 — Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Shape:          {df.shape}")
    print(f"Columns:        {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nNumeric summary:\n{df.describe()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
