# Metro Interstate Traffic Volume Dataset

## Source

| Field       | Value                                                                                 |
|-------------|---------------------------------------------------------------------------------------|
| **Name**    | Metro Interstate Traffic Volume                                                       |
| **Source**  | UCI Machine Learning Repository                                                       |
| **URL**     | <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume>             |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0)                            |

## Description

Hourly Interstate 94 Westbound traffic volume for MN DoT ATR station 301, roughly
midway between Minneapolis and St Paul, MN. The dataset includes weather features
sourced from OpenWeatherMap that correspond to each hourly traffic observation.

## Geographic Context

- **Road:** Interstate 94 (I-94), Westbound
- **Location:** MN DoT ATR station 301, approximately midway between Minneapolis and St Paul, Minnesota, USA
- **Coordinates:** ~44.88°N, 93.22°W

## Time Period

2012-10-02 to 2018-09-30 (hourly observations, ~48,204 rows).

## Columns

| Column                  | Type        | Unit          | Description                                                                 |
|-------------------------|-------------|---------------|-----------------------------------------------------------------------------|
| `holiday`               | Categorical | —             | Name of the US holiday observed, or `"None"` if not a holiday.              |
| `temp`                  | Numeric     | Kelvin (K)    | Average temperature for the hour.                                           |
| `rain_1h`               | Numeric     | mm            | Amount of rainfall in the last hour.                                        |
| `snow_1h`               | Numeric     | mm            | Amount of snowfall in the last hour.                                        |
| `clouds_all`            | Numeric     | %             | Percentage cloud cover (0–100).                                             |
| `weather_main`          | Categorical | —             | Short textual weather description (e.g., Clear, Clouds, Rain).              |
| `weather_description`   | Categorical | —             | Longer / more detailed weather description.                                 |
| `date_time`             | Datetime    | —             | Timestamp of the observation (hourly).                                      |
| `traffic_volume`        | Numeric     | vehicles/hour | Hourly I-94 ATR 301 reported westbound traffic volume.                      |

## Synthetic Data Notice

> [!IMPORTANT]
> This project uses a **synthetic dataset** generated to match the structure,
> column names, data types, and approximate statistical distributions of the
> original UCI dataset. It is intended for **demonstration and development
> purposes only** and does not contain real traffic or weather observations.
>
> The synthetic generation logic lives in
> [`scripts/prepare_data.py`](../scripts/prepare_data.py) → `generate_synthetic_dataset()`.
> A fixed random seed (`42`) is used for full reproducibility.
