# 🚦 Traffic Volume Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Framework-Flask%203.0%2B-black?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/ML%20Library-scikit--learn%201.4%2B-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Charts-Chart.js%20v4-FF6384?logo=chartdotjs&logoColor=white" alt="Chart.js">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Status-Maintained%20%26%20Production%20Ready-emerald" alt="Status">
</p>

An end-to-end Machine Learning web application and Intelligent Transportation System (ITS) platform that predicts hourly traffic volume on primary interstate highway corridors (Interstate 94 Westbound). 

By synthesizing environmental variables (temperature, precipitation, cloud cover) with engineered cyclical temporal indices (diurnal hours, day-of-week, seasonal cycles), the system provides instantaneous, real-time volume forecasts and Level of Service (LOS) traffic tier classifications.

---

### 📖 Complete Project Explanation & Viva Guide
For a comprehensive theoretical breakdown, mathematical derivations, codebase walkthrough, and 50+ viva Q&As for academic defense, read:
👉 **[PROJECT_EXPLANATION_AND_VIVA_GUIDE.md](PROJECT_EXPLANATION_AND_VIVA_GUIDE.md)** &nbsp;|&nbsp; 📊 **[Academic Project Report](reports/PROJECT_REPORT.md)**

---

## ✨ Key Features

- ⚡ **Real-Time Traffic Prediction** — Enter weather and calendar parameters in an intuitive UI to instantly forecast roadway vehicle count (veh/h).
- 🚦 **Intelligent Congestion Classification** — Automatic categorization into **Low** (< 2,000 veh/h), **Medium** (2,000 – 4,500 veh/h), and **High** (> 4,500 veh/h) traffic flow with visual progress gauge.
- 📈 **Interactive Analytics Dashboard** — Client-side Chart.js visualizations covering:
  - 24-hour diurnal traffic profiles (morning and evening rush-hour curves).
  - Day-of-week demand distributions (weekday vs. weekend patterns).
  - Monthly seasonal variations.
  - Histogram of traffic volume frequencies.
  - Actual vs. Predicted scatter plot with reference 45° line.
- 🔌 **RESTful API** — JSON-based endpoints for automated integration with Intelligent Transportation Systems (ITS) and traffic controllers.
- 🛡️ **Flexible Input Normalization** — API seamlessly accepts both canonical keys (`temperature_celsius`, `rain_1h`) and shorthand aliases (`temp`, `rain`, `snow`, `clouds`).
- 🚀 **Zero-Config 1-Click Setup** — Automated Windows `run.bat` creates a dedicated virtual environment, installs dependencies, verifies artifacts, and launches the browser.
- 🧪 **Comprehensive Test Suite** — Unit and integration tests covering model inference, metadata consistency, and Flask API endpoints.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                  USER / CLIENT                                    |
|              Web Browser (HTML5 / CSS3 / JavaScript / Chart.js)                   |
+-----------------------------------------------------------------------------------+
                                   │              ▲
       HTTP POST /api/predict      │              │  JSON Prediction & Metadata
       HTTP GET /api/analysis      │              │  JSON Dataset Aggregations
                                   ▼              │
+-----------------------------------------------------------------------------------+
|                             FLASK WEB APPLICATION (app.py)                        |
|                                                                                   |
|  [Input Validator & Alias Normalizer]  ──►  [Feature Engineering Engine]          |
|  - Validates numerical bounds               - Converts Celsius to Kelvin          |
|  - Maps temp, rain, snow, clouds            - Computes sin/cos cyclical harmonics |
+-----------------------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------------------+
|                        SCIKIT-LEARN ESTIMATION PIPELINE                           |
|                                                                                   |
|  [StandardScaler Preprocessor]         ──►  [Multiple Linear Regression Model]    |
|  - Z-score normalization                    - OLS parameter estimation            |
|  - Prevents data leakage                    - Predicts hourly veh/h               |
+-----------------------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------------------+
|                            DATA & ARTIFACT REPOSITORY                             |
|  • models/traffic_volume_model.joblib   • results/model_metrics.json              |
|  • models/model_metadata.json           • results/test_predictions.csv            |
|  • data/cleaned_traffic_data.csv        • data/Metro_Interstate_Traffic_Volume.csv|
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Quick Start

### 🪟 Windows (Recommended 1-Click Setup)

Double-click `run.bat` or run via Command Prompt:

```cmd
git clone https://github.com/chauhanaryan272-web/traffic-volume-prediction.git
cd traffic-volume-prediction
run.bat
```

`run.bat` automatically:
1. Detects your Python 3.8+ interpreter.
2. Creates an isolated virtual environment (`.venv`).
3. Installs dependencies from `requirements.txt`.
4. Trains the baseline pipeline if artifacts are not already compiled.
5. Launches your default web browser to **`http://localhost:5000`**.

---

### 🐧 macOS / Linux / Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/chauhanaryan272-web/traffic-volume-prediction.git
cd traffic-volume-prediction

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data and train model (if compiling from scratch)
python scripts/prepare_data.py
python scripts/train_model.py

# 5. Start the web server
python app.py
```

Open your browser and navigate to **`http://localhost:5000`**.

---

## 🧪 Running Unit Tests

Run the full test suite using Python's built-in `unittest`:

```bash
# Run all tests (model tests + Flask endpoint tests)
python -m unittest discover -s tests
```

---

## 📊 Dataset & Feature Engineering

The model is trained on the benchmark **Metro Interstate Traffic Volume Dataset** (UCI Machine Learning Repository ID 492) collected by the Minnesota Department of Transportation (MnDOT) at ATR Station 301 on Interstate 94 Westbound (Minneapolis–St. Paul, MN):

| Feature Name | Description | Range / Unit | Transformation |
|---|---|---|---|
| `temp` | Ambient temperature | 220 K – 320 K | Standardized ($Z$-score) |
| `rain_1h` | Liquid precipitation in preceding hour | mm/h | Standardized ($Z$-score) |
| `snow_1h` | Solid precipitation in preceding hour | mm/h | Standardized ($Z$-score) |
| `clouds_all` | Cloud cover percentage | 0 – 100% | Standardized ($Z$-score) |
| `hour` | Hour of day | 0 – 23 | $\sin\left(\frac{2\pi \cdot h}{24}\right), \cos\left(\frac{2\pi \cdot h}{24}\right)$ |
| `day_of_week` | Day of week | 0 (Mon) – 6 (Sun) | $\sin\left(\frac{2\pi \cdot dow}{7}\right), \cos\left(\frac{2\pi \cdot dow}{7}\right)$ |
| `month` | Month of year | 1 – 12 | $\sin\left(\frac{2\pi \cdot m}{12}\right), \cos\left(\frac{2\pi \cdot m}{12}\right)$ |
| `is_weekend` | Weekend binary indicator | 0 or 1 | Categorical Flag |
| `is_holiday` | Public holiday indicator | 0 or 1 | Categorical Flag |
| **`traffic_volume`** | **Target Variable** | **0 – 7,280 veh/h** | Hourly continuous volume |

### Cyclical Trigonometric Transformation
Time components are periodic. Linear scaling creates an artificial discontinuity between 23:00 (11 PM) and 00:00 (Midnight). We project periodic attributes onto the Cartesian unit circle:

$$\text{hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

---

## 🤖 Model Performance Benchmark

The model uses a chronological 80/20 train-test split (earlier 38,563 hours for training; subsequent 9,641 hours for holdout testing) to prevent lookahead data leakage:

| Metric | Training Set | Testing Set (Holdout) | Engineering Interpretation |
|:---|:---:|:---:|:---|
| **MAE** | 694.70 veh/h | **693.92 veh/h** | Average deviation of ~694 vehicles per hour |
| **RMSE** | 928.63 veh/h | **925.44 veh/h** | Penalizes peak-period deviations |
| **$R^2$ Score** | 0.6629 | **0.6644** | Explains 66.4% of total traffic variance |

---

## 🔌 REST API Reference

### 1. Predict Traffic Volume
- **Endpoint:** `POST /api/predict`
- **Content-Type:** `application/json`

**Request Body Example:**
```json
{
  "temperature_celsius": 22.5,
  "rain_1h": 0.0,
  "snow_1h": 0.0,
  "clouds_all": 25,
  "hour": 8,
  "day_of_week": 1,
  "month": 6,
  "is_weekend": 0,
  "is_holiday": 0
}
```
*(Also supports shorthand aliases: `temp`, `rain`, `snow`, `clouds`)*

**Response Example (200 OK):**
```json
{
  "status": "success",
  "prediction": 4821.50,
  "inputs_received": {
    "temperature_celsius": 22.5,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 25,
    "hour": 8,
    "day_of_week": 1,
    "month": 6,
    "is_weekend": 0,
    "is_holiday": 0
  }
}
```

### 2. Retrieve Model Metrics
- **Endpoint:** `GET /api/metrics`
- **Response:** JSON dictionary with train and test evaluation figures (MAE, RMSE, R²).

### 3. Retrieve Dataset Aggregations
- **Endpoint:** `GET /api/analysis`
- **Response:** Cached precomputed hourly profiles, weekday/weekend averages, and volume distribution frequencies for charts.

---

## 📁 Repository Structure

```text
traffic-volume-prediction/
├── app.py                             # Flask application & REST API server
├── requirements.txt                   # Pip package dependencies
├── run.bat                            # 1-click Windows setup & launcher
├── README.md                          # Documentation & project overview
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore definitions
├── PROJECT_EXPLANATION_AND_VIVA_GUIDE.md # Comprehensive 50+ viva guide & theory
│
├── data/                              # Datasets and data dictionary
│   ├── DATASET_README.md              # Dataset metadata & citation
│   ├── Metro_Interstate_Traffic_Volume.csv # Raw UCI dataset
│   └── cleaned_traffic_data.csv       # Preprocessed & engineered dataset
│
├── models/                            # Trained model artifacts
│   ├── traffic_volume_model.joblib    # Serialized scikit-learn pipeline
│   └── model_metadata.json            # Pipeline hyperparameters & feature schema
│
├── results/                           # Test results and evaluation metrics
│   ├── model_metrics.json             # MAE, RMSE, and R² scores
│   └── test_predictions.csv           # Test set actual vs. predicted values
│
├── notebooks/                         # Interactive Jupyter notebooks
│   └── Traffic_Volume_Prediction.ipynb # End-to-end EDA and ML workflow
│
├── reports/                           # Academic engineering documentation
│   └── PROJECT_REPORT.md              # Comprehensive B.Tech project report
│
├── scripts/                           # Automation scripts
│   ├── prepare_data.py                # Data cleaning & cyclical feature engineering
│   └── train_model.py                 # Pipeline training & artifact serialization
│
├── static/                            # Front-end static assets
│   ├── css/
│   │   └── style.css                  # Custom styling & responsive layouts
│   └── js/
│       └── main.js                    # Mobile navigation & UI helpers
│
├── templates/                         # Jinja2 HTML templates
│   ├── base.html                      # Base layout & navigation bar
│   ├── index.html                     # Hero & landing page
│   ├── predict.html                   # Interactive prediction form & result card
│   ├── dashboard.html                 # Analytics dashboard & Chart.js charts
│   └── about.html                     # Civil engineering context & architecture
│
└── tests/                             # Automated test suite
    ├── __init__.py
    ├── test_model.py                  # Model loading & inference tests
    └── test_app.py                    # Flask endpoints & API validation tests
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Acknowledgements

- **Author:** [Aryan Chauhan](https://github.com/chauhanaryan272-web)
- **Dataset Provider:** Minnesota Department of Transportation (MnDOT) & UCI Machine Learning Repository.
- **Academic Domain:** Civil Engineering (Transportation & ITS) & Computer Science (Machine Learning).
