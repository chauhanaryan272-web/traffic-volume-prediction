# 🚦 Traffic Volume Prediction System

> **Python 3.8+** &nbsp;|&nbsp; **Flask** &nbsp;|&nbsp; **scikit-learn** &nbsp;|&nbsp; **Chart.js** &nbsp;|&nbsp; **Civil Engineering ITS**

An end-to-end machine-learning web application that predicts hourly traffic volume on roadway segments (Interstate 94 Westbound). Enter road and weather conditions, and the system returns an estimated vehicle count along with a traffic-level classification.

---

### 📖 Complete Project Explanation & Viva Guide
For a comprehensive, beginner-to-expert theoretical breakdown, math derivations, codebase walkthrough, and 50+ viva Q&As, please read:
👉 **[PROJECT_EXPLANATION_AND_VIVA_GUIDE.md](PROJECT_EXPLANATION_AND_VIVA_GUIDE.md)**

---

## ✨ Features

- **Instant Predictions** – Submit road parameters via a clean web form and receive real-time volume estimates.
- **Interactive Dashboards** – Explore historical traffic trends with Chart.js visualizations.
- **Model Metrics** – View R², MAE, and RMSE for the trained model at a glance.
- **RESTful API** – Programmatic access to predictions through JSON endpoints.
- **One-Click Setup** – `run.bat` handles environment creation, dependency installation, data generation, model training, and server startup.

---

## 🚀 Quick Start

### Windows (recommended)

```
git clone <repo-url>
cd traffic-volume-prediction
run.bat
```

`run.bat` will automatically:

1. Detect your Python installation.
2. Create a virtual environment (`.venv`).
3. Install all dependencies from `requirements.txt`.
4. Generate a synthetic dataset and train the model (first run only).
5. Open your browser to **http://localhost:5000**.

### Manual Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data and train the model
python scripts/prepare_data.py
python scripts/train_model.py

# 4. Run the application
python app.py
```

Then visit **http://localhost:5000** in your browser.

---

## 📁 Project Structure

```
traffic-volume-prediction/
├── app.py                             # Flask application entry point & REST API
├── requirements.txt                   # Python package dependencies
├── run.bat                            # One-click Windows batch launcher
├── README.md                          # Project documentation
├── PROJECT_EXPLANATION_AND_VIVA_GUIDE.md # Complete project theory & viva guide
│
├── models/                            # Trained model artifacts & metadata
│   ├── traffic_volume_model.joblib
│   └── model_metadata.json
│
├── results/                           # Evaluation metrics & test predictions
│   ├── model_metrics.json
│   └── test_predictions.csv
│
├── notebooks/                         # Interactive Jupyter research notebook
│   └── Traffic_Volume_Prediction.ipynb
│
├── reports/                           # Academic technical report
│   └── PROJECT_REPORT.md
│
├── data/                              # Datasets & documentation
│   ├── DATASET_README.md
│   ├── Metro_Interstate_Traffic_Volume.csv
│   └── cleaned_traffic_data.csv
│
├── scripts/                           # Data preparation & training scripts
│   ├── prepare_data.py
│   └── train_model.py
│
├── static/
│   ├── css/
│   │   └── style.css                  # Application stylesheet
│   └── js/
│       └── main.js                    # Client-side UI helpers
│
├── templates/                         # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── dashboard.html
│   └── about.html
│
└── tests/                             # Unit tests
    ├── __init__.py
    └── test_model.py
```

---

## 📊 Dataset

The system ships with a **synthetic dataset** generator (`scripts/prepare_data.py`) that produces realistic traffic patterns including:

| Feature | Description |
|---|---|
| `hour` | Hour of the day (0 – 23) |
| `day_of_week` | Day of the week (0 = Monday … 6 = Sunday) |
| `month` | Month of the year (1 – 12) |
| `is_weekend` | Boolean flag for weekends |
| `temperature` | Temperature in °F |
| `weather_condition` | Categorical: Clear, Cloudy, Rain, Snow, Fog |
| `is_holiday` | Boolean flag for holidays |
| `traffic_volume` | **Target** – hourly vehicle count |

---

## 🤖 Machine Learning Model

- **Primary Deployed Model:** Multiple Linear Regression with Cyclical Feature Transformation & `StandardScaler` Pipeline
- **Benchmark Comparison:** Random Forest Regressor (explained in viva guide)
- **Training Script:** `scripts/train_model.py`
- **Serialization:** `joblib` (`models/traffic_volume_model.joblib`)
- **Metadata:** `models/model_metadata.json`
- **Splitting Strategy:** 80/20 Chronological Split (No Shuffling)
- **Evaluation Metrics (Holdout Test Set):**
  - **MAE:** 693.92 veh/h
  - **RMSE:** 925.44 veh/h
  - **R² Score:** 0.6644 (66.4% variance explained)

The trained pipeline is persisted to `models/traffic_volume_model.joblib` and loaded at application startup. If not yet compiled, `app.py` falls back to an internal calibrated baseline estimator.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing / Home page (`index.html`) |
| `GET` | `/predict` | Prediction form interface (`predict.html`) |
| `POST` | `/api/predict` | JSON prediction endpoint (validates bounds, converts °C to K, returns veh/h) |
| `GET` | `/dashboard` | Interactive Chart.js analytics dashboard (`dashboard.html`) |
| `GET` | `/about` | Civil engineering context and project documentation (`about.html`) |
| `GET` | `/api/metrics` | Model performance metrics JSON (`results/model_metrics.json`) |
| `GET` | `/api/analysis` | Aggregated dataset analytics JSON (hourly, daily, monthly averages, distribution, scatter) |

---

## 📸 Screenshots

> _Screenshots will be added here after the first run._

---

## 🛠 Technologies Used

- **Python 3.8+**
- **Flask** – lightweight web framework
- **scikit-learn** – machine learning
- **pandas / NumPy** – data manipulation
- **joblib** – model serialization
- **Chart.js** – client-side charting
- **HTML5 / CSS3 / JavaScript**

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
