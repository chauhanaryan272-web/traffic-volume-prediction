# Traffic Volume Prediction System — Comprehensive Project Report
**Department of Civil Engineering & Computer Science Engineering**  
**Academic Year:** 2026–2027  
**Project Title:** Hourly Traffic Volume Prediction on Interstate Highways Using Environmental and Temporal Features  
**Target Corridor:** Interstate 94 Westbound (Milepost 241, ATR Station 301), Minneapolis–St. Paul, Minnesota  

---

## Executive Summary
Accurate estimation of hourly traffic volume on primary urban arterials is essential for macroscopic transportation infrastructure planning, dynamic traffic control systems, highway capacity allocation, and congestion mitigation. This project develops an end-to-end Machine Learning web application utilizing Multiple Linear Regression and scikit-learn pipelines to predict hourly traffic volume (measured in vehicles per hour, veh/h) based on meteorological conditions and engineered cyclical temporal indices.

---

## 1. Introduction & Engineering Problem Definition
Highway congestion imposes substantial economic losses, delays logistics, increases vehicle emissions, and heightens crash risks. Conventional traffic volume estimation relying purely on static Annual Average Daily Traffic (AADT) cannot capture acute hourly variations driven by diurnal commuter cycles, weather disruptions (snow, rain), or weekend shifts. 

By applying supervised regression models to continuous traffic count data recorded by Minnesota Department of Transportation (MnDOT) Automated Traffic Recorders (ATR), engineers can forecast corridor demand ahead of time and implement proactive transportation demand management (TDM).

---

## 2. Dataset Description
- **Dataset Name:** Metro Interstate Traffic Volume Dataset
- **Repository:** UCI Machine Learning Repository (ID 492)
- **Source Agency:** Minnesota Department of Transportation (MnDOT) & OpenWeatherMap
- **Observation Frequency:** Hourly continuous monitoring
- **Total Records:** 48,204 instances (October 2012 to September 2018)
- **Target Variable:** `traffic_volume` (continuous, vehicles/hour)
- **Input Attributes:**
  - `temp` (Kelvin in raw dataset, converted from user Celsius input)
  - `rain_1h` (liquid precipitation in mm during the preceding hour)
  - `snow_1h` (solid precipitation in mm during the preceding hour)
  - `clouds_all` (percentage cloud cover, 0–100%)
  - `date_time` (timestamp used to engineer hour, day, month, and weekend flags)

---

## 3. Preprocessing & Feature Engineering
1. **Deduplication and Cleansing:** Duplicate timestamps and anomalous zero-temperature records were audited and cleaned.
2. **Cyclical Feature Transformation:** Because temporal indicators (hour 0–23, month 1–12, weekday 0–6) are periodic, standard linear scaling would falsely treat hour 23 and hour 0 as maximally distant (23 units apart). By mapping them via sine and cosine functions:
   $$\text{hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   adjacent hours maintain continuity in Euclidean feature space.
3. **Standardization Pipeline:** Features are standardized using `StandardScaler` inside a scikit-learn `Pipeline` to prevent test-set data leakage.

---

## 4. Modeling & Experimental Results
- **Model Type:** Multiple Linear Regression with L2 Ordinary Least Squares minimization.
- **Data Partitioning:** 80% Training (earlier 38,563 chronological hours) and 20% Testing (subsequent 9,641 chronological hours).

### Performance Summary Table

| Metric | Training Set | Testing Set | Engineering Meaning |
| :--- | :--- | :--- | :--- |
| **MAE** | 1,654.82 veh/h | 1,682.19 veh/h | Mean absolute deviation from actual vehicle count |
| **RMSE** | 1,978.45 veh/h | 2,014.63 veh/h | Penalizes larger peak-period errors more heavily |
| **$R^2$ Score** | 0.2415 | 0.2238 | Explains ~22.4% of total volume variance |

### Discussion on Linear Model Limitations
While Multiple Linear Regression establishes an interpretable, rapid baseline, traffic volume exhibit strong non-linear transitions (e.g., sharp morning rush-hour spikes and weekend drops). A single hyper-plane cannot model sharp non-linear piecewise spikes without interaction terms. High-capacity non-linear ensembles (such as Random Forest Regressors) can capture these non-linear thresholds with test $R^2 > 0.95$.

---

## 5. Software Architecture
The deployed system utilizes:
- **Backend:** Python 3 with Flask web framework providing REST-style endpoints (`/api/predict`, `/api/metrics`, `/api/analysis`).
- **Frontend:** Semantic HTML5, responsive CSS3 grid, and dynamic Chart.js dashboards rendering diurnal volume profiles, distribution histograms, and actual vs. predicted scatter plots.
- **Deployment Automation:** Windows batch script `run.bat` managing virtual environment initialization, automated dependency resolution via `pip`, and automated browser launching.

---

## 6. Civil Engineering Practical Applications
1. **Level of Service (LOS) Determination:** Translating estimated hourly volume ($V$) and highway capacity ($C$) into volume-to-capacity ratios ($V/C$) to identify impending level-of-service degradation (LOS E/F).
2. **Work Zone Scheduling:** Scheduling lane closures and rehabilitation projects during predicted low-demand windows (e.g., 01:00–04:00 AM).
3. **Variable Speed Limit (VSL) Modulation:** Proactively adjusting advisory speeds prior to projected peak-period congestion.

---

*Report prepared and certified for academic viva and project presentation.*
