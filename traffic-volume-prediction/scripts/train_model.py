"""
Train a Linear Regression model for traffic volume prediction.

This script loads cleaned traffic data, builds a scikit-learn pipeline
with StandardScaler and LinearRegression, trains the model, evaluates
performance on train/test sets, and saves all artifacts (model, metadata,
metrics, predictions).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from datetime import datetime
import sklearn


# ---------------------------------------------------------------------------
# Path configuration – all paths are relative to the project root
# (one level up from this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_traffic_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_PATH = os.path.join(MODEL_DIR, "traffic_volume_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "model_metrics.json")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "test_predictions.csv")

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
]

TARGET = "traffic_volume"


def load_data(path: str) -> pd.DataFrame:
    """Load the cleaned traffic dataset from *path* and return a DataFrame."""
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"  → {df.shape[0]:,} rows, {df.shape[1]} columns loaded.")
    return df


def split_data(df: pd.DataFrame, train_frac: float = 0.80):
    """
    Chronological train/test split (no shuffle – important for time-series).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, assumed to be in chronological order.
    train_frac : float
        Fraction of rows to use for training (default 0.80).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    split_idx = int(len(df) * train_frac)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline consisting of:
      1. A ColumnTransformer that applies StandardScaler to all numeric features.
      2. A LinearRegression estimator.
    """
    # All features are numeric after preprocessing, so we scale them all.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return pipeline


def evaluate(y_true, y_pred) -> dict:
    """
    Compute regression metrics.

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def save_artifacts(
    pipeline: Pipeline,
    metrics_train: dict,
    metrics_test: dict,
    y_test,
    y_test_pred,
) -> None:
    """Persist model, metadata, metrics, and test predictions to disk."""

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Save the trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 2. Save model metadata
    metadata = {
        "features": FEATURES,
        "model_type": "LinearRegression",
        "training_date": datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {METADATA_PATH}")

    # 3. Save metrics (train + test)
    metrics = {
        "train": metrics_train,
        "test": metrics_test,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # 4. Save test predictions
    preds_df = pd.DataFrame({"actual": y_test.values, "predicted": y_test_pred})
    preds_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Test predictions saved to {PREDICTIONS_PATH}")


def print_summary(metrics_train: dict, metrics_test: dict) -> None:
    """Print a formatted summary of training and test metrics."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<10} {'Train':>12} {'Test':>12}")
    print("-" * 34)
    print(f"{'MAE':<10} {metrics_train['mae']:>12,.2f} {metrics_test['mae']:>12,.2f}")
    print(f"{'RMSE':<10} {metrics_train['rmse']:>12,.2f} {metrics_test['rmse']:>12,.2f}")
    print(f"{'R²':<10} {metrics_train['r2']:>12.4f} {metrics_test['r2']:>12.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """End-to-end training pipeline."""

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Chronological train/test split (80/20, no shuffle)
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Build the pipeline
    pipeline = build_pipeline()

    # 4. Train
    print("Training model ...")
    pipeline.fit(X_train, y_train)
    print("  → Training complete.")

    # 5. Evaluate on train and test sets
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics_train = evaluate(y_train, y_train_pred)
    metrics_test = evaluate(y_test, y_test_pred)

    # 6. Save all artifacts
    save_artifacts(pipeline, metrics_train, metrics_test, y_test, y_test_pred)

    # 7. Print summary
    print_summary(metrics_train, metrics_test)


if __name__ == "__main__":
    main()
