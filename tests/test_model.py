"""
Unit tests for the trained traffic volume prediction model.

These tests verify that:
  - The saved model can be loaded and used for inference.
  - Predictions are positive numbers.
  - Model metadata contains the expected keys.
"""

import unittest
import os
import json

import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------------------------
# Path configuration – relative to project root (two levels up from tests/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "traffic_volume_model.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")


class TestTrafficVolumeModel(unittest.TestCase):
    """Tests for the saved Linear Regression traffic-volume model."""

    @classmethod
    def setUpClass(cls):
        """Load model and metadata once for all tests."""
        # Load the trained pipeline
        cls.model = joblib.load(MODEL_PATH)

        # Load model metadata
        with open(METADATA_PATH, "r") as f:
            cls.metadata = json.load(f)

    # -----------------------------------------------------------------------
    # Prediction tests
    # -----------------------------------------------------------------------
    def test_prediction_is_positive(self):
        """A prediction on reasonable sample inputs should be a positive number."""
        # Create a sample input DataFrame with plausible test values
        sample_input = pd.DataFrame(
            [
                {
                    "temp": 280.0,       # ~7 °C (Kelvin)
                    "rain_1h": 0.0,
                    "snow_1h": 0.0,
                    "clouds_all": 40,
                    "hour_sin": np.sin(2 * np.pi * 8 / 24),   # 8 AM
                    "hour_cos": np.cos(2 * np.pi * 8 / 24),
                    "month_sin": np.sin(2 * np.pi * 6 / 12),  # June
                    "month_cos": np.cos(2 * np.pi * 6 / 12),
                    "dow_sin": np.sin(2 * np.pi * 2 / 7),     # Tuesday
                    "dow_cos": np.cos(2 * np.pi * 2 / 7),
                    "is_weekend": 0,
                    "is_holiday": 0,
                }
            ]
        )

        prediction = self.model.predict(sample_input)

        # The prediction should be a single positive number
        self.assertEqual(len(prediction), 1)
        self.assertGreater(prediction[0], 0, "Predicted traffic volume should be positive.")

    # -----------------------------------------------------------------------
    # Metadata tests
    # -----------------------------------------------------------------------
    def test_metadata_has_expected_keys(self):
        """Model metadata JSON must contain essential keys."""
        expected_keys = {"features", "model_type", "training_date", "sklearn_version"}
        actual_keys = set(self.metadata.keys())
        self.assertTrue(
            expected_keys.issubset(actual_keys),
            f"Missing metadata keys: {expected_keys - actual_keys}",
        )

    def test_metadata_features_list(self):
        """The features list in metadata should have 12 entries."""
        self.assertEqual(len(self.metadata["features"]), 12)

    def test_metadata_model_type(self):
        """Model type recorded in metadata should be LinearRegression."""
        self.assertEqual(self.metadata["model_type"], "LinearRegression")


if __name__ == "__main__":
    unittest.main()
