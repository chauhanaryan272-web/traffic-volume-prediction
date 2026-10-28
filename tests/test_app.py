"""
Unit and integration tests for the Flask Web Application.

Tests all web routes, JSON API endpoints, input validation, and alias normalization.
"""

import unittest
import json
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app


class TestAppRoutes(unittest.TestCase):
    """Test web pages and HTTP endpoints."""

    def setUp(self):
        """Set up Flask test client in testing mode."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page(self):
        """GET / should render the landing page with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Traffic Volume Prediction System", response.data)

    def test_predict_page(self):
        """GET /predict should render the prediction form with 200 OK."""
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predict Traffic Volume", response.data)

    def test_dashboard_page(self):
        """GET /dashboard should render the dashboard with 200 OK."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model Dashboard", response.data)

    def test_about_page(self):
        """GET /about should render the about page with 200 OK."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About This Project", response.data)

    def test_api_metrics(self):
        """GET /api/metrics should return 200 OK and model metrics JSON."""
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("train", data)
        self.assertIn("test", data)
        self.assertIn("r2", data["test"])

    def test_api_analysis(self):
        """GET /api/analysis should return 200 OK and dataset statistics."""
        response = self.client.get("/api/analysis")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("hourly", data)
        self.assertIn("daily", data)
        self.assertIn("monthly", data)
        self.assertIn("distribution", data)

    def test_api_predict_canonical_keys(self):
        """POST /api/predict with canonical keys should return 200 and a positive prediction."""
        payload = {
            "temperature_celsius": 20.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "hour": 8,
            "day_of_week": 1,
            "month": 6,
            "is_weekend": 0,
            "is_holiday": 0,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("prediction", data)
        self.assertGreaterEqual(data["prediction"], 0.0)

    def test_api_predict_aliased_keys(self):
        """POST /api/predict with shorthand aliases (temp, rain, snow, clouds) should succeed."""
        payload = {
            "temp": 15.5,
            "rain": 2.0,
            "snow": 0.0,
            "clouds": 80,
            "hour": 17,
            "day_of_week": 4,
            "month": 10,
            "is_weekend": 0,
            "is_holiday": 0,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(data["prediction"], 0.0)

    def test_api_predict_missing_fields(self):
        """POST /api/predict with missing required fields should return 400 Bad Request."""
        payload = {
            "temp": 20.0,
            "hour": 8,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")
        self.assertIn("errors", data)

    def test_api_predict_out_of_range(self):
        """POST /api/predict with out-of-range values should return 400 Bad Request."""
        payload = {
            "temperature_celsius": 120.0,  # Exceeds max range (60°C)
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "hour": 8,
            "day_of_week": 1,
            "month": 6,
            "is_weekend": 0,
            "is_holiday": 0,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")


if __name__ == "__main__":
    unittest.main()
