import unittest

import numpy as np

from ml.inference import (
    forecast_utilization,
    FEATURE_NAMES,
    FORECAST_HORIZONS,
)


class TestCloudMorphForecast(unittest.TestCase):

    def setUp(self):

        # Create 12 observations with 6 features.
        self.history = np.array(
            [
                [20, 40, 30, 0.5, 0.6, 0.7]
                for _ in range(12)
            ],
            dtype=np.float32,
        )

        # Change the latest observation.
        self.history[-1] = [
            35,
            55,
            45,
            0.8,
            0.9,
            1.0,
        ]

    # --------------------------------------------------------
    # Test valid prediction
    # --------------------------------------------------------

    def test_valid_forecast(self):

        result = forecast_utilization(
            self.history
        )

        self.assertEqual(
            result["status"],
            "success",
        )

        self.assertEqual(
            result["forecast_method"],
            "Persistence",
        )

    # --------------------------------------------------------
    # Test forecast horizons
    # --------------------------------------------------------

    def test_forecast_horizons(self):

        result = forecast_utilization(
            self.history
        )

        self.assertIn(
            "15min",
            result["forecasts"],
        )

        self.assertIn(
            "30min",
            result["forecasts"],
        )

        self.assertEqual(
            set(result["forecasts"].keys()),
            set(FORECAST_HORIZONS.keys()),
        )

    # --------------------------------------------------------
    # Test CPU prediction
    # --------------------------------------------------------

    def test_cpu_prediction(self):

        result = forecast_utilization(
            self.history
        )

        self.assertEqual(
            result["forecasts"]["15min"]["cpu_utilization"],
            35.0,
        )

        self.assertEqual(
            result["forecasts"]["30min"]["cpu_utilization"],
            35.0,
        )

    # --------------------------------------------------------
    # Test memory prediction
    # --------------------------------------------------------

    def test_memory_prediction(self):

        result = forecast_utilization(
            self.history
        )

        self.assertEqual(
            result["forecasts"]["15min"]["memory_utilization"],
            55.0,
        )

        self.assertEqual(
            result["forecasts"]["30min"]["memory_utilization"],
            55.0,
        )

    # --------------------------------------------------------
    # Test invalid input shape
    # --------------------------------------------------------

    def test_invalid_shape(self):

        invalid_history = np.zeros(
            (10, 6),
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):

            forecast_utilization(
                invalid_history
            )

    # --------------------------------------------------------
    # Test missing values
    # --------------------------------------------------------

    def test_missing_values(self):

        invalid_history = self.history.copy()

        invalid_history[0, 0] = np.nan

        with self.assertRaises(ValueError):

            forecast_utilization(
                invalid_history
            )

    # --------------------------------------------------------
    # Test infinite values
    # --------------------------------------------------------

    def test_infinite_values(self):

        invalid_history = self.history.copy()

        invalid_history[0, 0] = np.inf

        with self.assertRaises(ValueError):

            forecast_utilization(
                invalid_history
            )

    # --------------------------------------------------------
    # Test utilization boundaries
    # --------------------------------------------------------

    def test_invalid_utilization(self):

        invalid_history = self.history.copy()

        invalid_history[-1, 0] = 150

        with self.assertRaises(ValueError):

            forecast_utilization(
                invalid_history
            )

    # --------------------------------------------------------
    # Test negative load
    # --------------------------------------------------------

    def test_negative_load(self):

        invalid_history = self.history.copy()

        invalid_history[-1, 3] = -1

        with self.assertRaises(ValueError):

            forecast_utilization(
                invalid_history
            )


if __name__ == "__main__":

    unittest.main(verbosity=2)