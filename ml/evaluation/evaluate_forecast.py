
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "sequences"
    / "server_multihorizon_sequences.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "evaluation"
)

OUTPUT_FILE = OUTPUT_DIR / "persistence_forecast_evaluation.json"


# --------------------------------------------------
# FORECAST CONFIGURATION
# --------------------------------------------------

FEATURE_NAMES = ["CPU", "Memory"]

# In our dataset:
# Feature index 0 = CPU utilization
# Feature index 1 = Memory utilization

CPU_FEATURE_INDEX = 0
MEMORY_FEATURE_INDEX = 1


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    errors = actual - predicted

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "samples": int(len(actual)),
    }


# --------------------------------------------------
# PERSISTENCE FORECAST
# --------------------------------------------------

def evaluate_horizon(X_test, y_test, horizon):

    # Persistence forecast:
    # Predict future utilization using the latest
    # observed CPU and memory utilization.

    predictions = X_test[
        :, -1, [CPU_FEATURE_INDEX, MEMORY_FEATURE_INDEX]
    ]

    actual = y_test

    results = {}

    for index, feature_name in enumerate(FEATURE_NAMES):

        results[feature_name.lower()] = calculate_metrics(
            actual[:, index],
            predictions[:, index],
        )

    return {
        "horizon": horizon,
        "forecast_method": "Persistence",
        "metrics": results,
    }


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("\nCloudMorph Forecast Evaluation")
    print("=" * 45)

    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"Sequence dataset not found: {SEQUENCE_FILE}"
        )

    data = np.load(SEQUENCE_FILE)

    required_keys = [
        "X_test",
        "y_test_15",
        "y_test_30",
    ]

    missing_keys = [
        key for key in required_keys if key not in data.files
    ]

    if missing_keys:
        raise KeyError(
            f"Missing arrays in sequence dataset: {missing_keys}"
        )

    X_test = data["X_test"]
    y_test_15 = data["y_test_15"]
    y_test_30 = data["y_test_30"]

    print(f"\nTest input shape: {X_test.shape}")
    print(f"15-minute target shape: {y_test_15.shape}")
    print(f"30-minute target shape: {y_test_30.shape}")

    if X_test.ndim != 3 or X_test.shape[2] < 2:
        raise ValueError("Unexpected test input dimensions.")

    if y_test_15.shape != (len(X_test), 2):
        raise ValueError("Invalid 15-minute target dimensions.")

    if y_test_30.shape != (len(X_test), 2):
        raise ValueError("Invalid 30-minute target dimensions.")

    if not all(
        np.isfinite(array).all()
        for array in [X_test, y_test_15, y_test_30]
    ):
        raise ValueError("Dataset contains non-finite values.")

    results_15 = evaluate_horizon(
        X_test, y_test_15, "15 minutes"
    )

    results_30 = evaluate_horizon(
        X_test, y_test_30, "30 minutes"
    )

    report = {
        "project": "CloudMorph",
        "evaluation_type": "Historical test-set evaluation",
        "forecast_method": "Persistence",
        "model_status": "Baseline",
        "input_window_observations": int(X_test.shape[1]),
        "input_interval_minutes": 5,
        "forecast_horizons": [
            results_15,
            results_30,
        ],
        "notes": [
            "Predictions use the latest observed CPU and memory utilization.",
            "MAE and RMSE are measured in percentage points.",
            "This is a baseline evaluation, not a trained AI model evaluation.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n15-MINUTE FORECAST")
    print("-" * 45)

    for feature, metrics in results_15["metrics"].items():
        print(
            f"{feature.upper()}: "
            f"MAE={metrics['mae']} | "
            f"RMSE={metrics['rmse']}"
        )

    print("\n30-MINUTE FORECAST")
    print("-" * 45)

    for feature, metrics in results_30["metrics"].items():
        print(
            f"{feature.upper()}: "
            f"MAE={metrics['mae']} | "
            f"RMSE={metrics['rmse']}"
        )

    print("\nEvaluation completed successfully.")
    print(f"Report saved at:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
