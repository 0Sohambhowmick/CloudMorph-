
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "sequences"
    / "server_multihorizon_sequences.npz"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

OUTPUT_PATH = MODEL_DIR / "forecast_benchmark_v1.json"

LSTM_METRICS_PATH = MODEL_DIR / "training_metrics_v3.json"

HORIZONS = {
    "15min": 3,
    "30min": 6,
}

TARGET_NAMES = [
    "CPU utilization",
    "Memory utilization",
]

RANDOM_STATE = 42

RF_ESTIMATORS = 100
RF_MAX_DEPTH = 14
RF_MIN_SAMPLES_LEAF = 5


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH) as data:

        X_train = data["X_train"].astype(np.float32)
        X_val = data["X_val"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)

        targets = {}

        for horizon in HORIZONS:

            targets[f"y_train_{horizon}"] = (
                data[f"y_train_{horizon.replace('min', '')}"]
                .astype(np.float32)
            )

            targets[f"y_val_{horizon}"] = (
                data[f"y_val_{horizon.replace('min', '')}"]
                .astype(np.float32)
            )

            targets[f"y_test_{horizon}"] = (
                data[f"y_test_{horizon.replace('min', '')}"]
                .astype(np.float32)
            )

    if X_train.shape[1:] != (12, 6):
        raise ValueError(
            f"Expected (*, 12, 6), got {X_train.shape}"
        )

    for horizon in HORIZONS:

        for split, X in [
            ("train", X_train),
            ("val", X_val),
            ("test", X_test),
        ]:

            y = targets[f"y_{split}_{horizon}"]

            if len(X) != len(y):
                raise ValueError(
                    f"Sample mismatch: {split}, {horizon}"
                )

            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError(
                    f"Invalid target shape: {split}, {horizon}"
                )

    return X_train, X_val, X_test, targets


# ============================================================
# FEATURE PREPARATION
# ============================================================

def flatten_sequences(X):

    return X.reshape(
        X.shape[0],
        -1
    )


# ============================================================
# BASELINE FORECASTING METHODS
# ============================================================

def persistence_forecast(X, horizon):

    # Predict the latest observed CPU and memory values.
    return X[:, -1, :2].copy()


def moving_average_forecast(X, horizon):

    # Average the latest six observations.
    recent_values = X[:, -6:, :2]

    return recent_values.mean(axis=1)


def linear_trend_forecast(X, horizon):

    # Use the latest six observations to estimate a trend.
    recent_values = X[:, -6:, :2]

    time = np.arange(6, dtype=np.float32)

    time_mean = time.mean()

    centered_time = time - time_mean

    denominator = np.sum(centered_time ** 2)

    value_mean = recent_values.mean(axis=1)

    slope = np.sum(
        centered_time[None, :, None]
        * (recent_values - value_mean[:, None, :]),
        axis=1,
    ) / denominator

    # Forecast 3 steps ahead for 15 minutes
    # and 6 steps ahead for 30 minutes.
    forecast_steps = HORIZONS[horizon]

    predicted = (
        value_mean
        + slope * (forecast_steps + 5 - time_mean)
    )

    return predicted


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
    }


def evaluate_predictions(actual, predicted):

    results = {}

    clipped = np.clip(
        predicted,
        0,
        100,
    )

    for index, target_name in enumerate(TARGET_NAMES):

        results[target_name] = {

            "MAE": calculate_metrics(
                actual[:, index],
                predicted[:, index],
            )["MAE"],

            "RMSE": calculate_metrics(
                actual[:, index],
                predicted[:, index],
            )["RMSE"],

            "Clipped MAE": calculate_metrics(
                actual[:, index],
                clipped[:, index],
            )["MAE"],

            "Clipped RMSE": calculate_metrics(
                actual[:, index],
                clipped[:, index],
            )["RMSE"],
        }

    return results


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_all_methods(X, targets, split, horizon):

    actual = targets[f"y_{split}_{horizon}"]

    results = {}

    methods = {
        "Persistence": persistence_forecast,
        "Moving Average": moving_average_forecast,
        "Linear Trend": linear_trend_forecast,
    }

    for method_name, method in methods.items():

        prediction = method(
            X,
            horizon,
        )

        results[method_name] = evaluate_predictions(
            actual,
            prediction,
        )

    return results


# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_linear_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    horizon,
):

    print(f"\nTraining Linear Regression - {horizon}")

    model = LinearRegression()

    model.fit(
        flatten_sequences(X_train),
        y_train,
    )

    predictions = {
        "validation": model.predict(
            flatten_sequences(X_val)
        ),

        "test": model.predict(
            flatten_sequences(X_test)
        ),
    }

    results = {
        "validation": evaluate_predictions(
            y_val,
            predictions["validation"],
        ),

        "test": evaluate_predictions(
            y_test,
            predictions["test"],
        ),
    }

    model_path = (
        MODEL_DIR
        / f"server_linear_regression_{horizon}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print("Saved:", model_path)

    return results


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    horizon,
):

    print(f"\nTraining Random Forest - {horizon}")

    model = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    model.fit(
        flatten_sequences(X_train),
        y_train,
    )

    predictions = {
        "validation": model.predict(
            flatten_sequences(X_val)
        ),

        "test": model.predict(
            flatten_sequences(X_test)
        ),
    }

    results = {
        "validation": evaluate_predictions(
            y_val,
            predictions["validation"],
        ),

        "test": evaluate_predictions(
            y_test,
            predictions["test"],
        ),
    }

    model_path = (
        MODEL_DIR
        / f"server_random_forest_{horizon}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print("Saved:", model_path)

    return results


# ============================================================
# LOAD LSTM V3 RESULTS
# ============================================================

def load_lstm_results():

    if not LSTM_METRICS_PATH.exists():
        return None

    with open(
        LSTM_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return data.get("test_metrics")


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n" + "=" * 75)
    print("CLOUDMORPH FORECASTING BENCHMARK RESULTS")
    print("=" * 75)

    for horizon, horizon_results in results.items():

        print(f"\n{'=' * 65}")
        print(f"{horizon.upper()} FORECAST")
        print(f"{'=' * 65}")

        for split in ["validation", "test"]:

            print(f"\n{split.upper()} RESULTS")

            for model_name, model_results in (
                horizon_results[split].items()
            ):

                print(f"\n{model_name}")

                for target_name, scores in model_results.items():

                    # Handle ordinary model metrics.
                    if "MAE" in scores and "RMSE" in scores:

                        print(
                            f"  {target_name}: "
                            f"MAE={scores['MAE']:.4f}, "
                            f"RMSE={scores['RMSE']:.4f}"
                        )

                    # Handle nested LSTM V3 metrics.
                    else:

                        print(f"  {target_name}")

                        for method_name, method_scores in scores.items():

                            print(
                                f"    {method_name}: "
                                f"MAE={method_scores['MAE']:.4f}, "
                                f"RMSE={method_scores['RMSE']:.4f}"
                            )


# ============================================================
# MAIN
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("CLOUDMORPH - FORECASTING BENCHMARK V1")
    print("=" * 75)

    print("\nLoading dataset...")

    X_train, X_val, X_test, targets = load_data()

    print("\nDataset loaded successfully.")

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    all_results = {}

    for horizon in HORIZONS:

        print("\n" + "#" * 65)
        print(f"PROCESSING {horizon.upper()} FORECAST")
        print("#" * 65)

        y_train = targets[f"y_train_{horizon}"]
        y_val = targets[f"y_val_{horizon}"]
        y_test = targets[f"y_test_{horizon}"]

        horizon_results = {
            "validation": {},
            "test": {},
        }

        # --------------------------------------------
        # Baseline methods
        # --------------------------------------------

        baseline_val = evaluate_all_methods(
            X_val,
            targets,
            "val",
            horizon,
        )

        baseline_test = evaluate_all_methods(
            X_test,
            targets,
            "test",
            horizon,
        )

        for method_name in baseline_val:

            horizon_results["validation"][method_name] = (
                baseline_val[method_name]
            )

            horizon_results["test"][method_name] = (
                baseline_test[method_name]
            )

        # --------------------------------------------
        # Linear Regression
        # --------------------------------------------

        linear_results = train_linear_regression(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            horizon,
        )

        horizon_results["validation"][
            "Linear Regression"
        ] = linear_results["validation"]

        horizon_results["test"][
            "Linear Regression"
        ] = linear_results["test"]

        # --------------------------------------------
        # Random Forest
        # --------------------------------------------

        rf_results = train_random_forest(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            horizon,
        )

        horizon_results["validation"][
            "Random Forest"
        ] = rf_results["validation"]

        horizon_results["test"][
            "Random Forest"
        ] = rf_results["test"]

        all_results[horizon] = horizon_results

    # --------------------------------------------
    # Include existing LSTM V3 test results
    # --------------------------------------------

    lstm_results = load_lstm_results()

    if lstm_results:

        for horizon in HORIZONS:

            if horizon in lstm_results:

                all_results[horizon]["test"]["LSTM V3"] = (
                    lstm_results[horizon]
                )

    # --------------------------------------------
    # Print benchmark results
    # --------------------------------------------

    print_results(all_results)

    # --------------------------------------------
    # Save benchmark results
    # --------------------------------------------

    output = {
        "experiment": "CloudMorph Forecasting Benchmark V1",

        "dataset": "Alibaba Cluster Trace 2017",

        "input_window": 12,

        "forecast_horizons_minutes": [15, 30],

        "training_samples": len(X_train),

        "validation_samples": len(X_val),

        "test_samples": len(X_test),

        "random_forest_config": {
            "n_estimators": RF_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "max_features": 0.8,
            "random_state": RANDOM_STATE,
        },

        "results": all_results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print("\nResults saved:")
    print(OUTPUT_PATH)

    print("\nCloudMorph Forecasting Benchmark V1 completed.")


if __name__ == "__main__":
    main()
