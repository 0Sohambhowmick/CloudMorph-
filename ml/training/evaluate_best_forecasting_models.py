
import json
from pathlib import Path

import joblib
import numpy as np

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

OUTPUT_PATH = (
    MODEL_DIR / "best_forecasting_models_evaluation.json"
)

HORIZONS = ["15min", "30min"]

TARGET_NAMES = [
    "CPU utilization",
    "Memory utilization",
]

RANDOM_STATE = 42


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

        X_val = data["X_val"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)

        targets = {}

        for horizon in HORIZONS:

            suffix = horizon.replace("min", "")

            targets[f"y_val_{horizon}"] = (
                data[f"y_val_{suffix}"].astype(np.float32)
            )

            targets[f"y_test_{horizon}"] = (
                data[f"y_test_{suffix}"].astype(np.float32)
            )

    if X_val.shape[1:] != (12, 6):
        raise ValueError(
            f"Expected validation shape (*, 12, 6), got {X_val.shape}"
        )

    if X_test.shape[1:] != (12, 6):
        raise ValueError(
            f"Expected test shape (*, 12, 6), got {X_test.shape}"
        )

    for horizon in HORIZONS:

        if len(X_val) != len(targets[f"y_val_{horizon}"]):
            raise ValueError(
                f"Validation sample mismatch for {horizon}"
            )

        if len(X_test) != len(targets[f"y_test_{horizon}"]):
            raise ValueError(
                f"Test sample mismatch for {horizon}"
            )

    return X_val, X_test, targets


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_features(X):

    time_steps = X.shape[1]

    time = np.arange(
        time_steps,
        dtype=np.float32,
    )

    centered_time = time - time.mean()

    denominator = np.sum(centered_time ** 2)

    mean_values = X.mean(axis=1)

    slope = np.sum(
        centered_time[None, :, None]
        * (X - mean_values[:, None, :]),
        axis=1,
    ) / denominator

    features = [

        X[:, -1, :],

        mean_values,

        X.std(axis=1),

        X.min(axis=1),

        X.max(axis=1),

        np.median(X, axis=1),

        X[:, 0, :],

        X[:, -1, :] - X[:, 0, :],

        X[:, -3:, :].mean(axis=1),

        X[:, -6:, :].mean(axis=1),

        X[:, -3:, :].std(axis=1),

        X[:, -6:, :].std(axis=1),

        slope,
    ]

    return np.concatenate(
        features,
        axis=1,
    ).astype(np.float32)


# ============================================================
# BASELINE METHODS
# ============================================================

def persistence_forecast(X):

    return X[:, -1, :2].copy()


def moving_average_forecast(X):

    return X[:, -6:, :2].mean(axis=1)


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


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_model(model_name, horizon):

    model_files = {

        "Linear Regression": (
            f"server_linear_regression_{horizon}.joblib"
        ),

        "Random Forest": (
            f"server_random_forest_{horizon}.joblib"
        ),

        "Extra Trees": (
            f"server_extra_trees_{horizon}.joblib"
        ),

        "Histogram Gradient Boosting": (
            f"server_hist_gradient_{horizon}.joblib"
        ),
    }

    if model_name not in model_files:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_path = MODEL_DIR / model_files[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(X, horizon):

    predictions = {}

    # Baseline predictions.
    predictions["Persistence"] = (
        persistence_forecast(X)
    )

    predictions["Moving Average"] = (
        moving_average_forecast(X)
    )

    # V1 models use flattened sequence features.
    flattened = X.reshape(
        X.shape[0],
        -1,
    )

    # V2 models use engineered historical features.
    engineered = extract_features(X)

    model_inputs = {

        "Linear Regression": flattened,

        "Random Forest": flattened,

        "Extra Trees": engineered,

        "Histogram Gradient Boosting": engineered,
    }

    for model_name, model_input in model_inputs.items():

        print(f"Loading {model_name} ({horizon})...")

        model = load_model(
            model_name,
            horizon,
        )

        predictions[model_name] = model.predict(
            model_input
        )

    return predictions


# ============================================================
# SELECT BEST MODEL USING VALIDATION MAE
# ============================================================

def select_best_models(
    validation_actual,
    validation_predictions,
):

    selected = {}

    validation_results = {}

    for index, target_name in enumerate(TARGET_NAMES):

        validation_results[target_name] = {}

        best_model = None
        best_mae = float("inf")

        for model_name, prediction in validation_predictions.items():

            metrics = calculate_metrics(
                validation_actual[:, index],
                prediction[:, index],
            )

            validation_results[target_name][model_name] = metrics

            if metrics["MAE"] < best_mae:

                best_mae = metrics["MAE"]

                best_model = model_name

        selected[target_name] = {
            "model": best_model,
            "validation_MAE": best_mae,
        }

    return selected, validation_results


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_selected_models(
    test_actual,
    test_predictions,
    selected,
):

    results = {}

    for index, target_name in enumerate(TARGET_NAMES):

        selected_model = selected[target_name]["model"]

        prediction = test_predictions[selected_model][
            :, index
        ]

        actual = test_actual[:, index]

        results[target_name] = {

            "selected_model": selected_model,

            "validation_MAE": selected[target_name][
                "validation_MAE"
            ],

            "test_metrics": calculate_metrics(
                actual,
                prediction,
            ),
        }

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    horizon,
    validation_results,
    selected,
    final_results,
):

    print("\n" + "=" * 75)

    print(f"{horizon.upper()} FORECAST MODEL COMPARISON")

    print("=" * 75)

    for target_name in TARGET_NAMES:

        print(f"\n{target_name}")

        print("\nVALIDATION RESULTS")

        sorted_models = sorted(
            validation_results[target_name].items(),
            key=lambda item: item[1]["MAE"],
        )

        for model_name, metrics in sorted_models:

            marker = (
                "  <-- SELECTED"
                if model_name == selected[target_name]["model"]
                else ""
            )

            print(
                f"{model_name}: "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}"
                f"{marker}"
            )

        print("\nFINAL TEST RESULT")

        result = final_results[target_name]

        print("Selected model:", result["selected_model"])

        print(
            f"Test MAE: "
            f"{result['test_metrics']['MAE']:.4f}"
        )

        print(
            f"Test RMSE: "
            f"{result['test_metrics']['RMSE']:.4f}"
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

    print("CLOUDMORPH - BEST FORECASTING MODEL EVALUATION")

    print("=" * 75)

    print("\nLoading validation and test data...")

    X_val, X_test, targets = load_data()

    print("Validation:", X_val.shape)

    print("Testing:", X_test.shape)

    all_results = {}

    # --------------------------------------------------------
    # Process each forecast horizon independently.
    # --------------------------------------------------------

    for horizon in HORIZONS:

        print("\n" + "#" * 65)

        print(f"PROCESSING {horizon.upper()} FORECAST")

        print("#" * 65)

        validation_actual = targets[
            f"y_val_{horizon}"
        ]

        test_actual = targets[
            f"y_test_{horizon}"
        ]

        # Generate validation predictions.
        print("\nGenerating validation predictions...")

        validation_predictions = generate_predictions(
            X_val,
            horizon,
        )

        # Select models using validation MAE only.
        selected, validation_results = select_best_models(
            validation_actual,
            validation_predictions,
        )

        print("\nSelected models using validation data:")

        for target_name, selection in selected.items():

            print(
                f"{target_name}: "
                f"{selection['model']} "
                f"(MAE={selection['validation_MAE']:.4f})"
            )

        # Generate test predictions only after selection.
        print("\nGenerating final test predictions...")

        test_predictions = generate_predictions(
            X_test,
            horizon,
        )

        final_results = evaluate_selected_models(
            test_actual,
            test_predictions,
            selected,
        )

        # Print horizon results.
        print_results(
            horizon,
            validation_results,
            selected,
            final_results,
        )

        all_results[horizon] = {

            "validation_results": validation_results,

            "selected_models": selected,

            "final_test_results": final_results,
        }

    # --------------------------------------------------------
    # Save evaluation report.
    # --------------------------------------------------------

    output = {

        "experiment": (
            "CloudMorph Best Forecasting Models Evaluation"
        ),

        "dataset": "Alibaba Cluster Trace 2017",

        "input_window": 12,

        "forecast_horizons_minutes": [15, 30],

        "validation_samples": len(X_val),

        "test_samples": len(X_test),

        "selection_metric": "Validation MAE",

        "test_usage": (
            "Final evaluation after model selection"
        ),

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

    print("\n" + "=" * 75)

    print("FINAL MODEL EVALUATION COMPLETED")

    print("=" * 75)

    print("\nEvaluation report saved:")

    print(OUTPUT_PATH)

    print("\nCloudMorph model selection completed.")


if __name__ == "__main__":
    main()
