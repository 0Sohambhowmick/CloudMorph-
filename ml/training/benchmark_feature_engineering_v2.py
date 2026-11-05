
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
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
    MODEL_DIR / "feature_engineering_benchmark_v2.json"
)

HORIZONS = {
    "15min": 3,
    "30min": 6,
}

TARGET_NAMES = [
    "CPU utilization",
    "Memory utilization",
]

RANDOM_STATE = 42

EXTRA_TREES_CONFIG = {
    "n_estimators": 200,
    "max_features": 0.8,
    "min_samples_leaf": 3,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

HIST_GRADIENT_CONFIG = {
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": RANDOM_STATE,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD TRAINING AND VALIDATION DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH) as data:

        X_train = data["X_train"].astype(np.float32)
        X_val = data["X_val"].astype(np.float32)

        targets = {}

        for horizon in HORIZONS:

            suffix = horizon.replace("min", "")

            targets[f"y_train_{horizon}"] = (
                data[f"y_train_{suffix}"].astype(np.float32)
            )

            targets[f"y_val_{horizon}"] = (
                data[f"y_val_{suffix}"].astype(np.float32)
            )

    if X_train.shape[1:] != (12, 6):
        raise ValueError(
            f"Expected input shape (*, 12, 6), got {X_train.shape}"
        )

    for horizon in HORIZONS:

        if len(X_train) != len(targets[f"y_train_{horizon}"]):
            raise ValueError(
                f"Training sample mismatch for {horizon}"
            )

        if len(X_val) != len(targets[f"y_val_{horizon}"]):
            raise ValueError(
                f"Validation sample mismatch for {horizon}"
            )

    return X_train, X_val, targets


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_features(X):

    # X shape: (samples, 12 observations, 6 features)

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

        # Latest observations.
        X[:, -1, :],

        # Overall statistics.
        mean_values,
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        np.median(X, axis=1),

        # Historical starting point.
        X[:, 0, :],

        # Overall change.
        X[:, -1, :] - X[:, 0, :],

        # Recent averages.
        X[:, -3:, :].mean(axis=1),
        X[:, -6:, :].mean(axis=1),

        # Recent volatility.
        X[:, -3:, :].std(axis=1),
        X[:, -6:, :].std(axis=1),

        # Historical trend.
        slope,
    ]

    result = np.concatenate(
        features,
        axis=1,
    )

    return result.astype(np.float32)


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


def evaluate_predictions(actual, predicted):

    results = {}

    for index, target_name in enumerate(TARGET_NAMES):

        results[target_name] = calculate_metrics(
            actual[:, index],
            predicted[:, index],
        )

    return results


# ============================================================
# BASELINE EVALUATION
# ============================================================

def evaluate_baselines(X_val, y_val):

    return {

        "Persistence": evaluate_predictions(
            y_val,
            persistence_forecast(X_val),
        ),

        "Moving Average": evaluate_predictions(
            y_val,
            moving_average_forecast(X_val),
        ),
    }


# ============================================================
# EXTRA TREES
# ============================================================

def train_extra_trees(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon,
):

    print(f"\nTraining Extra Trees - {horizon}")

    model = ExtraTreesRegressor(
        **EXTRA_TREES_CONFIG
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_val)

    results = evaluate_predictions(
        y_val,
        predictions,
    )

    model_path = (
        MODEL_DIR
        / f"server_extra_trees_{horizon}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print("Saved:", model_path)

    return results


# ============================================================
# HISTOGRAM GRADIENT BOOSTING
# ============================================================

def train_hist_gradient(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon,
):

    print(
        f"\nTraining Histogram Gradient Boosting - {horizon}"
    )

    base_model = HistGradientBoostingRegressor(
        **HIST_GRADIENT_CONFIG
    )

    model = MultiOutputRegressor(
        base_model,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_val)

    results = evaluate_predictions(
        y_val,
        predictions,
    )

    model_path = (
        MODEL_DIR
        / f"server_hist_gradient_{horizon}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print("Saved:", model_path)

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n" + "=" * 75)
    print("CLOUDMORPH FEATURE ENGINEERING BENCHMARK V2")
    print("=" * 75)

    for horizon, horizon_results in results.items():

        print(f"\n{horizon.upper()} VALIDATION RESULTS")

        for model_name, model_results in horizon_results.items():

            print(f"\n{model_name}")

            for target_name, scores in model_results.items():

                print(
                    f"  {target_name}: "
                    f"MAE={scores['MAE']:.4f}, "
                    f"RMSE={scores['RMSE']:.4f}"
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
    print("CLOUDMORPH - FEATURE ENGINEERING BENCHMARK V2")
    print("=" * 75)

    print("\nLoading training and validation data...")

    X_train, X_val, targets = load_data()

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)

    print("\nExtracting historical features...")

    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)

    print("Training features:", X_train_features.shape)
    print("Validation features:", X_val_features.shape)

    all_results = {}

    for horizon in HORIZONS:

        print("\n" + "#" * 65)
        print(f"PROCESSING {horizon.upper()}")
        print("#" * 65)

        y_train = targets[f"y_train_{horizon}"]
        y_val = targets[f"y_val_{horizon}"]

        horizon_results = {}

        # --------------------------------------------
        # Baseline methods
        # --------------------------------------------

        horizon_results.update(
            evaluate_baselines(
                X_val,
                y_val,
            )
        )

        # --------------------------------------------
        # Extra Trees
        # --------------------------------------------

        horizon_results["Extra Trees"] = train_extra_trees(
            X_train_features,
            y_train,
            X_val_features,
            y_val,
            horizon,
        )

        # --------------------------------------------
        # Histogram Gradient Boosting
        # --------------------------------------------

        horizon_results[
            "Histogram Gradient Boosting"
        ] = train_hist_gradient(
            X_train_features,
            y_train,
            X_val_features,
            y_val,
            horizon,
        )

        all_results[horizon] = horizon_results

    # --------------------------------------------
    # Print results
    # --------------------------------------------

    print_results(all_results)

    # --------------------------------------------
    # Save results
    # --------------------------------------------

    output = {

        "experiment": (
            "CloudMorph Feature Engineering Benchmark V2"
        ),

        "dataset": "Alibaba Cluster Trace 2017",

        "input_window": 12,

        "forecast_horizons_minutes": [15, 30],

        "training_samples": len(X_train),

        "validation_samples": len(X_val),

        "engineered_feature_count": X_train_features.shape[1],

        "feature_groups": [
            "latest_values",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "median",
            "first_observation",
            "overall_change",
            "recent_3_observation_mean",
            "recent_6_observation_mean",
            "recent_3_observation_std",
            "recent_6_observation_std",
            "historical_slope",
        ],

        "extra_trees_config": EXTRA_TREES_CONFIG,

        "hist_gradient_config": HIST_GRADIENT_CONFIG,

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
    print("FEATURE ENGINEERING BENCHMARK V2 COMPLETED")
    print("=" * 75)

    print("\nResults saved:")
    print(OUTPUT_PATH)

    print("\nAll model files saved in:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
