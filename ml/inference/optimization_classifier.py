
"""
CloudMorph - Optimization Classifier Inference

Loads the trained Extra Trees classifier and predicts
future workload conditions from the latest 12 observations.

This module does not execute cloud resource changes.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
    / "optimization_classifier.joblib"
)

WINDOW_SIZE = 12

EXPECTED_FEATURE_COUNT = 55

CLASS_LABELS = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]


logger = logging.getLogger("CloudMorphOptimizationInference")


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

_model_package = None


def load_model():

    global _model_package

    if _model_package is not None:
        return _model_package

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Optimization classifier not found: {MODEL_PATH}"
        )

    logger.info(
        "Loading CloudMorph optimization classifier..."
    )

    package = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "feature_columns",
        "class_labels",
        "model_name",
        "model_version",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in package
    ]

    if missing_keys:
        raise ValueError(
            f"Invalid model package. Missing keys: {missing_keys}"
        )

    if len(package["feature_columns"]) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected feature count in trained model."
        )

    if package["class_labels"] != CLASS_LABELS:
        raise ValueError(
            "Trained model class labels do not match inference labels."
        )

    _model_package = package

    logger.info(
        "Loaded %s version %s",
        package["model_name"],
        package["model_version"],
    )

    return _model_package


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def validate_history(history):

    if isinstance(history, pd.DataFrame):
        df = history.copy()

    else:
        df = pd.DataFrame(history)

    required_columns = [
        "util_CPU",
        "util_memory",
        "util_disk",
        "load1",
        "load5",
        "load15",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing workload columns: {missing}"
        )

    if len(df) != WINDOW_SIZE:
        raise ValueError(
            f"Expected exactly {WINDOW_SIZE} observations; "
            f"received {len(df)}."
        )

    df = df[required_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df.isna().any().any():
        raise ValueError(
            "Workload history contains missing or non-numeric values."
        )

    values = df.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Workload history contains infinite values."
        )

    utilization_columns = [
        "util_CPU",
        "util_memory",
        "util_disk",
    ]

    for column in utilization_columns:

        if (
            (df[column] < 0).any()
            or (df[column] > 100).any()
        ):
            raise ValueError(
                f"{column} must be between 0 and 100."
            )

    load_columns = [
        "load1",
        "load5",
        "load15",
    ]

    for column in load_columns:

        if (df[column] < 0).any():
            raise ValueError(
                f"{column} cannot contain negative values."
            )

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# Must match prepare_optimization_dataset.py exactly.
# ---------------------------------------------------------

def generate_features(history):

    df = validate_history(history)

    features = {}

    metrics = [
        "util_CPU",
        "util_memory",
        "util_disk",
        "load1",
        "load5",
        "load15",
    ]

    for metric in metrics:

        values = df[metric].to_numpy(dtype=float)

        features[f"{metric}_current"] = float(values[-1])

        features[f"{metric}_mean"] = float(np.mean(values))

        features[f"{metric}_std"] = float(np.std(values))

        features[f"{metric}_min"] = float(np.min(values))

        features[f"{metric}_max"] = float(np.max(values))

        features[f"{metric}_median"] = float(np.median(values))

        features[f"{metric}_recent_mean"] = float(
            np.mean(values[-3:])
        )

        features[f"{metric}_previous_mean"] = float(
            np.mean(values[-6:-3])
        )

        features[f"{metric}_trend"] = float(
        np.polyfit(np.arange(len(values)), values, 1)[0]
        )

    cpu_values = df["util_CPU"].to_numpy(dtype=float)

    memory_values = df["util_memory"].to_numpy(dtype=float)

    if (
        np.std(cpu_values) == 0
        or np.std(memory_values) == 0
    ):
        correlation = 0.0

    else:
        correlation = float(
            np.corrcoef(cpu_values, memory_values)[0, 1]
        )

    features["cpu_memory_correlation"] = correlation

    return features


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

def predict_workload_condition(history):

    package = load_model()

    model = package["model"]

    feature_columns = package["feature_columns"]

    features = generate_features(history)

    feature_frame = pd.DataFrame(
        [features],
        columns=feature_columns,
    )

    if feature_frame.isna().any().any():
        raise ValueError(
            "Generated inference features contain missing values."
        )

    probabilities = model.predict_proba(feature_frame)[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = model.classes_[predicted_index]

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    return {
        "predicted_condition": str(predicted_class),

        "confidence": float(
            probabilities[predicted_index]
        ),

        "class_probabilities": probability_map,

        "model": package["model_name"],

        "model_version": package["model_version"],

        "prediction_horizon_minutes": 30,

        "input_history_minutes": 60,

        "feature_count": len(feature_columns),

        "analysis_type": "ml_classifier",
    }
