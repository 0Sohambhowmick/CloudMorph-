
"""
CloudMorph - Memory High-Demand Inference

Predicts whether peak memory utilization is likely
to reach or exceed 85% within the next 30 minutes.

Inputs:
    Latest 12 workload observations (60 minutes).

Features:
    Reuses the existing 55-feature generation pipeline.

Model:
    Experimental Extra Trees memory classifier.

This module does not execute cloud resource changes.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.inference.optimization_classifier import (
    generate_features,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "memory_high_demand_classifier"
    / "memory_high_demand_classifier.joblib"
)

TRAINING_REPORT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "memory_high_demand_classifier"
    / "training_report.json"
)

MEMORY_RISK_THRESHOLD = 0.40302382398153497

EXPECTED_FEATURE_COUNT = 55

WINDOW_SIZE = 12

MEMORY_HIGH_DEMAND_LIMIT = 85.0

logger = logging.getLogger("CloudMorphMemoryRiskInference")


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
            f"Memory classifier not found: {MODEL_PATH}"
        )

    logger.info(
        "Loading experimental memory high-demand classifier..."
    )

    package = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "feature_columns",
        "model_name",
        "model_version",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in package
    ]

    if missing_keys:
        raise ValueError(
            f"Invalid memory model package. "
            f"Missing keys: {missing_keys}"
        )

    if len(package["feature_columns"]) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected feature count in memory classifier."
        )

    model = package["model"]

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Memory classifier does not support predict_proba."
        )

    if 1 not in model.classes_:
        raise ValueError(
            "Memory classifier does not contain positive class 1."
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

    if len(df) != WINDOW_SIZE:
        raise ValueError(
            f"Expected exactly {WINDOW_SIZE} observations; "
            f"received {len(df)}."
        )

    # Reuse the existing production feature generator
    # for validation and feature calculation.
    #
    # This validates all six required workload metrics.

    generate_features(df)

    return df


# ---------------------------------------------------------
# MEMORY HIGH-DEMAND PREDICTION
# ---------------------------------------------------------

def predict_memory_high_demand(history):

    package = load_model()

    model = package["model"]

    feature_columns = package["feature_columns"]

    df = validate_history(history)

    features = generate_features(df)

    feature_frame = pd.DataFrame(
        [features],
        columns=feature_columns,
    )

    if feature_frame.isna().any().any():
        raise ValueError(
            "Generated memory inference features contain "
            "missing values."
        )

    feature_values = feature_frame.to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Generated memory inference features contain "
            "infinite values."
        )

    # Find the probability corresponding specifically
    # to positive class 1.

    probabilities = model.predict_proba(feature_frame)[0]

    positive_class_index = list(model.classes_).index(1)

    risk_probability = float(
        probabilities[positive_class_index]
    )

    high_memory_risk = (
        risk_probability >= MEMORY_RISK_THRESHOLD
    )

    return {
        "high_memory_risk": bool(high_memory_risk),

        "risk_probability": risk_probability,

        "risk_probability_percent": round(
            risk_probability * 100,
            2,
        ),

        "threshold": MEMORY_RISK_THRESHOLD,

        "predicted_condition": (
            "HIGH_MEMORY_RISK"
            if high_memory_risk
            else "NO_HIGH_MEMORY_RISK_DETECTED"
        ),

        "target_definition": (
            "Future peak memory utilization >=85% "
            "over the next 30 minutes."
        ),

        "prediction_horizon_minutes": 30,

        "input_history_minutes": 60,

        "feature_count": len(feature_columns),

        "model": package["model_name"],

        "model_version": package["model_version"],

        "analysis_type": "experimental_ml_memory_risk_classifier",

        "automatic_resource_changes": False,
    }
