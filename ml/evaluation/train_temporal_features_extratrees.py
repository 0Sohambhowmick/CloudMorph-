"""
CloudMorph: 63-feature temporal Extra Trees experiment.

This is a separate experiment. It imports the established split/model/metric
helpers from train_extratrees_selected_features.py, but reads the already-built
63-feature dataset and writes all outputs to a new directory.

It does NOT modify the production model or the existing 57-feature experiment.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Project paths assume this file is saved in ml/evaluation/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "temporal_features_experiment"
    / "temporal_features_dataset.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "ml" / "models" / "temporal_features_extratrees"

MODEL_PATH = OUTPUT_DIR / "temporal_features_extratrees.joblib"
REPORT_PATH = OUTPUT_DIR / "temporal_features_report.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"
THRESHOLD_PATH = OUTPUT_DIR / "validation_threshold_results.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
CONFUSION_PATH = OUTPUT_DIR / "risk_confusion_matrix.csv"

METADATA_COLUMNS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
    "target",
]

EXPECTED_FEATURE_COUNT = 63
EXPERIMENTAL_FEATURES = [
    "memory_recent_max",
    "cpu_recent_max",
    "memory_recent_slope",
    "memory_previous_slope",
    "memory_acceleration",
    "memory_recent_long_gap",
    "memory_recent_range",
    "memory_growth_per_5min",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("cloudmorph_temporal_experiment")


def load_baseline_helpers():
    """
    Reuse the exact split, model, threshold, and metric implementations from
    the existing experiment instead of reimplementing them differently.
    """
    evaluation_path = str(EVALUATION_DIR)
    if evaluation_path not in sys.path:
        sys.path.insert(0, evaluation_path)

    import train_extratrees_selected_features as baseline

    required = [
        "create_temporal_splits",
        "create_model",
        "get_high_probabilities",
        "calculate_risk_metrics",
        "select_threshold",
        "classification_metrics",
        "CLASS_LABELS",
        "MIN_VALIDATION_PRECISION",
    ]
    missing = [name for name in required if not hasattr(baseline, name)]
    if missing:
        raise AttributeError(
            "The existing experiment script is missing required helpers: "
            + ", ".join(missing)
        )
    return baseline


def main():
    logger.info("=" * 72)
    logger.info("CLOUDMORPH 63-FEATURE TEMPORAL EXTRA TREES EXPERIMENT")
    logger.info("=" * 72)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Enhanced dataset not found: {DATASET_PATH}\n"
            "First run ml/evaluation/prepare_temporal_features_experiment.py."
        )

    baseline = load_baseline_helpers()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    missing_metadata = set(METADATA_COLUMNS) - set(df.columns)
    if missing_metadata:
        raise ValueError(f"Dataset is missing metadata columns: {missing_metadata}")

    feature_columns = [
        column for column in df.columns if column not in METADATA_COLUMNS
    ]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, found "
            f"{len(feature_columns)}. Check the enhanced dataset schema."
        )

    missing_experimental = set(EXPERIMENTAL_FEATURES) - set(feature_columns)
    if missing_experimental:
        raise ValueError(
            f"Enhanced dataset is missing experimental features: "
            f"{missing_experimental}"
        )

    if df["target"].isna().any():
        raise ValueError("Target column contains missing values.")

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains missing or non-finite values.")

    y = df["target"].astype(str)
    unexpected_labels = set(y.unique()) - set(baseline.CLASS_LABELS)
    if unexpected_labels:
        raise ValueError(f"Unexpected target labels: {unexpected_labels}")

    logger.info("Loaded enhanced dataset: %s", DATASET_PATH)
    logger.info("Samples: %s | Features: %s", len(df), len(feature_columns))
    logger.info("Experimental temporal features: %s", EXPERIMENTAL_FEATURES)
    logger.info("Class distribution: %s", y.value_counts().to_dict())

    (
        train_df,
        validation_df,
        test_df,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        validation_start,
        test_start,
        purged_count,
    ) = baseline.create_temporal_splits(df, X, y)

    # Same Extra Trees configuration as the existing 57-feature experiment.
    model = baseline.create_model()
    logger.info(
        "Training Extra Trees with %s features | train rows: %s",
        len(feature_columns),
        len(X_train),
    )
    model.fit(X_train, y_train)

    # Threshold is selected using validation data only.
    validation_probabilities = baseline.get_high_probabilities(
        model, X_validation
    )
    threshold_results, selected_threshold = baseline.select_threshold(
        y_validation, validation_probabilities
    )

    # The test set is evaluated only after model fitting and threshold selection.
    workload_predictions = model.predict(X_test)
    high_probabilities = baseline.get_high_probabilities(model, X_test)
    risk_alerts = high_probabilities >= selected_threshold

    risk_metrics = baseline.calculate_risk_metrics(y_test, risk_alerts)
    classification_results = baseline.classification_metrics(
        y_test, workload_predictions
    )

    split_info = {
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(validation_df)),
        "test_samples": int(len(test_df)),
        "purged_samples": int(purged_count),
        "validation_start_history_step": float(validation_start),
        "test_start_history_step": float(test_start),
        "counts": [
            int(len(train_df)),
            int(len(validation_df)),
            int(len(test_df)),
        ],
    }

    # Save validation threshold sweep.
    threshold_results.to_csv(THRESHOLD_PATH, index=False)

    # Save test predictions and risk alerts.
    prediction_columns = [
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
    ]
    prediction_df = test_df[prediction_columns].copy()
    prediction_df["actual_workload"] = y_test.to_numpy()
    prediction_df["predicted_workload"] = workload_predictions
    prediction_df["high_demand_probability"] = high_probabilities
    prediction_df["risk_threshold"] = selected_threshold
    prediction_df["risk_alert"] = np.where(
        risk_alerts, "HIGH_RISK", "NO_HIGH_RISK"
    )
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    matrix = confusion_matrix(
        y_test.to_numpy() == "HIGH_DEMAND",
        risk_alerts,
        labels=[False, True],
    )
    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual_No_High_Demand", "Actual_High_Demand"],
        columns=["Predicted_No_Risk", "Predicted_High_Risk"],
    )
    confusion_df.index.name = "Actual"
    confusion_df.to_csv(CONFUSION_PATH)

    pd.DataFrame(
        [
            {
                "evaluation": "Workload classification",
                **classification_results,
            },
            {
                "evaluation": "Independent high-demand risk",
                **risk_metrics,
            },
        ]
    ).to_csv(METRICS_PATH, index=False)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "class_labels": baseline.CLASS_LABELS,
        "model_name": "63-Feature Temporal Extra Trees",
        "model_version": "temporal-experimental-1.0.0",
        "risk_threshold": selected_threshold,
        "metadata": {
            "dataset": str(DATASET_PATH),
            "feature_count": len(feature_columns),
            "experimental_features": EXPERIMENTAL_FEATURES,
            "historical_window_minutes": 60,
            "prediction_horizon_minutes": 30,
            "training_method": "chronological_40_30_30_with_purging",
            "original_model_helpers_reused": True,
            "original_production_model_modified": False,
            "deployed_to_api": False,
        },
    }
    joblib.dump(model_package, MODEL_PATH)

    report = {
        "project": "CloudMorph",
        "experiment": "63-Feature Temporal Extra Trees",
        "status": "experimental",
        "model_version": "temporal-experimental-1.0.0",
        "dataset": {
            "path": str(DATASET_PATH),
            "samples": int(len(df)),
            "feature_count": len(feature_columns),
            "experimental_features": EXPERIMENTAL_FEATURES,
            "class_distribution": {
                str(label): int(count)
                for label, count in y.value_counts().items()
            },
        },
        "model_configuration": {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "split": split_info,
        "threshold_selection": {
            "method": "Maximum validation F1 subject to minimum precision",
            "minimum_validation_precision": float(
                baseline.MIN_VALIDATION_PRECISION
            ),
            "selected_threshold": float(selected_threshold),
            "test_data_used_for_threshold_selection": False,
        },
        "test_classification_metrics": classification_results,
        "test_independent_risk_metrics": risk_metrics,
        "test_classification_report": classification_report(
            y_test,
            workload_predictions,
            labels=baseline.CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "important_notes": [
            "The 63-feature enhanced dataset was loaded directly.",
            "All eight added features are derived from historical observations.",
            "Future observations are used exclusively for workload labels.",
            "The same chronological split and purging helper as the 57-feature experiment was reused.",
            "The same Extra Trees configuration was reused.",
            "The risk threshold was selected using validation data only.",
            "The test period was not used for model or threshold selection.",
            "The same machines may occur in all chronological periods; this is not an unseen-machine evaluation.",
            "The original production model and existing 57-feature experiment were not modified.",
            "This model is experimental and is not deployed to the API.",
            "HIGH_RISK is a model-based warning, not a confirmed incident.",
            "No cloud resources are modified.",
            "The validation high-demand sample count is small; threshold stability is limited.",
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("=" * 72)
    logger.info("63-FEATURE TEMPORAL TEST RESULTS")
    logger.info("=" * 72)
    logger.info("Selected validation threshold: %.2f", selected_threshold)
    logger.info("Classification metrics: %s", classification_results)
    logger.info("Independent risk metrics: %s", risk_metrics)
    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_test,
            workload_predictions,
            labels=baseline.CLASS_LABELS,
            zero_division=0,
        ),
    )
    logger.info("Saved model: %s", MODEL_PATH)
    logger.info("Saved report: %s", REPORT_PATH)
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)
    logger.info("Saved metrics: %s", METRICS_PATH)
    logger.info("Saved threshold results: %s", THRESHOLD_PATH)
    logger.info("Saved feature importance: %s", FEATURE_IMPORTANCE_PATH)
    logger.info("Saved confusion matrix: %s", CONFUSION_PATH)
    logger.info("63-feature temporal experiment completed.")


if __name__ == "__main__":
    main()
