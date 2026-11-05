
"""
CloudMorph - Validation-Based HIGH_DEMAND Threshold Evaluation

Uses chronological training, validation and testing.

Threshold selection uses validation data only.
The final test period is never used to select a threshold.

The production classifier is not modified.
"""

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
    / "optimization_classifier.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
    / "threshold_evaluation"
)

REPORT_PATH = OUTPUT_DIR / "threshold_evaluation_report.json"
METRICS_PATH = OUTPUT_DIR / "threshold_metrics.csv"
THRESHOLD_PATH = OUTPUT_DIR / "validation_threshold_analysis.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "threshold_test_predictions.csv"
CONFUSION_PATH = OUTPUT_DIR / "threshold_confusion_matrix.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 300

TRAIN_FRACTION = 0.40
TEST_FRACTION = 0.70

CLASS_LABELS = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]

TARGET_COLUMN = "target"
GROUP_COLUMN = "machineID"

METADATA_COLUMNS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
    "target",
]

THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
]

MIN_VALIDATION_PRECISION = 0.60


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphThresholdEvaluation")


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_dataset():

    logger.info("Loading optimization dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    missing = [
        col for col in METADATA_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metadata columns: {missing}"
        )

    if df[METADATA_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing metadata values detected."
        )

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    if len(feature_columns) != 55:
        raise ValueError(
            f"Expected 55 features, found {len(feature_columns)}."
        )

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "Missing or non-numeric feature values detected."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite feature values detected."
        )

    y = df[TARGET_COLUMN].astype(str)

    if set(y.unique()) != set(CLASS_LABELS):
        raise ValueError(
            "Unexpected workload classes detected."
        )

    logger.info("Samples: %s", len(df))
    logger.info("Machines: %s", df[GROUP_COLUMN].nunique())
    logger.info("Features: %s", len(feature_columns))

    return df, X, y, feature_columns


# ---------------------------------------------------------
# TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------

def create_temporal_splits(df, X, y):

    logger.info("Creating chronological splits...")

    steps = np.sort(
        df["history_start_step"].unique()
    )

    validation_index = math.ceil(
        len(steps) * TRAIN_FRACTION
    )

    test_index = math.ceil(
        len(steps) * TEST_FRACTION
    )

    validation_start = steps[validation_index]
    test_start = steps[test_index]

    # Training:
    # Earlier samples with targets ending before
    # the validation period starts.

    train_mask = (
        (df["history_start_step"] < validation_start)
        &
        (df["target_end_step"] < validation_start)
    )

    # Validation:
    # Samples beginning in the validation period
    # whose target windows end before the test period.

    validation_mask = (
        (df["history_start_step"] >= validation_start)
        &
        (df["target_end_step"] < test_start)
    )

    # Testing:
    # Samples beginning at or after test boundary.

    test_mask = (
        df["history_start_step"] >= test_start
    )

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    purged_count = len(df) - (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "One or more temporal splits are empty."
        )

    for name, part in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df),
    ]:

        missing = set(CLASS_LABELS) - set(
            part[TARGET_COLUMN].unique()
        )

        if missing:
            raise ValueError(
                f"{name} split is missing classes: {missing}"
            )

    # Verify target windows do not overlap the next period.

    if train_df["target_end_step"].max() >= validation_start:
        raise RuntimeError(
            "Training target overlaps validation period."
        )

    if validation_df["target_end_step"].max() >= test_start:
        raise RuntimeError(
            "Validation target overlaps test period."
        )

    if test_df["history_start_step"].min() != test_start:
        raise RuntimeError(
            "Unexpected test boundary."
        )

    logger.info("=" * 55)
    logger.info("CHRONOLOGICAL SPLIT SUMMARY")
    logger.info("=" * 55)

    logger.info(
        "Training: %s samples",
        len(train_df),
    )

    logger.info(
        "Validation: %s samples",
        len(validation_df),
    )

    logger.info(
        "Testing: %s samples",
        len(test_df),
    )

    logger.info(
        "Purged samples: %s",
        purged_count,
    )

    logger.info(
        "Validation starts at history step: %s",
        validation_start,
    )

    logger.info(
        "Testing starts at history step: %s",
        test_start,
    )

    for name, part in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df),
    ]:

        logger.info(
            "%s classes:\n%s",
            name,
            part[TARGET_COLUMN].value_counts().to_string(),
        )

    return (
        train_df,
        validation_df,
        test_df,
        X.loc[train_mask],
        y.loc[train_mask],
        X.loc[validation_mask],
        y.loc[validation_mask],
        X.loc[test_mask],
        y.loc[test_mask],
        validation_start,
        test_start,
        purged_count,
    )


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def create_model():

    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------
# HIGH-DEMAND PROBABILITIES
# ---------------------------------------------------------

def get_high_demand_probabilities(model, X):

    probabilities = model.predict_proba(X)

    high_index = list(model.classes_).index(
        "HIGH_DEMAND"
    )

    return probabilities[:, high_index]


# ---------------------------------------------------------
# THRESHOLD SELECTION
# ---------------------------------------------------------

def evaluate_thresholds(y_true, high_probabilities):

    actual_high = (
        y_true.to_numpy() == "HIGH_DEMAND"
    )

    results = []

    for threshold in THRESHOLDS:

        predicted_high = (
            high_probabilities >= threshold
        )

        precision = precision_score(
            actual_high,
            predicted_high,
            zero_division=0,
        )

        recall = recall_score(
            actual_high,
            predicted_high,
            zero_division=0,
        )

        f1 = f1_score(
            actual_high,
            predicted_high,
            zero_division=0,
        )

        results.append({
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_high_cases": int(
                predicted_high.sum()
            ),
            "actual_high_cases": int(
                actual_high.sum()
            ),
            "true_positives": int(
                (predicted_high & actual_high).sum()
            ),
            "false_positives": int(
                (predicted_high & ~actual_high).sum()
            ),
            "false_negatives": int(
                (~predicted_high & actual_high).sum()
            ),
        })

    results_df = pd.DataFrame(results)

    eligible = results_df[
        results_df["precision"] >= MIN_VALIDATION_PRECISION
    ]

    if eligible.empty:
        logger.warning(
            "No threshold meets minimum validation precision. "
            "Selecting the best validation F1 without a precision constraint."
        )

        eligible = results_df

    best = eligible.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]

    return results_df, float(best["threshold"])


# ---------------------------------------------------------
# MULTICLASS PREDICTION WITH HIGH-DEMAND OVERRIDE
# ---------------------------------------------------------

def predict_with_threshold(
    model,
    X,
    high_probabilities,
    threshold,
):

    predictions = model.predict(X).astype(object)

    high_mask = (
        high_probabilities >= threshold
    )

    predictions[high_mask] = "HIGH_DEMAND"

    return predictions


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):

    actual_high = (
        y_true.to_numpy() == "HIGH_DEMAND"
    )

    predicted_high = (
        predictions == "HIGH_DEMAND"
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),

        "high_demand_precision": float(
            precision_score(
                actual_high,
                predicted_high,
                zero_division=0,
            )
        ),

        "high_demand_recall": float(
            recall_score(
                actual_high,
                predicted_high,
                zero_division=0,
            )
        ),

        "high_demand_f1": float(
            f1_score(
                actual_high,
                predicted_high,
                zero_division=0,
            )
        ),

        "predicted_high_cases": int(
            predicted_high.sum()
        ),

        "actual_high_cases": int(
            actual_high.sum()
        ),
    }


# ---------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------

def save_results(
    train_df,
    validation_df,
    test_df,
    purged_count,
    validation_thresholds,
    selected_threshold,
    default_metrics,
    adjusted_metrics,
    y_test,
    default_predictions,
    adjusted_predictions,
    feature_columns,
    validation_start,
    test_start,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_thresholds.to_csv(
        THRESHOLD_PATH,
        index=False,
    )

    # Default vs adjusted metrics.

    metrics_df = pd.DataFrame([
        {
            "model": "Default Extra Trees",
            "threshold": 0.50,
            **default_metrics,
        },
        {
            "model": "Validation Threshold Extra Trees",
            "threshold": selected_threshold,
            **adjusted_metrics,
        },
    ])

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    # Confusion matrix for adjusted predictions.

    matrix = confusion_matrix(
        y_test,
        adjusted_predictions,
        labels=CLASS_LABELS,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )

    confusion_df.index.name = "Actual"

    confusion_df.to_csv(
        CONFUSION_PATH
    )

    # Test predictions.

    prediction_df = test_df[
        [
            "machineID",
            "history_start_step",
            "target_start_step",
            "target_end_step",
        ]
    ].copy()

    prediction_df["actual"] = y_test.to_numpy()
    prediction_df["default_prediction"] = default_predictions
    prediction_df["adjusted_prediction"] = adjusted_predictions

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    report = {
        "project": "CloudMorph",
        "evaluation": "Chronological validation-based threshold selection",
        "model": "Extra Trees",
        "model_version": "1.0.0",

        "split": {
            "training_samples": int(len(train_df)),
            "validation_samples": int(len(validation_df)),
            "test_samples": int(len(test_df)),
            "purged_samples": int(purged_count),
            "validation_start_history_step": float(validation_start),
            "test_start_history_step": float(test_start),
        },

        "threshold_selection": {
            "method": "Maximum validation HIGH_DEMAND F1",
            "minimum_validation_precision": MIN_VALIDATION_PRECISION,
            "selected_threshold": selected_threshold,
            "validation_results": validation_thresholds.to_dict(
                orient="records"
            ),
        },

        "default_test_metrics": default_metrics,
        "adjusted_test_metrics": adjusted_metrics,

        "adjusted_classification_report": classification_report(
            y_test,
            adjusted_predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),

        "important_notes": [
            "Threshold was selected using validation data only.",
            "The final test set was not used for threshold selection.",
            "The temporal model was trained only on the earliest period.",
            "Overlapping target windows were purged at both boundaries.",
            "The same machines appear in all chronological periods.",
            "The production classifier was not modified.",
            "Threshold adjustment changes the HIGH_DEMAND decision rule.",
            "This experiment does not establish long-term production performance.",
            "Workload labels represent derived future utilization conditions.",
            "No cloud resources are modified.",
        ],
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info("Saved report: %s", REPORT_PATH)
    logger.info("Saved metrics: %s", METRICS_PATH)
    logger.info("Saved threshold analysis: %s", THRESHOLD_PATH)
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)
    logger.info("Saved confusion matrix: %s", CONFUSION_PATH)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH THRESHOLD EVALUATION")
    logger.info("=" * 65)

    (
        df,
        X,
        y,
        feature_columns,
    ) = load_dataset()

    # Verify the existing production model's feature schema.

    if MODEL_PATH.exists():

        package = joblib.load(MODEL_PATH)

        if package.get("feature_columns") != feature_columns:
            raise ValueError(
                "Feature schema does not match production model."
            )

        logger.info(
            "Production feature schema verified."
        )

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
    ) = create_temporal_splits(
        df,
        X,
        y,
    )

    # Train once on the earliest period.

    model = create_model()

    logger.info("Training temporal model...")

    model.fit(
        X_train,
        y_train,
    )

    # Select threshold using validation only.

    logger.info(
        "Evaluating HIGH_DEMAND thresholds on validation data..."
    )

    validation_probabilities = (
        get_high_demand_probabilities(
            model,
            X_validation,
        )
    )

    threshold_results, selected_threshold = (
        evaluate_thresholds(
            y_validation,
            validation_probabilities,
        )
    )

    logger.info(
        "Selected validation threshold: %.4f",
        selected_threshold,
    )

    logger.info(
        "Validation threshold results:\n%s",
        threshold_results.to_string(index=False),
    )

    # Evaluate on the untouched later test period.

    logger.info(
        "Evaluating frozen threshold on final test period..."
    )

    test_probabilities = (
        get_high_demand_probabilities(
            model,
            X_test,
        )
    )

    default_predictions = model.predict(
        X_test
    )

    adjusted_predictions = predict_with_threshold(
        model,
        X_test,
        test_probabilities,
        selected_threshold,
    )

    default_metrics = calculate_metrics(
        y_test,
        default_predictions,
    )

    adjusted_metrics = calculate_metrics(
        y_test,
        adjusted_predictions,
    )

    logger.info("=" * 65)
    logger.info("FINAL TEMPORAL TEST RESULTS")
    logger.info("=" * 65)

    logger.info(
        "Default metrics: %s",
        default_metrics,
    )

    logger.info(
        "Adjusted metrics: %s",
        adjusted_metrics,
    )

    logger.info(
        "Adjusted classification report:\n%s",
        classification_report(
            y_test,
            adjusted_predictions,
            labels=CLASS_LABELS,
            zero_division=0,
        ),
    )

    save_results(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        purged_count=purged_count,
        validation_thresholds=threshold_results,
        selected_threshold=selected_threshold,
        default_metrics=default_metrics,
        adjusted_metrics=adjusted_metrics,
        y_test=y_test,
        default_predictions=default_predictions,
        adjusted_predictions=adjusted_predictions,
        feature_columns=feature_columns,
        validation_start=validation_start,
        test_start=test_start,
    )

    logger.info("=" * 65)
    logger.info("THRESHOLD EVALUATION COMPLETED")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
