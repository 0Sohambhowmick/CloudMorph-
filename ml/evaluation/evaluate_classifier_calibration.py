
"""
CloudMorph - Classifier Probability Calibration Evaluation

Compares:
1. Existing uncalibrated Extra Trees model.
2. Experimental sigmoid-calibrated Extra Trees model.

IMPORTANT:
- Reproduces the original machine-aware test split.
- Verifies test rows against saved predictions.
- Does not use test data to fit the calibrator.
- Does not overwrite the production model.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
)

MODEL_PATH = OUTPUT_DIR / "optimization_classifier.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

REPORT_PATH = OUTPUT_DIR / "calibration_evaluation_report.json"
METRICS_PATH = OUTPUT_DIR / "calibration_metrics.csv"
THRESHOLD_PATH = OUTPUT_DIR / "high_demand_threshold_analysis.csv"

RELIABILITY_PATH = OUTPUT_DIR / "calibration_reliability.png"
THRESHOLD_PLOT_PATH = OUTPUT_DIR / "high_demand_thresholds.png"

RANDOM_STATE = 42
N_ESTIMATORS = 300

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

HIGH_DEMAND_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphCalibration")


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset():

    logger.info("Loading optimization dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    required = METADATA_COLUMNS

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}"
        )

    feature_columns = [
        column for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError("Dataset contains missing features.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite features.")

    y = df[TARGET_COLUMN].astype(str)
    groups = df[GROUP_COLUMN]

    if set(y.unique()) != set(CLASS_LABELS):
        raise ValueError("Unexpected or missing target classes.")

    logger.info("Dataset samples: %s", len(df))
    logger.info("Unique machines: %s", groups.nunique())
    logger.info("Feature count: %s", len(feature_columns))

    return df, X, y, groups, feature_columns


# =========================================================
# REPRODUCE ORIGINAL MACHINE-AWARE SPLIT
# =========================================================

def reproduce_original_test_split(X, y, groups):

    logger.info("Reproducing original test split...")

    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_validation_idx, test_idx = next(
        outer_splitter.split(X, y, groups)
    )

    X_train_validation = X.iloc[train_validation_idx]
    y_train_validation = y.iloc[train_validation_idx]
    groups_train_validation = groups.iloc[train_validation_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    train_machines = set(groups_train_validation)
    test_machines = set(groups_test)

    if train_machines & test_machines:
        raise RuntimeError(
            "Machine leakage detected between training and test."
        )

    logger.info(
        "Training + validation: %s samples / %s machines",
        len(X_train_validation),
        len(train_machines),
    )

    logger.info(
        "Test: %s samples / %s machines",
        len(X_test),
        len(test_machines),
    )

    return (
        X_train_validation,
        y_train_validation,
        groups_train_validation,
        X_test,
        y_test,
        groups_test,
    )


# =========================================================
# VERIFY ORIGINAL TEST ROWS
# =========================================================

def verify_original_test_set(
    groups_test,
    y_test,
    existing_predictions,
):

    logger.info("Verifying saved test predictions...")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    saved = existing_predictions

    required = [
        "machineID",
        "actual",
        "predicted",
    ]

    if any(column not in saved.columns for column in required):
        raise ValueError(
            "Saved test predictions have an unexpected format."
        )

    expected_machines = (
        groups_test.astype(str).reset_index(drop=True)
    )

    saved_machines = (
        saved["machineID"].astype(str).reset_index(drop=True)
    )

    expected_actual = y_test.reset_index(drop=True)
    saved_actual = saved["actual"].astype(str).reset_index(drop=True)

    if len(saved) != len(y_test):
        raise RuntimeError(
            "Test sample count does not match saved predictions. "
            "Calibration evaluation stopped."
        )

    if not expected_machines.equals(saved_machines):
        raise RuntimeError(
            "Test machine IDs/order do not match saved predictions. "
            "Calibration evaluation stopped."
        )

    if not expected_actual.equals(saved_actual):
        raise RuntimeError(
            "Test labels/order do not match saved predictions. "
            "Calibration evaluation stopped."
        )

    logger.info(
        "SUCCESS: Reconstructed test rows match saved predictions."
    )


# =========================================================
# PROBABILITY ALIGNMENT
# =========================================================

def align_probabilities(model, probabilities):

    aligned = np.zeros(
        (len(probabilities), len(CLASS_LABELS)),
        dtype=float,
    )

    model_classes = list(model.classes_)

    for index, label in enumerate(CLASS_LABELS):

        if label not in model_classes:
            raise ValueError(
                f"Model is missing class: {label}"
            )

        source_index = model_classes.index(label)

        aligned[:, index] = probabilities[:, source_index]

    return aligned


# =========================================================
# METRICS
# =========================================================

def calculate_multiclass_brier(y_true, probabilities):

    actual = np.zeros_like(probabilities)

    label_to_index = {
        label: index
        for index, label in enumerate(CLASS_LABELS)
    }

    for row, label in enumerate(y_true):
        actual[row, label_to_index[label]] = 1.0

    return float(
        np.mean(np.sum((probabilities - actual) ** 2, axis=1))
    )


def calculate_top_label_ece(y_true, probabilities, bins=15):

    predicted_indices = np.argmax(probabilities, axis=1)

    confidence = np.max(probabilities, axis=1)

    predictions = np.array(CLASS_LABELS)[predicted_indices]

    correct = (predictions == np.asarray(y_true)).astype(float)

    ece = 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)

    for index in range(bins):

        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if not np.any(mask):
            continue

        average_confidence = float(np.mean(confidence[mask]))
        average_accuracy = float(np.mean(correct[mask]))

        ece += (
            np.mean(mask)
            * abs(average_accuracy - average_confidence)
        )

    return float(ece)


def calculate_classwise_ece(y_true, probabilities, bins=15):

    results = {}

    y_true = np.asarray(y_true)

    edges = np.linspace(0.0, 1.0, bins + 1)

    for class_index, label in enumerate(CLASS_LABELS):

        confidence = probabilities[:, class_index]

        actual = (y_true == label).astype(float)

        ece = 0.0

        for index in range(bins):

            lower = edges[index]
            upper = edges[index + 1]

            if index == bins - 1:
                mask = (
                    (confidence >= lower)
                    & (confidence <= upper)
                )
            else:
                mask = (
                    (confidence >= lower)
                    & (confidence < upper)
                )

            if not np.any(mask):
                continue

            avg_confidence = np.mean(confidence[mask])
            avg_frequency = np.mean(actual[mask])

            ece += (
                np.mean(mask)
                * abs(avg_frequency - avg_confidence)
            )

        results[label] = float(ece)

    return results


def evaluate_probabilities(
    model_name,
    model,
    X_test,
    y_test,
):

    raw_probabilities = model.predict_proba(X_test)

    probabilities = align_probabilities(
        model,
        raw_probabilities,
    )

    predictions = np.array(CLASS_LABELS)[
        np.argmax(probabilities, axis=1)
    ]

    # FIX: sklearn log_loss requires probability columns
    # to follow the same lexicographical order as labels.
    log_loss_labels = sorted(CLASS_LABELS)

    log_loss_column_indices = [
        CLASS_LABELS.index(label)
        for label in log_loss_labels
    ]

    log_loss_probabilities = probabilities[
        :, log_loss_column_indices
    ]

    metrics = {
        "model": model_name,

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),

        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),

        "log_loss": float(
            log_loss(
                y_test,
                log_loss_probabilities,
                labels=log_loss_labels,
            )
        ),

        "multiclass_brier_score": calculate_multiclass_brier(
            y_test,
            probabilities,
        ),

        "top_label_ece": calculate_top_label_ece(
            y_test,
            probabilities,
        ),

        "classwise_ece": calculate_classwise_ece(
            y_test,
            probabilities,
        ),

        "classification_report": classification_report(
            y_test,
            predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics, probabilities, predictions


# =========================================================
# HIGH-DEMAND THRESHOLD ANALYSIS
# =========================================================

def analyze_high_demand_thresholds(
    y_true,
    probabilities,
):

    high_index = CLASS_LABELS.index("HIGH_DEMAND")

    high_probabilities = probabilities[:, high_index]

    actual_high = (
        np.asarray(y_true) == "HIGH_DEMAND"
    )

    rows = []

    for threshold in HIGH_DEMAND_THRESHOLDS:

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

        rows.append({
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "predicted_high_cases": int(
                np.sum(predicted_high)
            ),
            "actual_high_cases": int(
                np.sum(actual_high)
            ),
            "true_positives": int(
                np.sum(predicted_high & actual_high)
            ),
            "false_positives": int(
                np.sum(predicted_high & ~actual_high)
            ),
            "false_negatives": int(
                np.sum(~predicted_high & actual_high)
            ),
        })

    return pd.DataFrame(rows)


# =========================================================
# RELIABILITY DIAGRAM
# =========================================================

def create_reliability_plot(
    y_test,
    baseline_probabilities,
    calibrated_probabilities,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    y_array = np.asarray(y_test)

    for index, label in enumerate(CLASS_LABELS):

        actual_binary = (
            y_array == label
        ).astype(int)

        baseline_fraction, baseline_mean = calibration_curve(
            actual_binary,
            baseline_probabilities[:, index],
            n_bins=10,
            strategy="quantile",
        )

        calibrated_fraction, calibrated_mean = calibration_curve(
            actual_binary,
            calibrated_probabilities[:, index],
            n_bins=10,
            strategy="quantile",
        )

        ax = axes[index]

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )

        ax.plot(
            baseline_mean,
            baseline_fraction,
            marker="o",
            label="Original",
        )

        ax.plot(
            calibrated_mean,
            calibrated_fraction,
            marker="s",
            label="Sigmoid calibrated",
        )

        ax.set_title(label)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        "CloudMorph — Class-wise Probability Calibration",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        RELIABILITY_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# HIGH-DEMAND THRESHOLD PLOT
# =========================================================

def create_threshold_plot(threshold_df):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="s",
        label="Recall",
    )

    ax.set_title("HIGH_DEMAND Threshold Analysis")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        THRESHOLD_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH CLASSIFIER CALIBRATION EVALUATION")
    logger.info("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    # Load data and original saved artifacts.

    df, X, y, groups, feature_columns = load_dataset()

    package = joblib.load(MODEL_PATH)

    existing_predictions = pd.read_csv(
        PREDICTIONS_PATH,
    )

    if feature_columns != package["feature_columns"]:
        raise RuntimeError(
            "Dataset feature order does not match trained model."
        )

    (
        X_train_validation,
        y_train_validation,
        groups_train_validation,
        X_test,
        y_test,
        groups_test,
    ) = reproduce_original_test_split(
        X,
        y,
        groups,
    )

    # Verify exact original test rows.

    verify_original_test_set(
        groups_test,
        y_test,
        existing_predictions,
    )

    # Verify original model predictions.

    baseline_model = package["model"]

    baseline_metrics, baseline_probabilities, baseline_predictions = (
        evaluate_probabilities(
            "Original Extra Trees",
            baseline_model,
            X_test,
            y_test,
        )
    )

    saved_predictions = (
        existing_predictions["predicted"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        baseline_predictions,
        saved_predictions,
    ):
        raise RuntimeError(
            "Saved model predictions do not match the original "
            "test predictions. Calibration evaluation stopped."
        )

    logger.info(
        "Original model test accuracy: %.4f",
        baseline_metrics["accuracy"],
    )

    # Prepare machine-aware calibration folds.
    # These folds use only training + validation machines.

    logger.info(
        "Preparing group-aware calibration folds..."
    )

    calibration_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    calibration_splits = list(
        calibration_splitter.split(
            X_train_validation,
            y_train_validation,
            groups_train_validation,
        )
    )

    for fold_number, (train_idx, validation_idx) in enumerate(
        calibration_splits,
        start=1,
    ):

        fold_train_machines = set(
            groups_train_validation.iloc[train_idx]
        )

        fold_validation_machines = set(
            groups_train_validation.iloc[validation_idx]
        )

        if fold_train_machines & fold_validation_machines:
            raise RuntimeError(
                f"Machine leakage in calibration fold {fold_number}."
            )

        fold_train_classes = set(
            y_train_validation.iloc[train_idx].unique()
        )

        if fold_train_classes != set(CLASS_LABELS):
            raise RuntimeError(
                f"Calibration fold {fold_number} training "
                "is missing a workload class."
            )

        logger.info(
            "Calibration fold %s: train=%s, validation=%s",
            fold_number,
            len(train_idx),
            len(validation_idx),
        )

    # Build experimental calibrated model.
    # Uses the same Extra Trees hyperparameters as training.

    logger.info(
        "Training experimental sigmoid-calibrated Extra Trees..."
    )

    base_estimator = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_estimator,
        method="sigmoid",
        cv=calibration_splits,
        ensemble=False,
        n_jobs=1,
    )

    calibrated_model.fit(
        X_train_validation,
        y_train_validation,
    )

    # Test only after calibration is fitted.

    calibrated_metrics, calibrated_probabilities, calibrated_predictions = (
        evaluate_probabilities(
            "Sigmoid Calibrated Extra Trees",
            calibrated_model,
            X_test,
            y_test,
        )
    )

    # Save comparison.

    comparison_rows = []

    for metrics in [
        baseline_metrics,
        calibrated_metrics,
    ]:

        comparison_rows.append({
            "model": metrics["model"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "log_loss": metrics["log_loss"],
            "multiclass_brier_score": metrics[
                "multiclass_brier_score"
            ],
            "top_label_ece": metrics["top_label_ece"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    # Threshold analysis.

    threshold_df = analyze_high_demand_thresholds(
        y_test,
        calibrated_probabilities,
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False,
    )

    # Visualizations.

    create_reliability_plot(
        y_test,
        baseline_probabilities,
        calibrated_probabilities,
    )

    create_threshold_plot(
        threshold_df,
    )

    # Report.

    report = {
        "project": "CloudMorph",

        "evaluation": "Probability calibration",

        "calibration_method": "sigmoid",

        "calibration_model": "Extra Trees",

        "model_version": package["model_version"],

        "dataset": {
            "total_samples": int(len(df)),
            "unique_machines": int(groups.nunique()),
            "feature_count": int(len(feature_columns)),
        },

        "test_split": {
            "samples": int(len(X_test)),
            "unique_machines": int(groups_test.nunique()),
            "machine_aware": True,
            "verified_against_saved_predictions": True,
        },

        "calibration_training": {
            "samples": int(len(X_train_validation)),
            "unique_machines": int(
                groups_train_validation.nunique()
            ),
            "method": "StratifiedGroupKFold",
            "folds": len(calibration_splits),
            "ensemble": False,
        },

        "baseline_metrics": baseline_metrics,

        "calibrated_metrics": calibrated_metrics,

        "threshold_analysis": threshold_df.to_dict(
            orient="records"
        ),

        "important_notes": [
            "The original production model was not modified.",
            "The test set was not used to fit the calibrator.",
            "Calibration used training and validation machines only.",
            "Machine-aware cross-validation was used.",
            "Calibration results are experimental.",
            "Probability calibration does not guarantee prediction accuracy.",
            "Workload labels are derived future utilization conditions.",
            "No cloud resources are modified by this evaluation.",
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

    # Completion.

    logger.info("=" * 65)
    logger.info("CALIBRATION EVALUATION COMPLETED")
    logger.info("=" * 65)

    logger.info(
        "Original Brier score: %.6f",
        baseline_metrics["multiclass_brier_score"],
    )

    logger.info(
        "Calibrated Brier score: %.6f",
        calibrated_metrics["multiclass_brier_score"],
    )

    logger.info(
        "Original log loss: %.6f",
        baseline_metrics["log_loss"],
    )

    logger.info(
        "Calibrated log loss: %.6f",
        calibrated_metrics["log_loss"],
    )

    logger.info("Saved report: %s", REPORT_PATH)
    logger.info("Saved metrics: %s", METRICS_PATH)
    logger.info("Saved thresholds: %s", THRESHOLD_PATH)
    logger.info("Saved reliability plot: %s", RELIABILITY_PATH)
    logger.info("Saved threshold plot: %s", THRESHOLD_PLOT_PATH)


if __name__ == "__main__":
    main()
