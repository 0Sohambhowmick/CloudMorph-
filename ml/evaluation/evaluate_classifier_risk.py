
"""
CloudMorph - Independent High-Demand Risk Evaluation

Keeps workload classification and high-demand risk detection
as two independent outputs.

- Workload prediction remains unchanged.
- Risk alert uses a threshold selected on validation data.
- Final testing uses the untouched chronological test period.
- Production model and API are not modified.
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
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
)

PRODUCTION_MODEL_PATH = (
    MODEL_DIR / "optimization_classifier.joblib"
)

THRESHOLD_REPORT_PATH = (
    MODEL_DIR
    / "threshold_evaluation"
    / "threshold_evaluation_report.json"
)

OUTPUT_DIR = MODEL_DIR / "risk_evaluation"

REPORT_PATH = OUTPUT_DIR / "risk_evaluation_report.json"
METRICS_PATH = OUTPUT_DIR / "risk_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "risk_predictions.csv"
CONFUSION_PATH = OUTPUT_DIR / "risk_confusion_matrix.csv"
FALSE_ALERTS_PATH = OUTPUT_DIR / "risk_false_alerts_by_class.csv"

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


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphRiskEvaluation")


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
            f"Missing required columns: {missing}"
        )

    if df[METADATA_COLUMNS].isna().any().any():
        raise ValueError(
            "Dataset metadata contains missing values."
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
            "Missing or non-numeric features detected."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite feature values detected."
        )

    y = df[TARGET_COLUMN].astype(str)

    if set(y.unique()) != set(CLASS_LABELS):
        raise ValueError(
            "Unexpected target classes detected."
        )

    logger.info("Samples: %s", len(df))
    logger.info("Machines: %s", df[GROUP_COLUMN].nunique())
    logger.info("Features: %s", len(feature_columns))

    return df, X, y, feature_columns


# ---------------------------------------------------------
# LOAD FROZEN VALIDATION THRESHOLD
# ---------------------------------------------------------

def load_selected_threshold():

    logger.info("Loading validation-selected risk threshold...")

    if not THRESHOLD_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Threshold evaluation report not found: "
            f"{THRESHOLD_REPORT_PATH}"
        )

    with open(
        THRESHOLD_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    threshold = report.get(
        "threshold_selection", {}
    ).get("selected_threshold")

    if threshold is None:
        raise ValueError(
            "Selected threshold is missing from the report."
        )

    threshold = float(threshold)

    if not 0 < threshold < 1:
        raise ValueError(
            f"Invalid risk threshold: {threshold}"
        )

    expected_training_samples = 33150
    expected_validation_samples = 20475
    expected_test_samples = 37050

    split = report.get("split", {})

    if (
        split.get("training_samples") != expected_training_samples
        or split.get("validation_samples") != expected_validation_samples
        or split.get("test_samples") != expected_test_samples
    ):
        raise ValueError(
            "Threshold report does not match the expected "
            "chronological experiment split."
        )

    logger.info(
        "Frozen validation threshold: %.4f",
        threshold,
    )

    return threshold


# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
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

    train_mask = (
        (df["history_start_step"] < validation_start)
        &
        (df["target_end_step"] < validation_start)
    )

    validation_mask = (
        (df["history_start_step"] >= validation_start)
        &
        (df["target_end_step"] < test_start)
    )

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

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "A chronological split is empty."
        )

    if train_df["target_end_step"].max() >= validation_start:
        raise RuntimeError(
            "Training targets overlap validation history."
        )

    if validation_df["target_end_step"].max() >= test_start:
        raise RuntimeError(
            "Validation targets overlap test history."
        )

    if test_df["history_start_step"].min() != test_start:
        raise RuntimeError(
            "Unexpected test period boundary."
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

    logger.info("Training samples: %s", len(train_df))
    logger.info("Validation samples: %s", len(validation_df))
    logger.info("Testing samples: %s", len(test_df))
    logger.info("Purged samples: %s", purged_count)

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
# PROBABILITY EXTRACTION
# ---------------------------------------------------------

def get_high_demand_probabilities(model, X):

    probabilities = model.predict_proba(X)

    high_index = list(model.classes_).index(
        "HIGH_DEMAND"
    )

    return probabilities[:, high_index]


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_classification_metrics(y_true, predictions):

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
    }


def calculate_risk_metrics(y_true, risk_alerts):

    actual_high = (
        y_true.to_numpy() == "HIGH_DEMAND"
    )

    risk_alerts = np.asarray(
        risk_alerts,
        dtype=bool,
    )

    true_positives = int(
        np.sum(actual_high & risk_alerts)
    )

    false_positives = int(
        np.sum(~actual_high & risk_alerts)
    )

    false_negatives = int(
        np.sum(actual_high & ~risk_alerts)
    )

    true_negatives = int(
        np.sum(~actual_high & ~risk_alerts)
    )

    precision = precision_score(
        actual_high,
        risk_alerts,
        zero_division=0,
    )

    recall = recall_score(
        actual_high,
        risk_alerts,
        zero_division=0,
    )

    specificity = (
        true_negatives / (true_negatives + false_positives)
        if true_negatives + false_positives > 0
        else 0.0
    )

    f1 = f1_score(
        actual_high,
        risk_alerts,
        zero_division=0,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_positive_rate": float(1 - specificity),
        "f1": float(f1),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "risk_alerts": int(risk_alerts.sum()),
        "actual_high_demand_cases": int(actual_high.sum()),
    }


# ---------------------------------------------------------
# FALSE ALERT BREAKDOWN
# ---------------------------------------------------------

def calculate_false_alert_breakdown(
    y_true,
    risk_alerts,
):

    results = []

    for label in CLASS_LABELS:

        class_mask = (
            y_true.to_numpy() == label
        )

        class_alerts = (
            np.asarray(risk_alerts, dtype=bool)[class_mask]
        )

        count = int(class_mask.sum())
        alerts = int(class_alerts.sum())

        results.append({
            "actual_class": label,
            "total_cases": count,
            "risk_alerts": alerts,
            "alert_rate_percent": (
                alerts / count * 100
                if count else 0.0
            ),
            "is_false_alert_class": (
                label != "HIGH_DEMAND"
            ),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(
    train_df,
    validation_df,
    test_df,
    purged_count,
    threshold,
    y_test,
    workload_predictions,
    high_probabilities,
    risk_alerts,
    classification_metrics,
    risk_metrics,
    false_alert_breakdown,
    validation_start,
    test_start,
    feature_columns,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve original workload predictions.

    prediction_df = test_df[
        [
            "machineID",
            "history_start_step",
            "history_end_step",
            "target_start_step",
            "target_end_step",
        ]
    ].copy()

    prediction_df["actual_workload"] = y_test.to_numpy()

    prediction_df["predicted_workload"] = (
        workload_predictions
    )

    prediction_df["high_demand_probability"] = (
        high_probabilities
    )

    prediction_df["risk_threshold"] = threshold

    prediction_df["risk_alert"] = np.where(
        risk_alerts,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Binary risk confusion matrix.

    risk_matrix = confusion_matrix(
        y_test.to_numpy() == "HIGH_DEMAND",
        risk_alerts,
        labels=[False, True],
    )

    risk_matrix_df = pd.DataFrame(
        risk_matrix,
        index=["Actual_No_High_Demand", "Actual_High_Demand"],
        columns=["Predicted_No_Risk", "Predicted_High_Risk"],
    )

    risk_matrix_df.index.name = "Actual"

    risk_matrix_df.to_csv(
        CONFUSION_PATH
    )

    false_alert_breakdown.to_csv(
        FALSE_ALERTS_PATH,
        index=False,
    )

    # Combined metrics.

    metrics_df = pd.DataFrame([
        {
            "evaluation": "Workload classification",
            **classification_metrics,
        },
        {
            "evaluation": "Independent high-demand risk",
            **risk_metrics,
        },
    ])

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    report = {
        "project": "CloudMorph",
        "evaluation": "Independent high-demand risk detection",
        "model": "Extra Trees",
        "model_version": "1.0.0",

        "model_configuration": {
            "n_estimators": N_ESTIMATORS,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },

        "split": {
            "training_samples": int(len(train_df)),
            "validation_samples": int(len(validation_df)),
            "test_samples": int(len(test_df)),
            "purged_samples": int(purged_count),
            "validation_start_history_step": float(validation_start),
            "test_start_history_step": float(test_start),
        },

        "risk_configuration": {
            "threshold": threshold,
            "threshold_source": "Previous validation-only threshold selection",
            "risk_alert_label": "HIGH_RISK",
            "no_risk_alert_label": "NO_HIGH_RISK",
        },

        "workload_classification_metrics": classification_metrics,

        "independent_risk_metrics": risk_metrics,

        "false_alert_breakdown": (
            false_alert_breakdown.to_dict(
                orient="records"
            )
        ),

        "workload_classification_report": classification_report(
            y_test,
            workload_predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),

        "risk_confusion_matrix": {
            "labels": [
                "Actual_No_High_Demand",
                "Actual_High_Demand",
            ],
            "matrix": risk_matrix.tolist(),
        },

        "feature_count": len(feature_columns),

        "important_notes": [
            "Workload classification and risk alerts are independent outputs.",
            "The original workload prediction is never overwritten.",
            "The risk threshold was selected using validation data only.",
            "The final test set was not used to select the threshold.",
            "Training and validation target windows were purged "
            "at chronological boundaries.",
            "The same machines appear in all chronological periods.",
            "The production model was not modified.",
            "A HIGH_RISK alert is a model-based warning, not a confirmed "
            "cloud incident or scaling instruction.",
            "Workload labels represent derived future utilization conditions.",
            "The experiment does not establish long-term production performance.",
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
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)
    logger.info("Saved confusion matrix: %s", CONFUSION_PATH)
    logger.info("Saved false alert breakdown: %s", FALSE_ALERTS_PATH)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH INDEPENDENT RISK EVALUATION")
    logger.info("=" * 65)

    (
        df,
        X,
        y,
        feature_columns,
    ) = load_dataset()

    # Verify production model schema.

    if PRODUCTION_MODEL_PATH.exists():

        package = joblib.load(
            PRODUCTION_MODEL_PATH
        )

        if package.get("feature_columns") != feature_columns:
            raise ValueError(
                "Production model feature schema mismatch."
            )

        logger.info(
            "Production model feature schema verified."
        )

    threshold = load_selected_threshold()

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

    # Train independent experimental model.

    model = create_model()

    logger.info("Training temporal Extra Trees model...")

    model.fit(
        X_train,
        y_train,
    )

    # Workload classification remains unchanged.

    logger.info("Generating original workload predictions...")

    workload_predictions = model.predict(
        X_test
    )

    # Risk detection is independent.

    high_probabilities = get_high_demand_probabilities(
        model,
        X_test,
    )

    risk_alerts = (
        high_probabilities >= threshold
    )

    classification_metrics = calculate_classification_metrics(
        y_test,
        workload_predictions,
    )

    risk_metrics = calculate_risk_metrics(
        y_test,
        risk_alerts,
    )

    false_alert_breakdown = calculate_false_alert_breakdown(
        y_test,
        risk_alerts,
    )

    logger.info("=" * 65)
    logger.info("WORKLOAD CLASSIFICATION RESULTS")
    logger.info("=" * 65)

    logger.info(
        "Classification metrics: %s",
        classification_metrics,
    )

    logger.info("=" * 65)
    logger.info("INDEPENDENT HIGH-DEMAND RISK RESULTS")
    logger.info("=" * 65)

    logger.info(
        "Frozen risk threshold: %.4f",
        threshold,
    )

    logger.info(
        "Risk metrics: %s",
        risk_metrics,
    )

    logger.info(
        "False alerts by actual class:\n%s",
        false_alert_breakdown.to_string(index=False),
    )

    save_results(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        purged_count=purged_count,
        threshold=threshold,
        y_test=y_test,
        workload_predictions=workload_predictions,
        high_probabilities=high_probabilities,
        risk_alerts=risk_alerts,
        classification_metrics=classification_metrics,
        risk_metrics=risk_metrics,
        false_alert_breakdown=false_alert_breakdown,
        validation_start=validation_start,
        test_start=test_start,
        feature_columns=feature_columns,
    )

    logger.info("=" * 65)
    logger.info("INDEPENDENT RISK EVALUATION COMPLETED")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
