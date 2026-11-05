"""
CloudMorph - Selected-Feature Extra Trees Experiment

Purpose:
    Test whether the two strongest new trend features
    (memory_recent_max and cpu_recent_max) improve the original
    55-feature Extra Trees approach.

Important:
    - The original production model is never modified.
    - Both added features use historical observations only.
    - Uses the same chronological 40/30/30 split with purging.
    - Selects the risk threshold using validation data only.
    - Evaluates once on the untouched chronological test period.
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


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "server_usage_clean.csv"
)
DATASET_PATH = (
    PROJECT_ROOT / "ml" / "data" / "optimization_training_dataset.csv"
)
PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT / "ml" / "models" / "optimization_classifier"
    / "optimization_classifier.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "ml" / "models" / "extratrees_selected_features"
)
MODEL_PATH = OUTPUT_DIR / "selected_features_extratrees.joblib"
REPORT_PATH = OUTPUT_DIR / "selected_features_report.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "risk_metrics.csv"
CONFUSION_PATH = OUTPUT_DIR / "risk_confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
FEATURE_DATASET_PATH = OUTPUT_DIR / "selected_features_dataset.csv"
THRESHOLD_PATH = OUTPUT_DIR / "validation_threshold_results.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 300
TRAIN_FRACTION = 0.40
TEST_FRACTION = 0.70
MIN_VALIDATION_PRECISION = 0.60
THRESHOLDS = np.round(np.arange(0.01, 0.501, 0.01), 2)

CLASS_LABELS = ["LOW_DEMAND", "NORMAL_DEMAND", "HIGH_DEMAND"]
METADATA_COLUMNS = [
    "machineID", "history_start_step", "history_end_step",
    "target_start_step", "target_end_step", "target",
]
HISTORICAL_COLUMNS = [
    "util_CPU", "util_memory", "util_disk", "load1", "load5", "load15",
]
SELECTED_NEW_FEATURES = ["memory_recent_max", "cpu_recent_max"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("CloudMorphSelectedFeatures")


# ============================================================
# DATA VALIDATION AND LOADING
# ============================================================

def validate_existing_dataset(df):
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if df.empty:
        raise ValueError("Training dataset is empty.")
    if df[METADATA_COLUMNS].isna().any().any():
        raise ValueError("Dataset metadata contains missing values.")

    feature_columns = [c for c in df.columns if c not in METADATA_COLUMNS]
    if len(feature_columns) != 55:
        raise ValueError(f"Expected 55 original features, found {len(feature_columns)}.")
    if set(df["target"].astype(str).unique()) != set(CLASS_LABELS):
        raise ValueError("Unexpected workload target classes.")
    return feature_columns


def load_raw_dataset():
    logger.info("Loading historical raw workload data...")
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(RAW_DATASET_PATH)

    df = pd.read_csv(RAW_DATASET_PATH)
    required = ["machineID", "step", *HISTORICAL_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset missing columns: {missing}")

    for column in HISTORICAL_COLUMNS + ["step"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[required].isna().any().any():
        raise ValueError("Raw dataset contains missing or invalid values.")
    if not np.isfinite(df[HISTORICAL_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Raw dataset contains infinite utilization values.")

    df = df.sort_values(["machineID", "step"]).reset_index(drop=True)
    if df.duplicated(subset=["machineID", "step"]).any():
        raise ValueError("Duplicate machine/step observations found.")
    return df


# ============================================================
# TWO SELECTED HISTORICAL-ONLY FEATURES
# ============================================================

def generate_selected_features(raw_df, original_df):
    logger.info("Generating memory_recent_max and cpu_recent_max...")

    # Build one sorted step array per machine. searchsorted avoids repeatedly
    # filtering an entire pandas DataFrame for every training window.
    machine_data = {}
    for machine_id, machine in raw_df.groupby("machineID", sort=False):
        machine = machine.reset_index(drop=True)
        machine_data[machine_id] = {
            "steps": machine["step"].to_numpy(dtype=np.int64),
            "cpu": machine["util_CPU"].to_numpy(dtype=float),
            "memory": machine["util_memory"].to_numpy(dtype=float),
        }

    records = []
    for row in original_df[
        ["machineID", "history_start_step", "history_end_step"]
    ].itertuples(index=False, name=None):
        machine_id, start_step, end_step = row
        if machine_id not in machine_data:
            raise ValueError(f"Machine {machine_id} not found in raw dataset.")

        data = machine_data[machine_id]
        steps = data["steps"]
        start = int(start_step)
        end = int(end_step)
        left = int(np.searchsorted(steps, start, side="left"))
        right = int(np.searchsorted(steps, end, side="right"))

        if right - left != 12:
            raise ValueError(
                f"Expected 12 historical observations for machine {machine_id}, "
                f"steps {start}-{end}; found {right - left}."
            )
        if int(steps[left]) != start or int(steps[right - 1]) != end:
            raise ValueError("Historical window boundaries do not match.")

        # The last six observations are the most recent 30 historical minutes.
        records.append({
            "memory_recent_max": float(np.max(data["memory"][left + 6:right])),
            "cpu_recent_max": float(np.max(data["cpu"][left + 6:right])),
        })

    features = pd.DataFrame(records, index=original_df.index)
    if features.isna().any().any():
        raise ValueError("Selected features contain missing values.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Selected features contain non-finite values.")
    return features


# ============================================================
# CHRONOLOGICAL SPLIT (SAME AS TREND-ENHANCED EXPERIMENT)
# ============================================================

def create_temporal_splits(df, X, y):
    logger.info("Creating chronological splits...")
    steps = np.sort(df["history_start_step"].unique())
    validation_index = math.ceil(len(steps) * TRAIN_FRACTION)
    test_index = math.ceil(len(steps) * TEST_FRACTION)
    if test_index >= len(steps):
        raise ValueError("Invalid chronological split boundaries.")

    validation_start = steps[validation_index]
    test_start = steps[test_index]

    train_mask = (
        (df["history_start_step"] < validation_start)
        & (df["target_end_step"] < validation_start)
    )
    validation_mask = (
        (df["history_start_step"] >= validation_start)
        & (df["target_end_step"] < test_start)
    )
    test_mask = df["history_start_step"] >= test_start

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()
    purged_count = len(df) - len(train_df) - len(validation_df) - len(test_df)

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("A chronological split is empty.")
    if train_df["target_end_step"].max() >= validation_start:
        raise RuntimeError("Training targets overlap validation history.")
    if validation_df["target_end_step"].max() >= test_start:
        raise RuntimeError("Validation targets overlap test history.")
    if test_df["history_start_step"].min() != test_start:
        raise RuntimeError("Unexpected test boundary.")

    for name, part in [("Training", train_df), ("Validation", validation_df), ("Testing", test_df)]:
        missing = set(CLASS_LABELS) - set(part["target"].unique())
        if missing:
            raise ValueError(f"{name} split missing classes: {missing}")

    logger.info(
        "Training: %s | Validation: %s | Test: %s | Purged: %s",
        len(train_df), len(validation_df), len(test_df), purged_count,
    )
    return (
        train_df, validation_df, test_df,
        X.loc[train_mask], y.loc[train_mask],
        X.loc[validation_mask], y.loc[validation_mask],
        X.loc[test_mask], y.loc[test_mask],
        validation_start, test_start, purged_count,
    )


# ============================================================
# MODEL AND METRICS
# ============================================================

def create_model():
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_high_probabilities(model, X):
    probabilities = model.predict_proba(X)
    high_index = list(model.classes_).index("HIGH_DEMAND")
    return probabilities[:, high_index]


def calculate_risk_metrics(y_true, alerts):
    actual = y_true.to_numpy() == "HIGH_DEMAND"
    alerts = np.asarray(alerts, dtype=bool)
    tp = int(np.sum(actual & alerts))
    fp = int(np.sum(~actual & alerts))
    fn = int(np.sum(actual & ~alerts))
    tn = int(np.sum(~actual & ~alerts))
    precision = precision_score(actual, alerts, zero_division=0)
    recall = recall_score(actual, alerts, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(actual, alerts, zero_division=0)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "risk_alerts": int(alerts.sum()),
        "actual_high_demand_cases": int(actual.sum()),
    }


def select_threshold(y_validation, probabilities):
    logger.info("Selecting threshold using validation data only...")
    results = []
    for threshold in THRESHOLDS:
        metrics = calculate_risk_metrics(
            y_validation, probabilities >= threshold
        )
        results.append({
            "threshold": float(threshold),
            **metrics,
            "meets_minimum_precision": metrics["precision"] >= MIN_VALIDATION_PRECISION,
        })

    results_df = pd.DataFrame(results)
    eligible = results_df[results_df["meets_minimum_precision"]]
    if eligible.empty:
        raise ValueError(
            "No validation threshold achieved minimum precision "
            f"of {MIN_VALIDATION_PRECISION:.2f}. Do not select using test data."
        )
    eligible = eligible.sort_values(
        ["f1", "recall", "precision"], ascending=False
    )
    selected = float(eligible.iloc[0]["threshold"])
    logger.info("Selected validation threshold: %.2f", selected)
    return results_df, selected


def classification_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(
            y_true, predictions, labels=CLASS_LABELS,
            average="macro", zero_division=0,
        )),
        "high_demand_precision": float(precision_score(
            y_true, predictions, labels=["HIGH_DEMAND"],
            average=None, zero_division=0,
        )[0]),
        "high_demand_recall": float(recall_score(
            y_true, predictions, labels=["HIGH_DEMAND"],
            average=None, zero_division=0,
        )[0]),
    }


# ============================================================
# SAVE EXPERIMENT OUTPUTS
# ============================================================

def save_results(
    model, test_df, y_test, workload_predictions, high_probabilities,
    risk_metrics, classification_results, feature_columns,
    selected_threshold, threshold_results, split_info,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_results.to_csv(THRESHOLD_PATH, index=False)

    prediction_df = test_df[[
        "machineID", "history_start_step", "history_end_step",
        "target_start_step", "target_end_step",
    ]].copy()
    prediction_df["actual_workload"] = y_test.to_numpy()
    prediction_df["predicted_workload"] = workload_predictions
    prediction_df["high_demand_probability"] = high_probabilities
    prediction_df["risk_threshold"] = selected_threshold
    prediction_df["risk_alert"] = np.where(
        high_probabilities >= selected_threshold, "HIGH_RISK", "NO_HIGH_RISK"
    )
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    risk_alerts = high_probabilities >= selected_threshold
    matrix = confusion_matrix(
        y_test.to_numpy() == "HIGH_DEMAND", risk_alerts, labels=[False, True]
    )
    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual_No_High_Demand", "Actual_High_Demand"],
        columns=["Predicted_No_Risk", "Predicted_High_Risk"],
    )
    confusion_df.index.name = "Actual"
    confusion_df.to_csv(CONFUSION_PATH)

    pd.DataFrame([
        {"evaluation": "Workload classification", **classification_results},
        {"evaluation": "Independent high-demand risk", **risk_metrics},
    ]).to_csv(METRICS_PATH, index=False)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS,
        "model_name": "Selected-Feature Extra Trees",
        "model_version": "experimental-1.0.0",
        "risk_threshold": selected_threshold,
        "metadata": {
            "dataset": str(DATASET_PATH),
            "historical_window_minutes": 60,
            "prediction_horizon_minutes": 30,
            "training_method": "chronological_40_30_30_with_purging",
            "random_state": RANDOM_STATE,
            "original_production_model_modified": False,
        },
    }
    joblib.dump(model_package, MODEL_PATH)

    report = {
        "project": "CloudMorph",
        "experiment": "Selected-Feature Extra Trees",
        "status": "experimental",
        "model_version": "experimental-1.0.0",
        "dataset": {
            "samples": int(sum(split_info["counts"]) + split_info["purged_samples"]),
            "original_feature_count": 55,
            "selected_new_feature_count": len(SELECTED_NEW_FEATURES),
            "total_feature_count": len(feature_columns),
            "selected_new_features": SELECTED_NEW_FEATURES,
            "removed_experimental_features": [
                "cpu_recent_change", "memory_recent_change",
                "cpu_acceleration", "memory_acceleration",
                "cpu_rolling_std", "memory_rolling_std",
            ],
        },
        "model_configuration": {
            "n_estimators": N_ESTIMATORS,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "split": split_info,
        "threshold_selection": {
            "method": "Maximum validation F1 subject to minimum precision",
            "minimum_validation_precision": MIN_VALIDATION_PRECISION,
            "selected_threshold": selected_threshold,
        },
        "test_classification_metrics": classification_results,
        "test_independent_risk_metrics": risk_metrics,
        "test_classification_report": classification_report(
            y_test, workload_predictions, labels=CLASS_LABELS,
            output_dict=True, zero_division=0,
        ),
        "important_notes": [
            "Both selected additional features use historical observations only.",
            "Future observations are used exclusively for workload labels.",
            "The test period was not used for threshold selection.",
            "Chronological boundaries were purged to prevent target overlap.",
            "The same machines occur in all chronological periods.",
            "The original production model was not modified.",
            "This model is experimental and is not deployed to the API.",
            "HIGH_RISK is a model-based warning, not a confirmed incident.",
            "No cloud resources are modified.",
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("Saved model: %s", MODEL_PATH)
    logger.info("Saved report: %s", REPORT_PATH)
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)
    logger.info("Saved metrics: %s", METRICS_PATH)
    logger.info("Saved feature importance: %s", FEATURE_IMPORTANCE_PATH)
    logger.info("Saved threshold results: %s", THRESHOLD_PATH)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 70)
    logger.info("CLOUDMORPH SELECTED-FEATURE EXTRA TREES EXPERIMENT")
    logger.info("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)
    original_df = pd.read_csv(DATASET_PATH)
    original_feature_columns = validate_existing_dataset(original_df)

    # Verify the production feature schema without changing the model.
    if PRODUCTION_MODEL_PATH.exists():
        production_package = joblib.load(PRODUCTION_MODEL_PATH)
        if production_package.get("feature_columns") != original_feature_columns:
            raise ValueError("Original feature schema does not match production model.")
        logger.info("Original production feature schema verified.")
    else:
        raise FileNotFoundError(
            f"Production model package required for schema verification: {PRODUCTION_MODEL_PATH}"
        )

    raw_df = load_raw_dataset()
    selected_features = generate_selected_features(raw_df, original_df)

    experiment_df = pd.concat([
        original_df[METADATA_COLUMNS].reset_index(drop=True),
        original_df[original_feature_columns].reset_index(drop=True),
        selected_features.reset_index(drop=True),
    ], axis=1)
    experiment_df.to_csv(FEATURE_DATASET_PATH, index=False)

    feature_columns = original_feature_columns + SELECTED_NEW_FEATURES
    if len(feature_columns) != 57:
        raise RuntimeError(f"Expected 57 features, got {len(feature_columns)}.")

    X = experiment_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains missing or non-finite values.")
    y = experiment_df["target"].astype(str)

    (
        train_df, validation_df, test_df,
        X_train, y_train, X_validation, y_validation,
        X_test, y_test, validation_start, test_start, purged_count,
    ) = create_temporal_splits(experiment_df, X, y)

    model = create_model()
    logger.info("Training model with %s features...", len(feature_columns))
    model.fit(X_train, y_train)

    validation_probabilities = get_high_probabilities(model, X_validation)
    threshold_results, selected_threshold = select_threshold(
        y_validation, validation_probabilities
    )

    workload_predictions = model.predict(X_test)
    high_probabilities = get_high_probabilities(model, X_test)
    risk_alerts = high_probabilities >= selected_threshold
    risk_metrics = calculate_risk_metrics(y_test, risk_alerts)
    classification_results = classification_metrics(y_test, workload_predictions)

    logger.info("=" * 70)
    logger.info("SELECTED-FEATURE TEST RESULTS")
    logger.info("=" * 70)
    logger.info("Selected threshold: %.2f", selected_threshold)
    logger.info("Classification metrics: %s", classification_results)
    logger.info("Independent risk metrics: %s", risk_metrics)
    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_test, workload_predictions, labels=CLASS_LABELS, zero_division=0
        ),
    )

    split_info = {
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(validation_df)),
        "test_samples": int(len(test_df)),
        "purged_samples": int(purged_count),
        "validation_start_history_step": float(validation_start),
        "test_start_history_step": float(test_start),
        "counts": [int(len(train_df)), int(len(validation_df)), int(len(test_df))],
    }
    save_results(
        model, test_df, y_test, workload_predictions, high_probabilities,
        risk_metrics, classification_results, feature_columns,
        selected_threshold, threshold_results, split_info,
    )
    logger.info("=" * 70)
    logger.info("SELECTED-FEATURE EXPERIMENT COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
