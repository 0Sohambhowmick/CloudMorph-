
"""
CloudMorph - Trend-Enhanced Extra Trees Experiment

Purpose:
    Investigate whether additional historical trend features
    improve future HIGH_DEMAND detection.

Important:
    - Original production model is never modified.
    - Uses historical observations only for features.
    - Uses chronological train/validation/test splits.
    - Selects risk threshold using validation only.
    - Evaluates on an untouched chronological test period.
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
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

PRODUCTION_MODEL_PATH = (
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
    / "extratrees_trend_enhanced"
)

MODEL_PATH = OUTPUT_DIR / "trend_enhanced_extratrees.joblib"
REPORT_PATH = OUTPUT_DIR / "trend_enhanced_report.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "risk_metrics.csv"
CONFUSION_PATH = OUTPUT_DIR / "risk_confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
FEATURE_DATASET_PATH = OUTPUT_DIR / "trend_enhanced_features.csv"
THRESHOLD_PATH = OUTPUT_DIR / "validation_threshold_results.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 300

TRAIN_FRACTION = 0.40
TEST_FRACTION = 0.70

MIN_VALIDATION_PRECISION = 0.60

THRESHOLDS = np.round(
    np.arange(0.01, 0.501, 0.01),
    2,
)

CLASS_LABELS = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]

METADATA_COLUMNS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
    "target",
]

HISTORICAL_COLUMNS = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

NEW_FEATURE_COLUMNS = [
    "cpu_recent_change",
    "memory_recent_change",
    "cpu_acceleration",
    "memory_acceleration",
    "cpu_rolling_std",
    "memory_rolling_std",
    "cpu_recent_max",
    "memory_recent_max",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphTrendEnhanced")


# ============================================================
# VALIDATION
# ============================================================

def validate_existing_dataset(df):

    missing = [
        column
        for column in METADATA_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metadata columns: {missing}"
        )

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df[METADATA_COLUMNS].isna().any().any():
        raise ValueError(
            "Dataset metadata contains missing values."
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    if len(feature_columns) != 55:
        raise ValueError(
            f"Expected 55 original features, "
            f"found {len(feature_columns)}."
        )

    if set(df["target"].astype(str).unique()) != set(CLASS_LABELS):
        raise ValueError("Unexpected workload target classes.")

    return feature_columns


def load_raw_dataset():

    logger.info("Loading historical raw workload data...")

    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(RAW_DATASET_PATH)

    df = pd.read_csv(RAW_DATASET_PATH)

    required = [
        "machineID",
        "step",
        *HISTORICAL_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Raw dataset missing columns: {missing}"
        )

    for column in HISTORICAL_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Raw dataset contains missing or invalid values."
        )

    if not np.isfinite(
        df[HISTORICAL_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Raw dataset contains infinite values."
        )

    df = df.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    if df.duplicated(
        subset=["machineID", "step"]
    ).any():
        raise ValueError(
            "Duplicate machine/step observations found."
        )

    return df


# ============================================================
# HISTORICAL-ONLY TREND FEATURES
# ============================================================

def calculate_trend_features(history):

    cpu = history["util_CPU"].to_numpy(dtype=float)
    memory = history["util_memory"].to_numpy(dtype=float)

    # The 12 observations represent 60 historical minutes.
    # Divide them into two consecutive 30-minute periods.

    cpu_previous = cpu[:6]
    cpu_recent = cpu[6:]

    memory_previous = memory[:6]
    memory_recent = memory[6:]

    cpu_previous_mean = float(np.mean(cpu_previous))
    cpu_recent_mean = float(np.mean(cpu_recent))

    memory_previous_mean = float(np.mean(memory_previous))
    memory_recent_mean = float(np.mean(memory_recent))

    # Change in utilization between historical periods.
    cpu_change = cpu_recent_mean - cpu_previous_mean
    memory_change = memory_recent_mean - memory_previous_mean

    # Acceleration = change in the recent period's mean
    # relative to the previous period's mean.
    # This is a historical trend-change indicator,
    # not a physical acceleration measurement.

    cpu_acceleration = (
        float(np.mean(cpu[9:12]))
        - float(np.mean(cpu[3:6]))
    )

    memory_acceleration = (
        float(np.mean(memory[9:12]))
        - float(np.mean(memory[3:6]))
    )

    return {
        "cpu_recent_change": cpu_change,
        "memory_recent_change": memory_change,

        "cpu_acceleration": cpu_acceleration,
        "memory_acceleration": memory_acceleration,

        "cpu_rolling_std": float(
            np.std(cpu[6:], ddof=0)
        ),

        "memory_rolling_std": float(
            np.std(memory[6:], ddof=0)
        ),

        "cpu_recent_max": float(
            np.max(cpu[6:])
        ),

        "memory_recent_max": float(
            np.max(memory[6:])
        ),
    }


def generate_enhanced_features(raw_df, original_df):

    logger.info(
        "Generating eight historical-only trend features..."
    )

    records = []

    grouped = {
        machine_id: machine.reset_index(drop=True)
        for machine_id, machine in raw_df.groupby(
            "machineID",
            sort=False,
        )
    }

    for row in original_df.itertuples(index=False):

        machine_id = row.machineID
        history_start = int(row.history_start_step)
        history_end = int(row.history_end_step)

        if machine_id not in grouped:
            raise ValueError(
                f"Machine {machine_id} not found in raw dataset."
            )

        machine = grouped[machine_id]

        history = machine[
            (machine["step"] >= history_start)
            & (machine["step"] <= history_end)
        ]

        if len(history) != 12:
            raise ValueError(
                "Expected exactly 12 historical observations "
                f"for machine {machine_id}, "
                f"steps {history_start}-{history_end}; "
                f"found {len(history)}."
            )

        if (
            history["step"].iloc[0] != history_start
            or history["step"].iloc[-1] != history_end
        ):
            raise ValueError(
                "Historical window boundaries do not match."
            )

        records.append(
            calculate_trend_features(history)
        )

    enhanced = pd.DataFrame(
        records,
        index=original_df.index,
    )

    if enhanced.isna().any().any():
        raise ValueError(
            "Generated trend features contain missing values."
        )

    if not np.isfinite(enhanced.to_numpy()).all():
        raise ValueError(
            "Generated trend features contain non-finite values."
        )

    return enhanced


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

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

    if test_index >= len(steps):
        raise ValueError("Invalid chronological split boundaries.")

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

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError("A chronological split is empty.")

    if train_df["target_end_step"].max() >= validation_start:
        raise RuntimeError(
            "Training targets overlap validation history."
        )

    if validation_df["target_end_step"].max() >= test_start:
        raise RuntimeError(
            "Validation targets overlap test history."
        )

    if test_df["history_start_step"].min() != test_start:
        raise RuntimeError("Unexpected test boundary.")

    for name, part in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df),
    ]:

        missing = set(CLASS_LABELS) - set(
            part["target"].unique()
        )

        if missing:
            raise ValueError(
                f"{name} split missing classes: {missing}"
            )

    logger.info(
        "Training: %s | Validation: %s | Test: %s | Purged: %s",
        len(train_df),
        len(validation_df),
        len(test_df),
        purged_count,
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


# ============================================================
# MODEL
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

    high_index = list(model.classes_).index(
        "HIGH_DEMAND"
    )

    return probabilities[:, high_index]


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def calculate_risk_metrics(y_true, alerts):

    actual = (
        y_true.to_numpy() == "HIGH_DEMAND"
    )

    alerts = np.asarray(alerts, dtype=bool)

    tp = int(np.sum(actual & alerts))
    fp = int(np.sum(~actual & alerts))
    fn = int(np.sum(actual & ~alerts))
    tn = int(np.sum(~actual & ~alerts))

    precision = precision_score(
        actual,
        alerts,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        alerts,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                actual,
                alerts,
                zero_division=0,
            )
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "risk_alerts": int(alerts.sum()),
        "actual_high_demand_cases": int(actual.sum()),
    }


def select_threshold(y_validation, probabilities):

    logger.info(
        "Selecting threshold using validation data only..."
    )

    results = []

    for threshold in THRESHOLDS:

        alerts = probabilities >= threshold

        metrics = calculate_risk_metrics(
            y_validation,
            alerts,
        )

        results.append({
            "threshold": float(threshold),
            **metrics,
            "meets_minimum_precision": (
                metrics["precision"] >= MIN_VALIDATION_PRECISION
            ),
        })

    results_df = pd.DataFrame(results)

    eligible = results_df[
        results_df["meets_minimum_precision"]
    ]

    if eligible.empty:
        raise ValueError(
            "No validation threshold achieved minimum precision "
            f"of {MIN_VALIDATION_PRECISION:.2f}. "
            "Do not select a threshold using test data."
        )

    eligible = eligible.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    )

    selected = float(
        eligible.iloc[0]["threshold"]
    )

    logger.info(
        "Selected validation threshold: %.2f",
        selected,
    )

    return results_df, selected


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(y_true, predictions):

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
                y_true,
                predictions,
                labels=["HIGH_DEMAND"],
                average=None,
                zero_division=0,
            )[0]
        ),
        "high_demand_recall": float(
            recall_score(
                y_true,
                predictions,
                labels=["HIGH_DEMAND"],
                average=None,
                zero_division=0,
            )[0]
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    model,
    train_df,
    validation_df,
    test_df,
    purged_count,
    validation_start,
    test_start,
    threshold_results,
    selected_threshold,
    y_test,
    workload_predictions,
    high_probabilities,
    risk_metrics,
    classification_results,
    feature_columns,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_results.to_csv(
        THRESHOLD_PATH,
        index=False,
    )

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

    prediction_df["predicted_workload"] = workload_predictions

    prediction_df["high_demand_probability"] = high_probabilities

    prediction_df["risk_threshold"] = selected_threshold

    prediction_df["risk_alert"] = np.where(
        high_probabilities >= selected_threshold,
        "HIGH_RISK",
        "NO_HIGH_RISK",
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    risk_alerts = high_probabilities >= selected_threshold

    matrix = confusion_matrix(
        y_test.to_numpy() == "HIGH_DEMAND",
        risk_alerts,
        labels=[False, True],
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            "Actual_No_High_Demand",
            "Actual_High_Demand",
        ],
        columns=[
            "Predicted_No_Risk",
            "Predicted_High_Risk",
        ],
    )

    confusion_df.index.name = "Actual"

    confusion_df.to_csv(CONFUSION_PATH)

    metrics_df = pd.DataFrame([
        {
            "evaluation": "Workload classification",
            **classification_results,
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

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS,
        "model_name": "Trend-Enhanced Extra Trees",
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

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    report = {
        "project": "CloudMorph",
        "experiment": "Trend-Enhanced Extra Trees",
        "status": "experimental",
        "model_version": "experimental-1.0.0",

        "dataset": {
            "samples": int(len(test_df) + len(train_df) + len(validation_df) + purged_count),
            "original_feature_count": 55,
            "new_trend_feature_count": len(NEW_FEATURE_COLUMNS),
            "total_feature_count": len(feature_columns),
            "new_features": NEW_FEATURE_COLUMNS,
        },

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

        "threshold_selection": {
            "method": "Maximum validation F1 subject to minimum precision",
            "minimum_validation_precision": MIN_VALIDATION_PRECISION,
            "selected_threshold": selected_threshold,
            "threshold_results": threshold_results.to_dict(
                orient="records"
            ),
        },

        "test_classification_metrics": classification_results,
        "test_independent_risk_metrics": risk_metrics,

        "test_classification_report": classification_report(
            y_test,
            workload_predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),

        "important_notes": [
            "All eight additional features use historical observations only.",
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
    logger.info("CLOUDMORPH TREND-ENHANCED EXTRA TREES EXPERIMENT")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    original_df = pd.read_csv(DATASET_PATH)

    original_feature_columns = validate_existing_dataset(
        original_df
    )

    # Verify original production schema without changing it.

    if PRODUCTION_MODEL_PATH.exists():

        package = joblib.load(
            PRODUCTION_MODEL_PATH
        )

        if package.get("feature_columns") != original_feature_columns:
            raise ValueError(
                "Original feature schema does not match "
                "the production model."
            )

        logger.info(
            "Original production feature schema verified."
        )

    raw_df = load_raw_dataset()

    trend_features = generate_enhanced_features(
        raw_df,
        original_df,
    )

    # Save generated feature table separately for auditing.

    enhanced_df = pd.concat(
        [
            original_df[METADATA_COLUMNS].reset_index(drop=True),
            original_df[original_feature_columns].reset_index(drop=True),
            trend_features.reset_index(drop=True),
        ],
        axis=1,
    )

    enhanced_df.to_csv(
        FEATURE_DATASET_PATH,
        index=False,
    )

    feature_columns = (
        original_feature_columns + NEW_FEATURE_COLUMNS
    )

    X = enhanced_df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "Enhanced feature matrix contains missing values."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Enhanced feature matrix contains infinite values."
        )

    y = enhanced_df["target"].astype(str)

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
        enhanced_df,
        X,
        y,
    )

    model = create_model()

    logger.info(
        "Training model with %s features...",
        len(feature_columns),
    )

    model.fit(
        X_train,
        y_train,
    )

    # Threshold is selected only on validation data.

    validation_probabilities = get_high_probabilities(
        model,
        X_validation,
    )

    threshold_results, selected_threshold = select_threshold(
        y_validation,
        validation_probabilities,
    )

    # Final untouched test evaluation.

    workload_predictions = model.predict(
        X_test
    )

    high_probabilities = get_high_probabilities(
        model,
        X_test,
    )

    risk_alerts = (
        high_probabilities >= selected_threshold
    )

    risk_metrics = calculate_risk_metrics(
        y_test,
        risk_alerts,
    )

    classification_results = classification_metrics(
        y_test,
        workload_predictions,
    )

    logger.info("=" * 70)
    logger.info("TREND-ENHANCED TEST RESULTS")
    logger.info("=" * 70)

    logger.info(
        "Selected threshold: %.2f",
        selected_threshold,
    )

    logger.info(
        "Classification metrics: %s",
        classification_results,
    )

    logger.info(
        "Independent risk metrics: %s",
        risk_metrics,
    )

    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_test,
            workload_predictions,
            labels=CLASS_LABELS,
            zero_division=0,
        ),
    )

    save_results(
        model=model,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        purged_count=purged_count,
        validation_start=validation_start,
        test_start=test_start,
        threshold_results=threshold_results,
        selected_threshold=selected_threshold,
        y_test=y_test,
        workload_predictions=workload_predictions,
        high_probabilities=high_probabilities,
        risk_metrics=risk_metrics,
        classification_results=classification_results,
        feature_columns=feature_columns,
    )

    logger.info("=" * 70)
    logger.info("TREND-ENHANCED EXPERIMENT COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
