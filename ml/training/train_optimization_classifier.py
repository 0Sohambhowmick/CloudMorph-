
"""
CloudMorph - Optimization Workload Classifier Training

Predicts future workload conditions:
    LOW_DEMAND
    NORMAL_DEMAND
    HIGH_DEMAND

Uses machine-aware stratified splitting to prevent machine leakage.
Compares Random Forest and Extra Trees.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
)

MODEL_PATH = OUTPUT_DIR / "optimization_classifier.joblib"
REPORT_PATH = OUTPUT_DIR / "training_report.json"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

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


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphClassifier")


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_dataset():

    logger.info("Loading optimization training dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    logger.info("Dataset samples: %s", len(df))

    required_columns = METADATA_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    if df[GROUP_COLUMN].isna().any():
        raise ValueError("machineID contains missing values.")

    unexpected_labels = (
        set(df[TARGET_COLUMN].unique()) - set(CLASS_LABELS)
    )

    if unexpected_labels:
        raise ValueError(
            f"Unexpected target labels: {unexpected_labels}"
        )

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    # Ensure all features are numeric.
    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()

        raise ValueError(
            "Features contain missing or non-numeric values: "
            f"{bad_columns}"
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain infinite values.")

    y = df[TARGET_COLUMN].astype(str)
    groups = df[GROUP_COLUMN]

    logger.info("Features: %s", len(feature_columns))
    logger.info("Unique machines: %s", groups.nunique())

    logger.info(
        "Class distribution:\n%s",
        y.value_counts().to_string(),
    )

    return df, X, y, groups, feature_columns


# ---------------------------------------------------------
# MACHINE-AWARE TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------

def create_splits(X, y, groups):

    logger.info(
        "Creating machine-aware stratified splits..."
    )

    # First split:
    # 80% train + validation
    # 20% test

    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    try:
        train_validation_idx, test_idx = next(
            outer_splitter.split(X, y, groups)
        )

    except ValueError as exc:
        raise ValueError(
            "Could not create the machine-aware test split. "
            "Check whether all classes occur across enough "
            "different machines."
        ) from exc

    X_train_validation = X.iloc[train_validation_idx]
    y_train_validation = y.iloc[train_validation_idx]
    groups_train_validation = groups.iloc[train_validation_idx]

    # Second split:
    # Divide train + validation into approximately
    # 80% train and 20% validation.

    inner_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    try:
        train_relative_idx, validation_relative_idx = next(
            inner_splitter.split(
                X_train_validation,
                y_train_validation,
                groups_train_validation,
            )
        )

    except ValueError as exc:
        raise ValueError(
            "Could not create the machine-aware validation split. "
            "Check the distribution of classes across machines."
        ) from exc

    train_idx = train_validation_idx[train_relative_idx]
    validation_idx = train_validation_idx[validation_relative_idx]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_validation = X.iloc[validation_idx]
    y_validation = y.iloc[validation_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_validation = groups.iloc[validation_idx]
    groups_test = groups.iloc[test_idx]

    # Confirm no machine appears in multiple splits.

    train_machines = set(groups_train)
    validation_machines = set(groups_validation)
    test_machines = set(groups_test)

    if train_machines & validation_machines:
        raise RuntimeError("Machine leakage between train and validation.")

    if train_machines & test_machines:
        raise RuntimeError("Machine leakage between train and test.")

    if validation_machines & test_machines:
        raise RuntimeError("Machine leakage between validation and test.")

    # Every class must be represented in every split.

    for split_name, labels in [
        ("Training", y_train),
        ("Validation", y_validation),
        ("Testing", y_test),
    ]:

        missing = set(CLASS_LABELS) - set(labels.unique())

        if missing:
            raise ValueError(
                f"{split_name} split is missing classes: {missing}. "
                "Do not use a random row split to work around this. "
                "We need to adjust the machine-aware splitting strategy."
            )

    logger.info(
        "Training samples: %s | Machines: %s",
        len(X_train),
        len(train_machines),
    )

    logger.info(
        "Validation samples: %s | Machines: %s",
        len(X_validation),
        len(validation_machines),
    )

    logger.info(
        "Test samples: %s | Machines: %s",
        len(X_test),
        len(test_machines),
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        groups_test,
        X_train_validation,
        y_train_validation,
    )


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------

def create_models():

    logger.info("Initializing ML classifiers...")

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

    }

    return models


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    high_demand_recall = recall_score(
        y,
        predictions,
        labels=["HIGH_DEMAND"],
        average=None,
        zero_division=0,
    )[0]

    metrics = {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),

        "high_demand_recall": float(high_demand_recall),
    }

    return metrics, predictions


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH OPTIMIZATION CLASSIFIER TRAINING")
    logger.info("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        X,
        y,
        groups,
        feature_columns,
    ) = load_dataset()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        groups_test,
        X_train_validation,
        y_train_validation,
    ) = create_splits(X, y, groups)

    models = create_models()

    comparison_results = []
    trained_models = {}

    # -----------------------------------------------------
    # TRAIN AND VALIDATE BOTH MODELS
    # -----------------------------------------------------

    for model_name, model in models.items():

        logger.info("-" * 55)
        logger.info("Training %s...", model_name)

        model.fit(
            X_train,
            y_train,
        )

        metrics, _ = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        logger.info(
            "%s validation results: %s",
            model_name,
            metrics,
        )

        comparison_results.append({
            "model": model_name,
            **metrics,
        })

        trained_models[model_name] = model

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df = comparison_df.sort_values(
        by=[
            "macro_f1",
            "high_demand_recall",
            "balanced_accuracy",
        ],
        ascending=False,
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # SELECT BEST MODEL USING VALIDATION RESULTS
    # -----------------------------------------------------

    best_model_name = comparison_df.iloc[0]["model"]

    logger.info(
        "Selected model based on validation: %s",
        best_model_name,
    )

    best_model = trained_models[best_model_name]

    # -----------------------------------------------------
    # REFIT SELECTED MODEL ON TRAIN + VALIDATION
    # -----------------------------------------------------

    logger.info(
        "Refitting selected model on training + validation data..."
    )

    best_model.fit(
        X_train_validation,
        y_train_validation,
    )

    # -----------------------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------------------

    logger.info("Running final test evaluation...")

    test_metrics, test_predictions = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    logger.info("Final test metrics: %s", test_metrics)

    test_report = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    logger.info(
        "Test classification report:\n%s",
        classification_report(
            y_test,
            test_predictions,
            labels=CLASS_LABELS,
            zero_division=0,
        ),
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=CLASS_LABELS,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )

    confusion_df.index.name = "Actual"

    confusion_df.to_csv(CONFUSION_PATH)

    # -----------------------------------------------------
    # SAVE TEST PREDICTIONS
    # -----------------------------------------------------

    prediction_df = pd.DataFrame({
        "machineID": groups_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": test_predictions,
    })

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE TRAINED MODEL
    # -----------------------------------------------------

    model_package = {
        "model": best_model,
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS,
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "target": TARGET_COLUMN,
        "training_samples": int(len(X_train_validation)),
        "feature_count": int(len(feature_columns)),
        "metadata": {
            "dataset": str(DATASET_PATH),
            "input_history_minutes": 60,
            "prediction_horizon_minutes": 30,
            "training_method": "machine_aware_stratified_group_split",
            "random_state": RANDOM_STATE,
        },
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    # -----------------------------------------------------
    # SAVE TRAINING REPORT
    # -----------------------------------------------------

    report = {
        "project": "CloudMorph",
        "task": "Future workload condition classification",
        "selected_model": best_model_name,
        "model_version": "1.0.0",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS,

        "dataset": {
            "total_samples": int(len(df)),
            "unique_machines": int(groups.nunique()),
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
        },

        "split": {
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_validation)),
            "test_samples": int(len(X_test)),
            "training_machines": int(groups.iloc[X_train.index].nunique()),
            "validation_machines": int(
                groups.iloc[X_validation.index].nunique()
            ),
            "test_machines": int(groups_test.nunique()),
        },

        "validation_comparison": comparison_results,
        "test_metrics": test_metrics,
        "test_classification_report": test_report,

        "model_selection": (
            "Highest validation macro F1, followed by "
            "HIGH_DEMAND recall and balanced accuracy."
        ),

        "important_notes": [
            "Labels represent derived future workload conditions.",
            "Labels are not actual cloud scaling decisions.",
            "No cloud resources are modified by this model.",
            "Evaluation uses machine-aware splits.",
            "Cost estimation is handled separately.",
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

    # -----------------------------------------------------
    # COMPLETION
    # -----------------------------------------------------

    logger.info("=" * 65)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)

    logger.info("Selected model: %s", best_model_name)
    logger.info("Test macro F1: %.4f", test_metrics["macro_f1"])
    logger.info(
        "Test balanced accuracy: %.4f",
        test_metrics["balanced_accuracy"],
    )
    logger.info(
        "HIGH_DEMAND recall: %.4f",
        test_metrics["high_demand_recall"],
    )

    logger.info("Saved model: %s", MODEL_PATH)
    logger.info("Saved report: %s", REPORT_PATH)
    logger.info("Saved comparison: %s", COMPARISON_PATH)
    logger.info("Saved confusion matrix: %s", CONFUSION_PATH)
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
