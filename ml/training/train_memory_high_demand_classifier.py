
"""
CloudMorph - Memory High-Demand Classifier

Predicts whether future peak memory utilization will reach
85% or higher during the next 30 minutes.

Models:
    1. ExtraTreesClassifier
    2. RandomForestClassifier

Methodology:
    - 55 historical features
    - 60-minute historical window
    - 30-minute prediction horizon
    - Machine-aware 60/20/20 split
    - Validation Average Precision for model selection
    - Validation F1 for threshold selection
    - Independent final test evaluation

Does not modify existing CloudMorph models.
"""

from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

RAW_DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "memory_high_demand_classifier"
)

MODEL_PATH = OUTPUT_DIR / "memory_high_demand_classifier.joblib"

REPORT_PATH = OUTPUT_DIR / "training_report.json"

COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

RANDOM_STATE = 42

N_ESTIMATORS = 300

HIGH_DEMAND_THRESHOLD = 85.0

GROUP_COLUMN = "machineID"

TARGET_COLUMN = "future_high_memory"


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

logger = logging.getLogger("CloudMorphMemoryClassifier")


# ---------------------------------------------------------
# DATA LOADING AND TARGET GENERATION
# ---------------------------------------------------------

def load_dataset():

    logger.info("Loading feature and original datasets...")

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {RAW_DATASET_PATH}"
        )

    features_df = pd.read_csv(FEATURE_DATASET_PATH)

    raw_df = pd.read_csv(RAW_DATASET_PATH)

    if features_df.empty or raw_df.empty:
        raise ValueError("One of the datasets is empty.")

    required_feature_columns = METADATA_COLUMNS

    required_raw_columns = [
        "machineID",
        "step",
        "util_memory",
    ]

    for column in required_feature_columns:
        if column not in features_df.columns:
            raise ValueError(
                f"Missing feature column: {column}"
            )

    for column in required_raw_columns:
        if column not in raw_df.columns:
            raise ValueError(
                f"Missing raw dataset column: {column}"
            )

    # Match the original training pipeline.

    features_df[GROUP_COLUMN] = (
        features_df[GROUP_COLUMN].astype(str)
    )

    raw_df[GROUP_COLUMN] = (
        raw_df[GROUP_COLUMN].astype(str)
    )

    for column in [
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
    ]:

        features_df[column] = pd.to_numeric(
            features_df[column],
            errors="raise",
        ).astype(int)

    raw_df["step"] = pd.to_numeric(
        raw_df["step"],
        errors="raise",
    ).astype(int)

    raw_df["util_memory"] = pd.to_numeric(
        raw_df["util_memory"],
        errors="raise",
    )

    if not np.isfinite(
        raw_df["util_memory"].to_numpy()
    ).all():

        raise ValueError(
            "Original memory utilization contains invalid values."
        )

    if (
        (raw_df["util_memory"] < 0).any()
        or (raw_df["util_memory"] > 100).any()
    ):

        raise ValueError(
            "Memory utilization must be between 0 and 100."
        )

    if raw_df.duplicated(
        subset=["machineID", "step"]
    ).any():

        raise ValueError(
            "Duplicate machineID/step records found."
        )

    logger.info("Building future memory lookup...")

    memory_lookup = {}

    for row in raw_df.itertuples(index=False):

        memory_lookup[
            (row.machineID, row.step)
        ] = float(row.util_memory)

    future_peak_memory = []

    for row in features_df.itertuples(index=False):

        machine_id = row.machineID

        start_step = int(row.target_start_step)

        end_step = int(row.target_end_step)

        expected_steps = list(
            range(start_step, end_step + 1)
        )

        if len(expected_steps) != 6:

            raise ValueError(
                f"Expected 6 future observations for machine "
                f"{machine_id}, steps {start_step}-{end_step}."
            )

        future_values = []

        for step in expected_steps:

            key = (machine_id, step)

            if key not in memory_lookup:

                raise ValueError(
                    f"Missing future memory observation: "
                    f"machine={machine_id}, step={step}"
                )

            future_values.append(memory_lookup[key])

        future_peak_memory.append(
            max(future_values)
        )

    features_df["future_peak_memory"] = (
        future_peak_memory
    )

    features_df[TARGET_COLUMN] = (
        features_df["future_peak_memory"]
        >= HIGH_DEMAND_THRESHOLD
    ).astype(int)

    feature_columns = [
        column
        for column in features_df.columns
        if column not in (
            METADATA_COLUMNS
            + [
                "future_peak_memory",
                TARGET_COLUMN,
            ]
        )
    ]

    if len(feature_columns) != 55:

        raise ValueError(
            f"Expected 55 historical features, "
            f"found {len(feature_columns)}."
        )

    X = features_df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():

        bad_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            f"Invalid historical features: {bad_columns}"
        )

    if not np.isfinite(X.to_numpy()).all():

        raise ValueError(
            "Historical features contain infinite values."
        )

    y = features_df[TARGET_COLUMN].astype(int)

    groups = features_df[GROUP_COLUMN]

    if y.nunique() != 2:

        raise ValueError(
            "Target must contain both positive and negative samples."
        )

    logger.info("Samples: %s", len(features_df))

    logger.info("Machines: %s", groups.nunique())

    logger.info("Features: %s", len(feature_columns))

    logger.info(
        "High-demand samples: %s (%.4f%%)",
        int(y.sum()),
        float(y.mean() * 100),
    )

    logger.info(
        "Normal-demand samples: %s",
        int((y == 0).sum()),
    )

    return (
        features_df,
        X,
        y,
        groups,
        feature_columns,
    )


# ---------------------------------------------------------
# MACHINE-AWARE SPLITTING
# ---------------------------------------------------------

def create_splits(X, y, groups):

    logger.info("Creating machine-aware splits...")

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_validation_idx, test_idx = next(
        outer_splitter.split(
            X,
            y,
            groups,
        )
    )

    X_train_validation = X.iloc[
        train_validation_idx
    ]

    y_train_validation = y.iloc[
        train_validation_idx
    ]

    groups_train_validation = groups.iloc[
        train_validation_idx
    ]

    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE + 1,
    )

    train_relative_idx, validation_relative_idx = next(
        inner_splitter.split(
            X_train_validation,
            y_train_validation,
            groups_train_validation,
        )
    )

    train_idx = train_validation_idx[
        train_relative_idx
    ]

    validation_idx = train_validation_idx[
        validation_relative_idx
    ]

    X_train = X.iloc[train_idx]

    y_train = y.iloc[train_idx]

    X_validation = X.iloc[validation_idx]

    y_validation = y.iloc[validation_idx]

    X_test = X.iloc[test_idx]

    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    groups_validation = groups.iloc[validation_idx]

    groups_test = groups.iloc[test_idx]

    train_machines = set(groups_train)

    validation_machines = set(groups_validation)

    test_machines = set(groups_test)

    if train_machines & validation_machines:
        raise RuntimeError(
            "Machine leakage between training and validation."
        )

    if train_machines & test_machines:
        raise RuntimeError(
            "Machine leakage between training and testing."
        )

    if validation_machines & test_machines:
        raise RuntimeError(
            "Machine leakage between validation and testing."
        )

    for split_name, target in [
        ("training", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:

        positives = int(target.sum())

        logger.info(
            "%s: %s samples | %s positives (%.4f%%)",
            split_name,
            len(target),
            positives,
            target.mean() * 100,
        )

        if positives == 0:
            raise ValueError(
                f"No positive samples in {split_name} split. "
                "Cannot reliably train or evaluate this experiment."
            )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "test_idx": test_idx,
        "train_machines": train_machines,
        "validation_machines": validation_machines,
        "test_machines": test_machines,
    }


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------

def create_models():

    return {
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "predicted_alerts": int(predictions.sum()),
        "actual_high_demand": int(np.sum(y_true)),
    }


# ---------------------------------------------------------
# VALIDATION THRESHOLD SELECTION
# ---------------------------------------------------------

def select_validation_threshold(y_validation, probabilities):

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_validation,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate validation thresholds."
        )

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (
            precision_values[:-1]
            + recall_values[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f1_values))

    selected_threshold = float(
        thresholds[best_index]
    )

    validation_metrics = calculate_metrics(
        y_validation,
        probabilities,
        selected_threshold,
    )

    return selected_threshold, validation_metrics


# ---------------------------------------------------------
# TRAINING AND MODEL COMPARISON
# ---------------------------------------------------------

def train_and_compare(splits):

    X_train = splits["X_train"]

    y_train = splits["y_train"]

    X_validation = splits["X_validation"]

    y_validation = splits["y_validation"]

    X_test = splits["X_test"]

    y_test = splits["y_test"]

    comparison = []

    trained_models = {}

    for model_name, model in create_models().items():

        logger.info("Training %s...", model_name)

        model.fit(
            X_train,
            y_train,
        )

        validation_probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        validation_ap = average_precision_score(
            y_validation,
            validation_probabilities,
        )

        validation_roc_auc = roc_auc_score(
            y_validation,
            validation_probabilities,
        )

        threshold, validation_metrics = (
            select_validation_threshold(
                y_validation,
                validation_probabilities,
            )
        )

        result = {
            "model_name": model_name,
            "validation_average_precision": float(
                validation_ap
            ),
            "validation_roc_auc": float(
                validation_roc_auc
            ),
            "validation_f1": validation_metrics["f1"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_threshold": threshold,
        }

        comparison.append(result)

        trained_models[model_name] = {
            "model": model,
            "validation_average_precision": validation_ap,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
        }

        logger.info(
            "%s | AP=%.5f | ROC-AUC=%.5f | "
            "F1=%.5f | Precision=%.5f | Recall=%.5f | "
            "Threshold=%.5f",
            model_name,
            validation_ap,
            validation_roc_auc,
            validation_metrics["f1"],
            validation_metrics["precision"],
            validation_metrics["recall"],
            threshold,
        )

    # Select by validation Average Precision only.

    best_model_name = max(
        trained_models,
        key=lambda name: (
            trained_models[name]["validation_average_precision"]
        ),
    )

    best_result = trained_models[best_model_name]

    best_model = best_result["model"]

    best_threshold = best_result["threshold"]

    logger.info(
        "Selected model: %s",
        best_model_name,
    )

    # Final evaluation on untouched test data.

    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    test_metrics["average_precision"] = float(
        average_precision_score(
            y_test,
            test_probabilities,
        )
    )

    test_metrics["roc_auc"] = float(
        roc_auc_score(
            y_test,
            test_probabilities,
        )
    )

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    logger.info("Final test metrics: %s", test_metrics)

    logger.info(
        "Test classification report:\n%s",
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        ),
    )

    return (
        best_model_name,
        best_model,
        best_threshold,
        comparison,
        best_result["validation_metrics"],
        test_metrics,
        test_probabilities,
    )


# ---------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------

def save_artifacts(
    features_df,
    feature_columns,
    splits,
    best_model_name,
    best_model,
    best_threshold,
    comparison,
    validation_metrics,
    test_metrics,
    test_probabilities,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save selected model package.

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_definition": (
            "1 if maximum memory utilization across the next "
            "6 observations (30 minutes) is >=85%; otherwise 0."
        ),
        "history_window_minutes": 60,
        "prediction_horizon_minutes": 30,
        "alert_threshold": float(best_threshold),
        "training_samples": len(splits["X_train"]),
        "validation_samples": len(splits["X_validation"]),
        "test_samples": len(splits["X_test"]),
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    logger.info("Model saved: %s", MODEL_PATH)

    # Save model comparison.

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # Save test predictions.

    test_predictions_df = pd.DataFrame({
        "machineID": (
            splits["groups_test"].to_numpy()
        ),
        "actual_high_memory": (
            splits["y_test"].to_numpy()
        ),
        "predicted_high_memory": (
            test_probabilities >= best_threshold
        ).astype(int),
        "predicted_probability": test_probabilities,
        "actual_peak_memory": (
            features_df.iloc[
                splits["test_idx"]
            ]["future_peak_memory"].to_numpy()
        ),
        "alert_threshold": best_threshold,
    })

    test_predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Save training report.

    report = {
        "project": "CloudMorph",
        "task": "Future high memory demand classification",
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "target": {
            "column": TARGET_COLUMN,
            "definition": (
                "Future peak memory >=85% over the next "
                "6 observations."
            ),
            "positive_class": 1,
            "negative_class": 0,
        },
        "data": {
            "total_samples": len(features_df),
            "total_machines": int(
                features_df["machineID"].nunique()
            ),
            "feature_count": len(feature_columns),
            "training_samples": len(splits["X_train"]),
            "validation_samples": len(splits["X_validation"]),
            "test_samples": len(splits["X_test"]),
            "training_positive_samples": int(
                splits["y_train"].sum()
            ),
            "validation_positive_samples": int(
                splits["y_validation"].sum()
            ),
            "test_positive_samples": int(
                splits["y_test"].sum()
            ),
            "training_machines": len(
                splits["train_machines"]
            ),
            "validation_machines": len(
                splits["validation_machines"]
            ),
            "test_machines": len(
                splits["test_machines"]
            ),
        },
        "methodology": {
            "split": "GroupShuffleSplit by machineID",
            "split_proportions": {
                "training": 0.60,
                "validation": 0.20,
                "testing": 0.20,
            },
            "model_selection": (
                "Highest validation Average Precision."
            ),
            "threshold_selection": (
                "Maximum validation F1."
            ),
            "class_imbalance_handling": (
                "Class weighting."
            ),
            "automatic_resource_changes": False,
        },
        "selected_threshold": float(best_threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
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

    logger.info("Training report saved: %s", REPORT_PATH)

    logger.info(
        "Model comparison saved: %s",
        COMPARISON_PATH,
    )

    logger.info(
        "Test predictions saved: %s",
        PREDICTIONS_PATH,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    logger.info("=" * 70)

    logger.info(
        "CLOUDMORPH MEMORY HIGH-DEMAND CLASSIFIER TRAINING"
    )

    logger.info("=" * 70)

    (
        features_df,
        X,
        y,
        groups,
        feature_columns,
    ) = load_dataset()

    splits = create_splits(
        X,
        y,
        groups,
    )

    (
        best_model_name,
        best_model,
        best_threshold,
        comparison,
        validation_metrics,
        test_metrics,
        test_probabilities,
    ) = train_and_compare(splits)

    save_artifacts(
        features_df,
        feature_columns,
        splits,
        best_model_name,
        best_model,
        best_threshold,
        comparison,
        validation_metrics,
        test_metrics,
        test_probabilities,
    )

    logger.info("=" * 70)

    logger.info("TRAINING COMPLETED SUCCESSFULLY")

    logger.info("Selected model: %s", best_model_name)

    logger.info(
        "Selected alert threshold: %.5f",
        best_threshold,
    )

    logger.info(
        "Test Average Precision: %.5f",
        test_metrics["average_precision"],
    )

    logger.info(
        "Test Precision: %.5f",
        test_metrics["precision"],
    )

    logger.info(
        "Test Recall: %.5f",
        test_metrics["recall"],
    )

    logger.info(
        "Test F1: %.5f",
        test_metrics["f1"],
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
