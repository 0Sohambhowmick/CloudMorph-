
"""
CloudMorph - Temporal Validation and Robustness Evaluation

Evaluates future workload classification using a chronological split.

Important:
- Uses the original Extra Trees configuration.
- Trains only on earlier workload windows.
- Purges overlapping training target windows.
- Tests on later windows from the same machines.
- Does not modify or overwrite the production model.
- Saves results in a separate temporal_evaluation directory.
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

OUTPUT_DIR = MODEL_DIR / "temporal_evaluation"

MODEL_PACKAGE_PATH = (
    MODEL_DIR / "optimization_classifier.joblib"
)

TRAINING_REPORT_PATH = (
    MODEL_DIR / "training_report.json"
)

REPORT_PATH = OUTPUT_DIR / "temporal_evaluation_report.json"
METRICS_PATH = OUTPUT_DIR / "temporal_metrics.csv"
CONFUSION_PATH = OUTPUT_DIR / "temporal_confusion_matrix.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "temporal_predictions.csv"

RANDOM_STATE = 42

N_ESTIMATORS = 300

TEST_START_FRACTION = 0.70

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

logger = logging.getLogger("CloudMorphTemporalEvaluation")


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_dataset():

    logger.info("Loading optimization dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        col for col in METADATA_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[METADATA_COLUMNS].isna().any().any():
        raise ValueError(
            "Dataset metadata contains missing values."
        )

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
            "Feature data contains missing or non-numeric values."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Feature data contains infinite values."
        )

    y = df[TARGET_COLUMN].astype(str)

    logger.info("Dataset samples: %s", len(df))
    logger.info("Unique machines: %s", df[GROUP_COLUMN].nunique())
    logger.info("Feature count: %s", len(feature_columns))

    return df, X, y, feature_columns


# ---------------------------------------------------------
# TEMPORAL DATA VALIDATION
# ---------------------------------------------------------

def validate_temporal_structure(df):

    logger.info("Validating chronological dataset structure...")

    # Each machine must have the same chronological windows.

    machine_steps = (
        df.groupby(GROUP_COLUMN)["history_start_step"]
        .apply(lambda values: tuple(sorted(values.tolist())))
    )

    reference_steps = machine_steps.iloc[0]

    if not machine_steps.apply(
        lambda steps: steps == reference_steps
    ).all():

        raise ValueError(
            "Machines do not have identical chronological coverage."
        )

    if df.duplicated(
        subset=[GROUP_COLUMN, "history_start_step"]
    ).any():

        raise ValueError(
            "Duplicate machine/time-window combinations detected."
        )

    # Validate the expected history and prediction windows.

    if not (
        df["history_end_step"]
        == df["history_start_step"] + 11
    ).all():

        raise ValueError(
            "Unexpected historical window length."
        )

    if not (
        df["target_start_step"]
        == df["history_end_step"] + 1
    ).all():

        raise ValueError(
            "Unexpected target start step."
        )

    if not (
        df["target_end_step"]
        == df["target_start_step"] + 5
    ).all():

        raise ValueError(
            "Unexpected target window length."
        )

    logger.info(
        "Chronological structure verified across %s machines.",
        len(machine_steps),
    )


# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT WITH PURGING
# ---------------------------------------------------------

def create_temporal_split(df, X, y):

    logger.info("Creating chronological train/test split...")

    unique_steps = np.sort(
        df["history_start_step"].unique()
    )

    if len(unique_steps) < 10:
        raise ValueError(
            "Insufficient chronological windows."
        )

    boundary_index = math.ceil(
        len(unique_steps) * TEST_START_FRACTION
    )

    if boundary_index >= len(unique_steps):
        raise ValueError(
            "Invalid temporal test boundary."
        )

    test_start_history_step = unique_steps[boundary_index]

    # Training:
    # Earlier history windows whose future target
    # ends before the first test history window.

    train_mask = (
        (df["history_start_step"] < test_start_history_step)
        &
        (df["target_end_step"] < test_start_history_step)
    )

    # Testing:
    # Later history windows beginning at or after
    # the chronological boundary.

    test_mask = (
        df["history_start_step"] >= test_start_history_step
    )

    # Purged:
    # Earlier samples whose target windows overlap
    # the beginning of the test period.

    purged_mask = ~(train_mask | test_mask)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    purged_df = df.loc[purged_mask].copy()

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()

    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced an empty training or test set."
        )

    if purged_df.empty:
        logger.warning(
            "No samples were purged. Check temporal window boundaries."
        )

    # Verify strict chronological separation.

    max_training_target_end = float(
        train_df["target_end_step"].max()
    )

    min_test_history_start = float(
        test_df["history_start_step"].min()
    )

    if max_training_target_end >= min_test_history_start:
        raise RuntimeError(
            "Temporal overlap detected between training targets "
            "and test history windows."
        )

    # All classes must occur in both splits.

    for split_name, labels in [
        ("Training", y_train),
        ("Testing", y_test),
    ]:

        missing = set(CLASS_LABELS) - set(labels.unique())

        if missing:
            raise ValueError(
                f"{split_name} split is missing classes: {missing}"
            )

    # Confirm all machines appear in both periods.
    # This is intentional: the experiment measures
    # future-time generalization on known machines.

    train_machines = set(train_df[GROUP_COLUMN].unique())
    test_machines = set(test_df[GROUP_COLUMN].unique())

    if train_machines != test_machines:
        raise RuntimeError(
            "Expected identical machine coverage in both periods."
        )

    logger.info("=" * 60)
    logger.info("TEMPORAL SPLIT SUMMARY")
    logger.info("=" * 60)

    logger.info(
        "Training samples: %s",
        len(train_df),
    )

    logger.info(
        "Testing samples: %s",
        len(test_df),
    )

    logger.info(
        "Purged samples: %s",
        len(purged_df),
    )

    logger.info(
        "Machines in each period: %s",
        len(train_machines),
    )

    logger.info(
        "Test starts at history step: %s",
        test_start_history_step,
    )

    logger.info(
        "Maximum training target end: %s",
        max_training_target_end,
    )

    logger.info(
        "Minimum test history start: %s",
        min_test_history_start,
    )

    logger.info(
        "Training class distribution:\n%s",
        y_train.value_counts().to_string(),
    )

    logger.info(
        "Testing class distribution:\n%s",
        y_test.value_counts().to_string(),
    )

    return (
        train_df,
        test_df,
        purged_df,
        X_train,
        y_train,
        X_test,
        y_test,
        test_start_history_step,
    )


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

def create_model():

    logger.info("Initializing Extra Trees classifier...")

    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):

    logger.info("Generating temporal test predictions...")

    predictions = model.predict(X_test)

    high_demand_recall = recall_score(
        y_test,
        predictions,
        labels=["HIGH_DEMAND"],
        average=None,
        zero_division=0,
    )[0]

    metrics = {
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

        "high_demand_recall": float(
            high_demand_recall
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS,
    )

    return metrics, predictions, report, matrix


# ---------------------------------------------------------
# SAVE EVALUATION RESULTS
# ---------------------------------------------------------

def save_results(
    train_df,
    test_df,
    purged_df,
    y_test,
    predictions,
    metrics,
    report,
    matrix,
    feature_columns,
    test_start_history_step,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Confusion matrix.

    confusion_df = pd.DataFrame(
        matrix,
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )

    confusion_df.index.name = "Actual"

    confusion_df.to_csv(CONFUSION_PATH)

    # Test predictions.

    prediction_df = test_df[
        [
            "machineID",
            "history_start_step",
            "history_end_step",
            "target_start_step",
            "target_end_step",
        ]
    ].copy()

    prediction_df["actual"] = y_test.to_numpy()
    prediction_df["predicted"] = predictions

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Metrics CSV.

    metrics_df = pd.DataFrame([
        {
            "model": "Temporal Extra Trees",
            **metrics,
        }
    ])

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    # Reference machine-aware test metrics.
    # These are contextual, not a direct comparison:
    # the evaluation splits test different conditions.

    reference_metrics = None

    if TRAINING_REPORT_PATH.exists():

        with open(
            TRAINING_REPORT_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            training_report = json.load(file)

        reference_metrics = training_report.get(
            "test_metrics"
        )

    # Full JSON report.

    evaluation_report = {
        "project": "CloudMorph",
        "evaluation": "Chronological temporal validation",
        "model": "Extra Trees",
        "model_version": "1.0.0",

        "model_configuration": {
            "n_estimators": N_ESTIMATORS,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },

        "dataset": {
            "total_samples": int(
                len(train_df) + len(test_df) + len(purged_df)
            ),
            "training_samples": int(len(train_df)),
            "testing_samples": int(len(test_df)),
            "purged_samples": int(len(purged_df)),
            "unique_machines": int(
                train_df["machineID"].nunique()
            ),
            "feature_count": int(len(feature_columns)),
        },

        "temporal_split": {
            "test_start_history_step": float(
                test_start_history_step
            ),
            "training_first_history_step": float(
                train_df["history_start_step"].min()
            ),
            "training_last_history_step": float(
                train_df["history_start_step"].max()
            ),
            "training_last_target_end_step": float(
                train_df["target_end_step"].max()
            ),
            "testing_first_history_step": float(
                test_df["history_start_step"].min()
            ),
            "testing_last_history_step": float(
                test_df["history_start_step"].max()
            ),
            "testing_first_target_start_step": float(
                test_df["target_start_step"].min()
            ),
            "testing_last_target_end_step": float(
                test_df["target_end_step"].max()
            ),
            "purging_applied": True,
            "target_history_overlap": False,
            "same_machines_in_both_periods": True,
        },

        "training_class_distribution": {
            str(key): int(value)
            for key, value in
            train_df["target"].value_counts().items()
        },

        "testing_class_distribution": {
            str(key): int(value)
            for key, value in
            test_df["target"].value_counts().items()
        },

        "temporal_metrics": metrics,

        "classification_report": report,

        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": matrix.tolist(),
        },

        "machine_aware_random_test_reference": reference_metrics,

        "important_notes": [
            "The production classifier was not modified.",
            "The temporal model was trained only on earlier windows.",
            "Training target windows were purged to avoid overlap "
            "with test history windows.",
            "The same machines appear in training and testing.",
            "This experiment measures future-time generalization "
            "on known machines.",
            "The earlier machine-aware experiment measures "
            "generalization to unseen machines.",
            "The two evaluation results are not directly comparable "
            "because their split strategies differ.",
            "Dataset windows are overlapping sliding windows.",
            "Workload labels represent derived future utilization "
            "conditions, not actual cloud scaling decisions.",
            "The dataset covers a limited historical trace and "
            "does not establish long-term production performance.",
            "No cloud resources are modified by this evaluation.",
        ],
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation_report,
            file,
            indent=4,
        )

    logger.info("Saved temporal report: %s", REPORT_PATH)
    logger.info("Saved temporal metrics: %s", METRICS_PATH)
    logger.info("Saved confusion matrix: %s", CONFUSION_PATH)
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH TEMPORAL CLASSIFIER EVALUATION")
    logger.info("=" * 65)

    (
        df,
        X,
        y,
        feature_columns,
    ) = load_dataset()

    validate_temporal_structure(df)

    (
        train_df,
        test_df,
        purged_df,
        X_train,
        y_train,
        X_test,
        y_test,
        test_start_history_step,
    ) = create_temporal_split(
        df,
        X,
        y,
    )

    # Verify production feature schema without
    # modifying or using the production model.

    if MODEL_PACKAGE_PATH.exists():

        package = joblib.load(MODEL_PACKAGE_PATH)

        saved_features = package.get("feature_columns")

        if saved_features != feature_columns:
            raise ValueError(
                "Dataset feature columns do not match "
                "the production model feature schema."
            )

        logger.info(
            "Production model feature schema verified."
        )

    # Train a separate temporal experiment model.

    model = create_model()

    logger.info(
        "Training Extra Trees using earlier workload windows..."
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Temporal model training completed."
    )

    metrics, predictions, report, matrix = evaluate_model(
        model,
        X_test,
        y_test,
    )

    logger.info("=" * 65)
    logger.info("TEMPORAL TEST RESULTS")
    logger.info("=" * 65)

    logger.info(
        "Accuracy: %.4f",
        metrics["accuracy"],
    )

    logger.info(
        "Balanced accuracy: %.4f",
        metrics["balanced_accuracy"],
    )

    logger.info(
        "Macro F1: %.4f",
        metrics["macro_f1"],
    )

    logger.info(
        "HIGH_DEMAND recall: %.4f",
        metrics["high_demand_recall"],
    )

    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_test,
            predictions,
            labels=CLASS_LABELS,
            zero_division=0,
        ),
    )

    save_results(
        train_df=train_df,
        test_df=test_df,
        purged_df=purged_df,
        y_test=y_test,
        predictions=predictions,
        metrics=metrics,
        report=report,
        matrix=matrix,
        feature_columns=feature_columns,
        test_start_history_step=test_start_history_step,
    )

    logger.info("=" * 65)
    logger.info("TEMPORAL EVALUATION COMPLETED")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
