
"""
CloudMorph - Resource Demand Regression Training

Predicts maximum CPU and memory utilization over the next 30 minutes.

Inputs:
    55 historical workload features calculated from the previous 60 minutes.

Targets:
    Maximum observed CPU utilization over the next 30 minutes.
    Maximum observed memory utilization over the next 30 minutes.

Models:
    Random Forest Regressor
    Extra Trees Regressor

Uses machine-aware splitting to prevent machine leakage.
Does not predict monetary cost or claim actual cost savings.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    / "resource_demand_regressor"
)

CPU_MODEL_PATH = OUTPUT_DIR / "cpu_peak_regressor.joblib"
MEMORY_MODEL_PATH = OUTPUT_DIR / "memory_peak_regressor.joblib"

REPORT_PATH = OUTPUT_DIR / "training_report.json"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 200

GROUP_COLUMN = "machineID"

CPU_TARGET = "future_peak_cpu"
MEMORY_TARGET = "future_peak_memory"

METADATA_COLUMNS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
    "target",
]

RAW_REQUIRED_COLUMNS = [
    "machineID",
    "step",
    "util_CPU",
    "util_memory",
]


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphResourceRegressor")


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

    missing_feature_columns = [
        col for col in METADATA_COLUMNS
        if col not in features_df.columns
    ]

    missing_raw_columns = [
        col for col in RAW_REQUIRED_COLUMNS
        if col not in raw_df.columns
    ]

    if missing_feature_columns:
        raise ValueError(
            f"Missing feature metadata: {missing_feature_columns}"
        )

    if missing_raw_columns:
        raise ValueError(
            f"Missing original dataset columns: {missing_raw_columns}"
        )

    # Normalize machine IDs and step values.

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
        values = pd.to_numeric(
            features_df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Invalid values in {column}."
            )

        if not np.allclose(values, np.round(values)):
            raise ValueError(
                f"Non-integer step values in {column}."
            )

        features_df[column] = values.astype(int)

    raw_df["step"] = pd.to_numeric(
        raw_df["step"],
        errors="coerce",
    )

    if raw_df["step"].isna().any():
        raise ValueError("Original dataset contains invalid steps.")

    if not np.allclose(
        raw_df["step"],
        np.round(raw_df["step"]),
    ):
        raise ValueError("Original dataset has non-integer steps.")

    raw_df["step"] = raw_df["step"].astype(int)

    for column in ["util_CPU", "util_memory"]:
        raw_df[column] = pd.to_numeric(
            raw_df[column],
            errors="coerce",
        )

        if raw_df[column].isna().any():
            raise ValueError(
                f"Invalid values in original {column}."
            )

        if not np.isfinite(raw_df[column].to_numpy()).all():
            raise ValueError(
                f"Infinite values found in {column}."
            )

        if (
            (raw_df[column] < 0).any()
            or (raw_df[column] > 100).any()
        ):
            raise ValueError(
                f"{column} must be between 0 and 100."
            )

    if raw_df.duplicated(
        subset=[GROUP_COLUMN, "step"]
    ).any():
        raise ValueError(
            "Duplicate machineID/step records found."
        )

    # Build direct machine-step lookup.

    logger.info("Building future workload lookup...")

    lookup = {}

    for row in raw_df.itertuples(index=False):

        machine_id = getattr(row, GROUP_COLUMN)
        step = getattr(row, "step")

        lookup[(machine_id, step)] = (
            float(getattr(row, "util_CPU")),
            float(getattr(row, "util_memory")),
        )

    # Generate regression targets using future observations.

    cpu_targets = []
    memory_targets = []

    logger.info("Generating future CPU and memory targets...")

    for row in features_df.itertuples(index=False):

        machine_id = getattr(row, GROUP_COLUMN)

        start_step = int(
            getattr(row, "target_start_step")
        )

        end_step = int(
            getattr(row, "target_end_step")
        )

        expected_steps = list(
            range(start_step, end_step + 1)
        )

        if len(expected_steps) != 6:
            raise ValueError(
                "Expected exactly 6 future observations "
                f"for machine {machine_id}, "
                f"steps {start_step}-{end_step}."
            )

        future_records = []

        for step in expected_steps:

            key = (machine_id, step)

            if key not in lookup:
                raise ValueError(
                    "Missing future observation for "
                    f"machine {machine_id}, step {step}."
                )

            future_records.append(lookup[key])

        cpu_targets.append(
            max(record[0] for record in future_records)
        )

        memory_targets.append(
            max(record[1] for record in future_records)
        )

    features_df[CPU_TARGET] = cpu_targets
    features_df[MEMORY_TARGET] = memory_targets

    # Keep only historical features.
    # Exclude classification labels and all metadata.

    feature_columns = [
        column
        for column in features_df.columns
        if column not in (
            METADATA_COLUMNS
            + [CPU_TARGET, MEMORY_TARGET]
        )
    ]

    if len(feature_columns) != 55:
        raise ValueError(
            "Expected 55 historical features, "
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
            f"Missing or non-numeric features: {bad_columns}"
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain infinite values.")

    y_cpu = features_df[CPU_TARGET].astype(float)
    y_memory = features_df[MEMORY_TARGET].astype(float)

    groups = features_df[GROUP_COLUMN]

    if not np.isfinite(y_cpu.to_numpy()).all():
        raise ValueError("CPU targets contain invalid values.")

    if not np.isfinite(y_memory.to_numpy()).all():
        raise ValueError("Memory targets contain invalid values.")

    logger.info("Samples: %s", len(features_df))
    logger.info("Machines: %s", groups.nunique())
    logger.info("Features: %s", len(feature_columns))

    logger.info(
        "CPU target range: %.2f - %.2f",
        y_cpu.min(),
        y_cpu.max(),
    )

    logger.info(
        "Memory target range: %.2f - %.2f",
        y_memory.min(),
        y_memory.max(),
    )

    return (
        features_df,
        X,
        y_cpu,
        y_memory,
        groups,
        feature_columns,
    )


# ---------------------------------------------------------
# MACHINE-AWARE SPLITTING
# ---------------------------------------------------------

def create_splits(X, y, groups):

    logger.info("Creating machine-aware splits...")

    # 80% train + validation, 20% test.

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_validation_idx, test_idx = next(
        outer_splitter.split(X, y, groups)
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

    # 75% train, 25% validation of train-validation.
    # Overall approximately 60/20/20.

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

    logger.info(
        "Training: %s samples | %s machines",
        len(X_train),
        len(train_machines),
    )

    logger.info(
        "Validation: %s samples | %s machines",
        len(X_validation),
        len(validation_machines),
    )

    logger.info(
        "Testing: %s samples | %s machines",
        len(X_test),
        len(test_machines),
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "X_train_validation": X_train_validation,
        "y_train_validation": y_train_validation,
        "train_machines": train_machines,
        "validation_machines": validation_machines,
        "test_machines": test_machines,
    }


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------

def create_models():

    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    # Utilization is physically bounded between 0 and 100.
    # Clip predictions for reporting and downstream use.

    predictions = np.clip(
        predictions,
        0.0,
        100.0,
    )

    metrics = {
        "mae": float(
            mean_absolute_error(y, predictions)
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(y, predictions)
            )
        ),

        "r2": float(
            r2_score(y, predictions)
        ),
    }

    return metrics, predictions


# ---------------------------------------------------------
# TRAIN ONE TARGET
# ---------------------------------------------------------

def train_target(
    target_name,
    target_column,
    X,
    y,
    groups,
    feature_columns,
):

    logger.info("=" * 65)
    logger.info("TRAINING TARGET: %s", target_name)
    logger.info("=" * 65)

    splits = create_splits(X, y, groups)

    X_train = splits["X_train"]
    y_train = splits["y_train"]

    X_validation = splits["X_validation"]
    y_validation = splits["y_validation"]

    X_test = splits["X_test"]
    y_test = splits["y_test"]

    comparison_results = []
    trained_models = {}

    # Train and compare candidate models.

    for model_name, model in create_models().items():

        logger.info("Training %s for %s...", model_name, target_name)

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
            "%s validation metrics: %s",
            model_name,
            metrics,
        )

        comparison_results.append({
            "target": target_name,
            "model": model_name,
            **metrics,
        })

        trained_models[model_name] = model

    comparison_df = pd.DataFrame(
        comparison_results
    )

    comparison_df = comparison_df.sort_values(
        by=["mae", "rmse"],
        ascending=True,
    )

    best_model_name = comparison_df.iloc[0]["model"]

    logger.info(
        "Selected %s model: %s",
        target_name,
        best_model_name,
    )

    best_model = trained_models[best_model_name]

    # Refit using training + validation.

    best_model.fit(
        splits["X_train_validation"],
        splits["y_train_validation"],
    )

    # Final test evaluation on unseen machines.

    test_metrics, test_predictions = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    logger.info(
        "%s final test metrics: %s",
        target_name,
        test_metrics,
    )

    prediction_df = pd.DataFrame({
        "machineID": splits["groups_test"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": test_predictions,
        "absolute_error": np.abs(
            y_test.to_numpy() - test_predictions
        ),
    })

    # Package model with its inference schema.

    model_package = {
        "model": best_model,
        "feature_columns": feature_columns,
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "target": target_column,
        "target_description": (
            f"Maximum observed {target_name} utilization "
            "over the next 30 minutes."
        ),
        "training_samples": int(
            len(splits["X_train_validation"])
        ),
        "feature_count": len(feature_columns),
        "metadata": {
            "dataset": str(FEATURE_DATASET_PATH),
            "target_source": str(RAW_DATASET_PATH),
            "input_history_minutes": 60,
            "prediction_horizon_minutes": 30,
            "future_observations": 6,
            "training_method": "machine_aware_group_split",
            "random_state": RANDOM_STATE,
        },
    }

    model_path = (
        CPU_MODEL_PATH
        if target_name == "cpu"
        else MEMORY_MODEL_PATH
    )

    joblib.dump(
        model_package,
        model_path,
    )

    return {
        "target": target_name,
        "selected_model": best_model_name,
        "test_metrics": test_metrics,
        "validation_comparison": comparison_results,
        "model_path": str(model_path),
        "predictions": prediction_df,
        "split": {
            "training_samples": len(splits["X_train"]),
            "validation_samples": len(splits["X_validation"]),
            "test_samples": len(splits["X_test"]),
            "training_machines": len(splits["train_machines"]),
            "validation_machines": len(splits["validation_machines"]),
            "test_machines": len(splits["test_machines"]),
        },
    }


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH RESOURCE DEMAND REGRESSION")
    logger.info("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        X,
        y_cpu,
        y_memory,
        groups,
        feature_columns,
    ) = load_dataset()

    cpu_result = train_target(
        "cpu",
        CPU_TARGET,
        X,
        y_cpu,
        groups,
        feature_columns,
    )

    memory_result = train_target(
        "memory",
        MEMORY_TARGET,
        X,
        y_memory,
        groups,
        feature_columns,
    )

    # Combined model comparison.

    comparison_df = pd.DataFrame(
        cpu_result["validation_comparison"]
        + memory_result["validation_comparison"]
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # Combined test predictions.

    cpu_predictions = cpu_result["predictions"].rename(
        columns={
            "actual": "actual_peak_cpu",
            "predicted": "predicted_peak_cpu",
            "absolute_error": "cpu_absolute_error",
        }
    )

    memory_predictions = memory_result["predictions"].rename(
        columns={
            "actual": "actual_peak_memory",
            "predicted": "predicted_peak_memory",
            "absolute_error": "memory_absolute_error",
        }
    )

    if not cpu_predictions["machineID"].equals(
        memory_predictions["machineID"]
    ):
        raise RuntimeError(
            "CPU and memory test machine ordering differs."
        )

    predictions_df = pd.concat(
        [
            cpu_predictions.reset_index(drop=True),
            memory_predictions[
                [
                    "actual_peak_memory",
                    "predicted_peak_memory",
                    "memory_absolute_error",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Training report.

    report = {
        "project": "CloudMorph",
        "task": "Future peak resource demand regression",
        "model_version": "1.0.0",
        "input_history_minutes": 60,
        "prediction_horizon_minutes": 30,
        "future_observations": 6,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,

        "dataset": {
            "total_samples": int(len(df)),
            "unique_machines": int(groups.nunique()),
            "target_source": str(RAW_DATASET_PATH),
            "feature_source": str(FEATURE_DATASET_PATH),
        },

        "targets": {
            "cpu": {
                "target_column": CPU_TARGET,
                "description": (
                    "Maximum observed CPU utilization "
                    "in the next 30 minutes."
                ),
                "selected_model": cpu_result["selected_model"],
                "test_metrics": cpu_result["test_metrics"],
                "model_path": cpu_result["model_path"],
                "split": cpu_result["split"],
                "validation_comparison": (
                    cpu_result["validation_comparison"]
                ),
            },

            "memory": {
                "target_column": MEMORY_TARGET,
                "description": (
                    "Maximum observed memory utilization "
                    "in the next 30 minutes."
                ),
                "selected_model": memory_result["selected_model"],
                "test_metrics": memory_result["test_metrics"],
                "model_path": memory_result["model_path"],
                "split": memory_result["split"],
                "validation_comparison": (
                    memory_result["validation_comparison"]
                ),
            },
        },

        "methodology": {
            "split": "GroupShuffleSplit by machineID",
            "split_proportions": {
                "training": 0.60,
                "validation": 0.20,
                "testing": 0.20,
            },
            "model_selection": (
                "Lowest validation MAE, followed by validation RMSE."
            ),
            "evaluation_metrics": [
                "MAE",
                "RMSE",
                "R2",
            ],
        },

        "important_notes": [
            "Targets are derived from future observed workload.",
            "Historical features only are used as model inputs.",
            "Machine-aware splitting prevents machine overlap.",
            "The model predicts resource demand, not cloud cost.",
            "Predictions are not verified cloud resizing outcomes.",
            "No cloud resources are modified by this model.",
            "Manual USD cost calculations remain separate.",
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

    logger.info("=" * 65)
    logger.info("RESOURCE REGRESSION TRAINING COMPLETED")
    logger.info("=" * 65)

    logger.info(
        "CPU model: %s | MAE: %.4f | RMSE: %.4f | R2: %.4f",
        cpu_result["selected_model"],
        cpu_result["test_metrics"]["mae"],
        cpu_result["test_metrics"]["rmse"],
        cpu_result["test_metrics"]["r2"],
    )

    logger.info(
        "Memory model: %s | MAE: %.4f | RMSE: %.4f | R2: %.4f",
        memory_result["selected_model"],
        memory_result["test_metrics"]["mae"],
        memory_result["test_metrics"]["rmse"],
        memory_result["test_metrics"]["r2"],
    )

    logger.info("CPU model saved: %s", CPU_MODEL_PATH)
    logger.info("Memory model saved: %s", MEMORY_MODEL_PATH)
    logger.info("Report saved: %s", REPORT_PATH)
    logger.info("Comparison saved: %s", COMPARISON_PATH)
    logger.info("Predictions saved: %s", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
