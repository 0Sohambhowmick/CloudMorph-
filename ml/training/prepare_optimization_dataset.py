
"""
CloudMorph - Optimization Dataset Preparation

Purpose:
    Prepare a supervised ML dataset for predicting future
    workload conditions using historical server metrics.

Input:
    datasets/processed/server_usage_clean.csv

Historical input:
    12 observations (60 minutes)

Prediction horizon:
    6 observations (30 minutes)

Target classes:
    LOW_DEMAND
    NORMAL_DEMAND
    HIGH_DEMAND

Important:
    Future observations are used ONLY to generate labels.
    They are never included in model input features.
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "data"

OUTPUT_FILE = OUTPUT_DIR / "optimization_training_dataset.csv"

SUMMARY_FILE = OUTPUT_DIR / "optimization_dataset_summary.json"


FEATURE_COLUMNS = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

UTILIZATION_COLUMNS = [
    "util_CPU",
    "util_memory",
]

WINDOW_SIZE = 12
FORECAST_HORIZON = 6

UNDERUTILIZATION_THRESHOLD = 35.0
HIGH_UTILIZATION_THRESHOLD = 85.0

TARGET_CLASSES = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphDataset")


# ============================================================
# DATA VALIDATION
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    logger.info("Loading dataset: %s", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "machineID",
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if "step" not in df.columns and "timestamp" not in df.columns:
        raise ValueError(
            "Dataset must contain either 'step' or 'timestamp'."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Convert workload features to numeric values.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing or invalid values found in required columns. "
            "Clean the source dataset before training."
        )

    if not np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in workload features."
        )

    # Utilization columns must be percentages.
    for column in UTILIZATION_COLUMNS:
        invalid = (
            (df[column] < 0)
            | (df[column] > 100)
        )

        if invalid.any():
            raise ValueError(
                f"{column} contains values outside 0-100."
            )

    # Load averages cannot be negative.
    for column in ["load1", "load5", "load15"]:
        if (df[column] < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )

    if df["machineID"].isnull().any():
        raise ValueError(
            "Missing machine identifiers found."
        )

    # Sort observations within each machine.
    sort_columns = ["machineID"]

    if "step" in df.columns:
        sort_columns.append("step")
    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if df["timestamp"].isnull().any():
            raise ValueError(
                "Invalid timestamps found."
            )

        sort_columns.append("timestamp")

    df = df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    # Prevent duplicate observations within a machine.
    sequence_column = (
        "step" if "step" in df.columns else "timestamp"
    )

    duplicates = df.duplicated(
        subset=["machineID", sequence_column]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate machine observations found. "
            "Resolve duplicates before preparing training data."
        )

    logger.info("Dataset rows: %s", len(df))

    logger.info(
        "Unique machines: %s",
        df["machineID"].nunique(),
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def generate_features(history):

    features = {}

    for column in FEATURE_COLUMNS:

        values = history[column].to_numpy(
            dtype=np.float64
        )

        features[f"{column}_current"] = float(
            values[-1]
        )

        features[f"{column}_mean"] = float(
            np.mean(values)
        )

        features[f"{column}_std"] = float(
            np.std(values)
        )

        features[f"{column}_min"] = float(
            np.min(values)
        )

        features[f"{column}_max"] = float(
            np.max(values)
        )

        features[f"{column}_median"] = float(
            np.median(values)
        )

        features[f"{column}_recent_mean"] = float(
            np.mean(values[-3:])
        )

        features[f"{column}_previous_mean"] = float(
            np.mean(values[-6:-3])
        )

        features[f"{column}_trend"] = float(
            np.polyfit(
                np.arange(len(values)),
                values,
                1,
            )[0]
        )

    # CPU-memory relationship.
    cpu_values = history["util_CPU"].to_numpy(
        dtype=np.float64
    )

    memory_values = history["util_memory"].to_numpy(
        dtype=np.float64
    )

    features["cpu_memory_correlation"] = float(
        np.corrcoef(
            cpu_values,
            memory_values,
        )[0, 1]
    ) if (
        np.std(cpu_values) > 0
        and np.std(memory_values) > 0
    ) else 0.0

    return features


# ============================================================
# FUTURE WORKLOAD LABEL GENERATION
# ============================================================

def generate_target(future):

    future_cpu = future["util_CPU"].to_numpy(
        dtype=np.float64
    )

    future_memory = future["util_memory"].to_numpy(
        dtype=np.float64
    )

    # High demand takes priority.
    # A resource reaching 85% at any point is a high-demand case.
    high_demand = (
        np.any(
            future_cpu >= HIGH_UTILIZATION_THRESHOLD
        )
        or np.any(
            future_memory >= HIGH_UTILIZATION_THRESHOLD
        )
    )

    if high_demand:
        return "HIGH_DEMAND"

    # Both CPU and memory must remain <=35% throughout
    # the entire future prediction window.
    low_demand = (
        np.all(
            future_cpu <= UNDERUTILIZATION_THRESHOLD
        )
        and np.all(
            future_memory <= UNDERUTILIZATION_THRESHOLD
        )
    )

    if low_demand:
        return "LOW_DEMAND"

    return "NORMAL_DEMAND"


# ============================================================
# DATASET GENERATION
# ============================================================

def prepare_optimization_dataset(df):

    records = []

    machines = df.groupby(
        "machineID",
        sort=False,
    )

    total_machines = df["machineID"].nunique()

    logger.info(
        "Preparing windows for %s machines...",
        total_machines,
    )

    for machine_id, machine_data in machines:

        machine_data = machine_data.reset_index(
            drop=True
        )

        total_rows = len(machine_data)

        minimum_required = (
            WINDOW_SIZE + FORECAST_HORIZON
        )

        if total_rows < minimum_required:
            logger.warning(
                "Skipping machine %s: insufficient observations.",
                machine_id,
            )
            continue

        # Each window contains:
        # 12 historical observations + 6 future observations.
        for start in range(
            total_rows - minimum_required + 1
        ):

            history_start = start

            history_end = start + WINDOW_SIZE

            future_end = (
                history_end + FORECAST_HORIZON
            )

            history = machine_data.iloc[
                history_start:history_end
            ]

            future = machine_data.iloc[
                history_end:future_end
            ]

            features = generate_features(
                history
            )

            target = generate_target(
                future
            )

            # Metadata is retained for validation and auditing.
            # These fields must NOT be used as ML features.
            record = {
                "machineID": machine_id,
                "history_start_step": (
                    history.iloc[0]["step"]
                    if "step" in history.columns
                    else None
                ),
                "history_end_step": (
                    history.iloc[-1]["step"]
                    if "step" in history.columns
                    else None
                ),
                "target_start_step": (
                    future.iloc[0]["step"]
                    if "step" in future.columns
                    else None
                ),
                "target_end_step": (
                    future.iloc[-1]["step"]
                    if "step" in future.columns
                    else None
                ),
                "target": target,
                **features,
            }

            records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError(
            "No training samples could be generated."
        )

    # Ensure all expected target classes are recognized.
    unexpected = set(result["target"].unique()) - set(
        TARGET_CLASSES
    )

    if unexpected:
        raise ValueError(
            f"Unexpected target classes: {unexpected}"
        )

    return result


# ============================================================
# DATASET SUMMARY
# ============================================================

def generate_summary(source, prepared):

    class_counts = (
        prepared["target"]
        .value_counts()
        .reindex(
            TARGET_CLASSES,
            fill_value=0,
        )
    )

    class_distribution = {}

    for label, count in class_counts.items():

        percentage = (
            count / len(prepared)
        ) * 100

        class_distribution[label] = {
            "samples": int(count),
            "percentage": round(
                float(percentage),
                2,
            ),
        }

    metadata_columns = {
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
        "target",
    }

    feature_columns = [
        column
        for column in prepared.columns
        if column not in metadata_columns
    ]

    return {
        "project": "CloudMorph",
        "task": "Optimization Workload Classification",
        "source_dataset": str(INPUT_FILE),
        "source_rows": int(len(source)),
        "machine_count": int(
            source["machineID"].nunique()
        ),
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "sampling_interval_minutes": 5,
        "historical_window_minutes": 60,
        "prediction_horizon_minutes": 30,
        "underutilization_threshold": (
            UNDERUTILIZATION_THRESHOLD
        ),
        "high_utilization_threshold": (
            HIGH_UTILIZATION_THRESHOLD
        ),
        "total_training_samples": int(
            len(prepared)
        ),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_classes": TARGET_CLASSES,
        "class_distribution": class_distribution,
        "label_definition": {
            "HIGH_DEMAND": (
                "CPU or memory reaches at least 85% "
                "at any point in the next 30 minutes."
            ),
            "LOW_DEMAND": (
                "CPU and memory both remain at or below 35% "
                "throughout the next 30 minutes."
            ),
            "NORMAL_DEMAND": (
                "All remaining future workload conditions."
            ),
        },
        "label_source": (
            "Future observed workload; derived labels, "
            "not actual cloud resizing outcomes."
        ),
        "feature_leakage_policy": (
            "Only historical observations are used "
            "to calculate model features."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting CloudMorph optimization dataset preparation."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = load_dataset()

    prepared = prepare_optimization_dataset(
        source
    )

    # Verify no missing feature values were generated.
    metadata_columns = {
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
        "target",
    }

    feature_columns = [
        column
        for column in prepared.columns
        if column not in metadata_columns
    ]

    if prepared[feature_columns].isnull().any().any():
        raise ValueError(
            "Missing values generated during feature engineering."
        )

    if not np.isfinite(
        prepared[feature_columns].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Non-finite values generated during feature engineering."
        )

    prepared.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = generate_summary(
        source,
        prepared,
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    logger.info("Dataset preparation completed.")

    logger.info(
        "Prepared samples: %s",
        len(prepared),
    )

    logger.info(
        "Generated features: %s",
        len(feature_columns),
    )

    logger.info(
        "Class distribution: %s",
        summary["class_distribution"],
    )

    logger.info(
        "Training dataset saved: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Dataset summary saved: %s",
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()
