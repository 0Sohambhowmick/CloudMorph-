import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CLOUDMORPH - TEMPORAL FEATURE EXPERIMENT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "datasets" / "processed" / "server_usage_clean.csv"

OUTPUT_DIR = ROOT / "ml" / "models" / "temporal_features_experiment"

OUTPUT_FILE = OUTPUT_DIR / "temporal_features_dataset.csv"

WINDOW_SIZE = 12
FORECAST_HORIZON = 6

HIGH_UTILIZATION_THRESHOLD = 85.0
UNDERUTILIZATION_THRESHOLD = 35.0

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
    "util_disk",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphTemporalExperiment")


# ============================================================
# LOAD AND VALIDATE SOURCE DATA
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = ["machineID", *FEATURE_COLUMNS]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "step" not in df.columns and "timestamp" not in df.columns:
        raise ValueError("Missing step/timestamp column.")

    if df.empty:
        raise ValueError("Source dataset is empty.")

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[required].isnull().any().any():
        raise ValueError("Missing or invalid source values.")

    if not np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError("Infinite source values detected.")

    for column in UTILIZATION_COLUMNS:
        if (
            (df[column] < 0)
            | (df[column] > 100)
        ).any():
            raise ValueError(
                f"{column} outside the 0-100 range."
            )

    for column in ["load1", "load5", "load15"]:
        if (df[column] < 0).any():
            raise ValueError(f"Negative values in {column}.")

    if df["machineID"].isnull().any():
        raise ValueError("Missing machine IDs.")

    sort_columns = ["machineID"]

    if "step" in df.columns:
        df["step"] = pd.to_numeric(
            df["step"],
            errors="coerce",
        )

        if df["step"].isnull().any():
            raise ValueError("Invalid step values.")

        sort_columns.append("step")
        sequence_column = "step"

    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if df["timestamp"].isnull().any():
            raise ValueError("Invalid timestamps.")

        sort_columns.append("timestamp")
        sequence_column = "timestamp"

    df = df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    if df.duplicated(
        subset=["machineID", sequence_column]
    ).any():
        raise ValueError("Duplicate machine observations.")

    logger.info("Source rows: %s", len(df))
    logger.info("Machines: %s", df.machineID.nunique())

    return df


# ============================================================
# ORIGINAL TARGET GENERATION
# ============================================================

def generate_target(future):

    future_cpu = future["util_CPU"].to_numpy(
        dtype=np.float64
    )

    future_memory = future["util_memory"].to_numpy(
        dtype=np.float64
    )

    high_demand = (
        np.any(future_cpu >= HIGH_UTILIZATION_THRESHOLD)
        or np.any(future_memory >= HIGH_UTILIZATION_THRESHOLD)
    )

    if high_demand:
        return "HIGH_DEMAND"

    low_demand = (
        np.all(future_cpu <= UNDERUTILIZATION_THRESHOLD)
        and np.all(future_memory <= UNDERUTILIZATION_THRESHOLD)
    )

    if low_demand:
        return "LOW_DEMAND"

    return "NORMAL_DEMAND"


# ============================================================
# ORIGINAL FEATURE GENERATION
# ============================================================

def generate_base_features(history):

    features = {}

    for column in FEATURE_COLUMNS:

        values = history[column].to_numpy(
            dtype=np.float64
        )

        features[f"{column}_current"] = float(values[-1])
        features[f"{column}_mean"] = float(np.mean(values))
        features[f"{column}_std"] = float(np.std(values))
        features[f"{column}_min"] = float(np.min(values))
        features[f"{column}_max"] = float(np.max(values))
        features[f"{column}_median"] = float(np.median(values))

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

    cpu = history["util_CPU"].to_numpy(dtype=np.float64)
    memory = history["util_memory"].to_numpy(dtype=np.float64)

    features["cpu_memory_correlation"] = (
        float(np.corrcoef(cpu, memory)[0, 1])
        if np.std(cpu) > 0 and np.std(memory) > 0
        else 0.0
    )

    # Preserve the two features from the existing experiment.
    features["memory_recent_max"] = float(
        np.max(memory[-6:])
    )

    features["cpu_recent_max"] = float(
        np.max(cpu[-6:])
    )

    return features


# ============================================================
# NEW EXPERIMENTAL MEMORY FEATURES
# ============================================================

def generate_temporal_memory_features(history):

    memory = history["util_memory"].to_numpy(
        dtype=np.float64
    )

    recent = memory[-3:]
    previous = memory[-6:-3]

    recent_slope = float(
        np.polyfit(np.arange(3), recent, 1)[0]
    )

    previous_slope = float(
        np.polyfit(np.arange(3), previous, 1)[0]
    )

    recent_mean = float(np.mean(recent))
    previous_mean = float(np.mean(previous))

    features = {

        # 1. Recent 15-minute memory growth.
        "memory_recent_slope": recent_slope,

        # 2. Earlier 15-minute memory growth.
        "memory_previous_slope": previous_slope,

        # 3. Change in growth rate.
        "memory_acceleration": (
            recent_slope - previous_slope
        ),

        # 4. Recent average relative to the full hour.
        "memory_recent_long_gap": (
            recent_mean - float(np.mean(memory))
        ),

        # 5. Recent memory fluctuation.
        "memory_recent_range": float(
            np.max(recent) - np.min(recent)
        ),

        # 6. Recent memory change per five-minute interval.
        "memory_growth_per_5min": (
            (recent_mean - previous_mean) / 3.0
        ),
    }

    return features


# ============================================================
# EXPERIMENTAL DATASET GENERATION
# ============================================================

def prepare_experiment(df):

    records = []

    for machine_id, machine_data in df.groupby(
        "machineID",
        sort=False,
    ):

        machine_data = machine_data.reset_index(drop=True)

        total_rows = len(machine_data)

        minimum_required = (
            WINDOW_SIZE + FORECAST_HORIZON
        )

        if total_rows < minimum_required:
            continue

        for start in range(
            total_rows - minimum_required + 1
        ):

            history_end = start + WINDOW_SIZE
            future_end = history_end + FORECAST_HORIZON

            history = machine_data.iloc[
                start:history_end
            ]

            future = machine_data.iloc[
                history_end:future_end
            ]

            base_features = generate_base_features(history)

            temporal_features = (
                generate_temporal_memory_features(history)
            )

            target = generate_target(future)

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

                **base_features,
                **temporal_features,
            }

            records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError("No experimental samples generated.")

    feature_columns = [
        c for c in result.columns
        if c not in {
            "machineID",
            "history_start_step",
            "history_end_step",
            "target_start_step",
            "target_end_step",
            "target",
        }
    ]

    if not np.isfinite(
        result[feature_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite experimental features.")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("Starting temporal feature experiment.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = load_dataset()

    prepared = prepare_experiment(source)

    prepared.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info("Experimental dataset saved: %s", OUTPUT_FILE)

    logger.info("Total samples: %s", len(prepared))

    logger.info(
        "Feature count: %s",
        len([
            c for c in prepared.columns
            if c not in {
                "machineID",
                "history_start_step",
                "history_end_step",
                "target_start_step",
                "target_end_step",
                "target",
            }
        ]),
    )

    logger.info(
        "Target distribution:\n%s",
        prepared["target"].value_counts().to_string(),
    )

    logger.info("Temporal feature experiment completed.")


if __name__ == "__main__":
    main()
