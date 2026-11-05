
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "sequences"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "server_multihorizon_sequences.npz"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "multihorizon_sequence_manifest.json"
)

WINDOW_SIZE = 12

# 15 minutes = 3 observations
# 30 minutes = 6 observations
HORIZONS = {
    "15min": 3,
    "30min": 6,
}

FEATURES = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

TARGETS = [
    "util_CPU",
    "util_memory",
]

# Same temporal split as the original sequence pipeline.
TRAIN_END = 100
VALIDATION_END = 122


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "machineID",
        "timestamp",
        *FEATURES,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.sort_values(
        ["machineID", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# VALIDATE MACHINE TIMELINES
# ============================================================

def validate_data(df):

    counts = df.groupby("machineID").size()

    if not (counts == 144).all():
        raise ValueError(
            "Every machine must have exactly 144 observations."
        )

    for machine_id, group in df.groupby("machineID"):

        timestamps = group["timestamp"].to_numpy()

        if not np.all(np.diff(timestamps) == 300):
            raise ValueError(
                f"Invalid timestamp interval for machine {machine_id}"
            )

    if df[FEATURES].isnull().any().any():
        raise ValueError(
            "Missing feature values detected."
        )


# ============================================================
# GENERATE MULTI-HORIZON SEQUENCES
# ============================================================

def generate_sequences(df):

    X_train, X_val, X_test = [], [], []

    y_train_15, y_val_15, y_test_15 = [], [], []
    y_train_30, y_val_30, y_test_30 = [], [], []

    metadata_train = []
    metadata_val = []
    metadata_test = []

    for machine_id, group in df.groupby("machineID"):

        group = group.sort_values("timestamp")

        features = group[FEATURES].to_numpy(
            dtype=np.float32
        )

        targets = group[TARGETS].to_numpy(
            dtype=np.float32
        )

        timestamps = group["timestamp"].to_numpy()

        if len(group) != 144:
            raise ValueError(
                f"Machine {machine_id} has invalid history length."
            )

        # Forecast origin is the final observation in each window.
        # Both targets are indexed relative to that origin.

        for origin in range(
            WINDOW_SIZE - 1,
            144 - max(HORIZONS.values()),
        ):

            start = origin - WINDOW_SIZE + 1

            X = features[start:origin + 1]

            target_15 = targets[
                origin + HORIZONS["15min"]
            ]

            target_30 = targets[
                origin + HORIZONS["30min"]
            ]

            metadata = {
                "machineID": int(machine_id),
                "origin_index": int(origin),
                "origin_timestamp": int(timestamps[origin]),
                "target_15_timestamp": int(
                    timestamps[origin + 3]
                ),
                "target_30_timestamp": int(
                    timestamps[origin + 6]
                ),
            }

            # A sample belongs to a split based on BOTH targets.
            # Training: both target indices below 100.
            # Validation: both target indices 100-121.
            # Testing: both target indices 122 or later.

            index_15 = origin + HORIZONS["15min"]
            index_30 = origin + HORIZONS["30min"]

            if index_30 < TRAIN_END:

                X_train.append(X)
                y_train_15.append(target_15)
                y_train_30.append(target_30)
                metadata_train.append(metadata)

            elif (
                index_15 >= TRAIN_END
                and index_30 < VALIDATION_END
            ):

                X_val.append(X)
                y_val_15.append(target_15)
                y_val_30.append(target_30)
                metadata_val.append(metadata)

            elif index_15 >= VALIDATION_END:

                X_test.append(X)
                y_test_15.append(target_15)
                y_test_30.append(target_30)
                metadata_test.append(metadata)

    def stack(values, shape):

        if not values:
            return np.empty(shape, dtype=np.float32)

        return np.asarray(
            values,
            dtype=np.float32,
        )

    X_train = stack(
        X_train,
        (0, WINDOW_SIZE, len(FEATURES)),
    )

    X_val = stack(
        X_val,
        (0, WINDOW_SIZE, len(FEATURES)),
    )

    X_test = stack(
        X_test,
        (0, WINDOW_SIZE, len(FEATURES)),
    )

    arrays = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train_15": stack(
            y_train_15, (0, 2)
        ),

        "y_val_15": stack(
            y_val_15, (0, 2)
        ),

        "y_test_15": stack(
            y_test_15, (0, 2)
        ),

        "y_train_30": stack(
            y_train_30, (0, 2)
        ),

        "y_val_30": stack(
            y_val_30, (0, 2)
        ),

        "y_test_30": stack(
            y_test_30, (0, 2)
        ),
    }

    metadata = {
        "train": metadata_train,
        "validation": metadata_val,
        "test": metadata_test,
    }

    return arrays, metadata


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(arrays, metadata):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        **arrays,
    )

    manifest = {
        "dataset": "Alibaba Cluster Trace 2017",
        "window_size": WINDOW_SIZE,
        "window_duration_minutes": 60,
        "forecast_horizons": {
            "15min": 15,
            "30min": 30,
        },
        "forecast_origin": (
            "Last observation in the input window"
        ),
        "features": FEATURES,
        "targets": TARGETS,
        "split": {
            "training_target_indices": (
                "Both horizon targets below 100"
            ),
            "validation_target_indices": (
                "Both horizon targets between 100 and 121"
            ),
            "testing_target_indices": (
                "Both horizon targets at or above 122"
            ),
        },
        "shapes": {
            name: list(array.shape)
            for name, array in arrays.items()
        },
        "metadata_counts": {
            name: len(items)
            for name, items in metadata.items()
        },
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    print("\nSEQUENCE GENERATION SUCCESSFUL")

    for name, array in arrays.items():

        print(
            f"{name}: {array.shape}"
        )

    print("\nMetadata counts:")

    for name, items in metadata.items():

        print(
            f"{name}: {len(items)}"
        )

    print("\nSaved sequence dataset:")
    print(OUTPUT_PATH)

    print("\nSaved manifest:")
    print(MANIFEST_PATH)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("CLOUDMORPH - MULTI-HORIZON SEQUENCE GENERATION")
    print("=" * 65)

    df = load_data()

    validate_data(df)

    print("\nInput records:", len(df))
    print("Machines:", df["machineID"].nunique())
    print("Input window:", WINDOW_SIZE)
    print("Forecast horizons:", HORIZONS)

    arrays, metadata = generate_sequences(df)

    save_results(arrays, metadata)

    print("\nMULTI-HORIZON SEQUENCE GENERATION COMPLETED")


if __name__ == "__main__":
    main()
