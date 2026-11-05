
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "datasets" / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = ROOT / "datasets" / "processed" / "sequences"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Forecasting configuration
WINDOW_SIZE = 24
EXPECTED_STEPS = 144

TRAIN_END = 100
VALIDATION_END = 122

FEATURE_COLUMNS = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

TARGET_COLUMNS = [
    "util_CPU",
    "util_memory",
]


def generate_sequences(df):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    train_ids, val_ids, test_ids = [], [], []
    train_times, val_times, test_times = [], [], []

    machine_count = 0

    for machine_id, group in df.groupby("machineID", sort=True):
        group = group.sort_values("step")

        if len(group) != EXPECTED_STEPS:
            raise ValueError(
                f"Machine {machine_id} has {len(group)} observations."
            )

        if not np.array_equal(
            group["step"].to_numpy(),
            np.arange(EXPECTED_STEPS)
        ):
            raise ValueError(
                f"Invalid step sequence for machine {machine_id}."
            )

        features = group[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        targets = group[TARGET_COLUMNS].to_numpy(
            dtype=np.float32
        )

        timestamps = group["timestamp"].to_numpy(
            dtype=np.int64
        )

        # Target index is the observation we want to predict.
        for target_index in range(
            WINDOW_SIZE, EXPECTED_STEPS
        ):
            X_window = features[
                target_index - WINDOW_SIZE:target_index
            ]

            y_target = targets[target_index]
            target_timestamp = timestamps[target_index]

            if target_index < TRAIN_END:
                X_train.append(X_window)
                y_train.append(y_target)
                train_ids.append(machine_id)
                train_times.append(target_timestamp)

            elif target_index < VALIDATION_END:
                X_val.append(X_window)
                y_val.append(y_target)
                val_ids.append(machine_id)
                val_times.append(target_timestamp)

            else:
                X_test.append(X_window)
                y_test.append(y_target)
                test_ids.append(machine_id)
                test_times.append(target_timestamp)

        machine_count += 1

    def convert(values, shape, dtype=np.float32):
        return np.asarray(values, dtype=dtype).reshape(shape)

    feature_count = len(FEATURE_COLUMNS)
    target_count = len(TARGET_COLUMNS)

    result = {
        "X_train": convert(
            X_train,
            (-1, WINDOW_SIZE, feature_count)
        ),
        "y_train": convert(
            y_train,
            (-1, target_count)
        ),
        "X_val": convert(
            X_val,
            (-1, WINDOW_SIZE, feature_count)
        ),
        "y_val": convert(
            y_val,
            (-1, target_count)
        ),
        "X_test": convert(
            X_test,
            (-1, WINDOW_SIZE, feature_count)
        ),
        "y_test": convert(
            y_test,
            (-1, target_count)
        ),
        "train_machine_ids": np.asarray(train_ids),
        "val_machine_ids": np.asarray(val_ids),
        "test_machine_ids": np.asarray(test_ids),
        "train_timestamps": np.asarray(train_times),
        "val_timestamps": np.asarray(val_times),
        "test_timestamps": np.asarray(test_times),
        "feature_names": np.asarray(FEATURE_COLUMNS),
        "target_names": np.asarray(TARGET_COLUMNS),
    }

    return result, machine_count


def main():
    print("=" * 65)
    print("CLOUDMORPH - LSTM SEQUENCE GENERATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_columns = (
        ["machineID", "timestamp", "step"]
        + FEATURE_COLUMNS
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in processed dataset."
        )

    print(f"Input records: {len(df):,}")
    print(f"Machines: {df['machineID'].nunique():,}")
    print(f"Input window: {WINDOW_SIZE} observations")
    print("Forecast horizon: 1 observation (5 minutes)")

    sequences, machine_count = generate_sequences(df)

    output_file = OUTPUT_DIR / "server_lstm_sequences.npz"

    np.savez_compressed(
        output_file,
        **sequences
    )

    manifest = {
        "dataset": "Alibaba Cluster Trace 2017",
        "model_task": "Next-step CPU and memory forecasting",
        "machines": machine_count,
        "input_window_steps": WINDOW_SIZE,
        "input_interval_seconds": 300,
        "forecast_horizon_steps": 1,
        "forecast_horizon_seconds": 300,
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "chronological_split": {
            "training_target_indices": [12, 99],
            "validation_target_indices": [100, 121],
            "testing_target_indices": [122, 143],
        },
        "shapes": {
            name: list(value.shape)
            for name, value in sequences.items()
            if name.startswith(("X_", "y_"))
        },
        "output_file": str(output_file),
    }

    manifest_file = OUTPUT_DIR / "sequence_manifest.json"

    with open(
        manifest_file, "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=4)

    print("\nSEQUENCE GENERATION SUCCESSFUL")

    for name in [
        "X_train", "y_train",
        "X_val", "y_val",
        "X_test", "y_test",
    ]:
        print(f"{name}: {sequences[name].shape}")

    print("\nTotal sequences:")
    total = (
        len(sequences["X_train"])
        + len(sequences["X_val"])
        + len(sequences["X_test"])
    )
    print(f"{total:,}")

    print("\nSaved sequence dataset:")
    print(output_file)

    print("\nSaved manifest:")
    print(manifest_file)


if __name__ == "__main__":
    main()
