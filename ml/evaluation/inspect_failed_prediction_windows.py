
"""
CloudMorph - Exact Failed Prediction Window Inspection

Reconstructs the original test split, verifies saved predictions,
and displays the historical and future windows for failed predictions.

Does not retrain or modify any model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "resource_demand_regressor"
    / "test_predictions.csv"
)

PROBLEM_MACHINES = ["436", "449", "995"]

RANDOM_STATE = 42


def load_and_reconstruct():

    print("Loading datasets...")

    raw = pd.read_csv(RAW_PATH)
    features = pd.read_csv(FEATURE_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)

    required_raw = [
        "machineID",
        "step",
        "util_CPU",
        "util_memory",
    ]

    required_features = [
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
        "target",
    ]

    for column in required_raw:
        if column not in raw.columns:
            raise ValueError(f"Missing raw column: {column}")

    for column in required_features:
        if column not in features.columns:
            raise ValueError(f"Missing feature column: {column}")

    # Match the original training script's normalization.

    features["machineID"] = features["machineID"].astype(str)
    raw["machineID"] = raw["machineID"].astype(str)

    for column in [
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
    ]:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        ).astype(int)

    raw["step"] = pd.to_numeric(
        raw["step"],
        errors="raise",
    ).astype(int)

    for column in ["util_CPU", "util_memory"]:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="raise",
        )

    if raw.duplicated(["machineID", "step"]).any():
        raise ValueError("Duplicate machineID/step records found.")

    # Reconstruct the same future targets used during training.

    print("Reconstructing future CPU and memory targets...")

    raw_lookup = {}

    for row in raw.itertuples(index=False):
        raw_lookup[(row.machineID, row.step)] = (
            float(row.util_CPU),
            float(row.util_memory),
        )

    cpu_targets = []
    memory_targets = []

    for row in features.itertuples(index=False):

        machine_id = row.machineID

        start_step = int(row.target_start_step)
        end_step = int(row.target_end_step)

        expected_steps = list(range(start_step, end_step + 1))

        if len(expected_steps) != 6:
            raise ValueError(
                f"Expected 6 future observations for machine "
                f"{machine_id}, steps {start_step}-{end_step}."
            )

        future_records = []

        for step in expected_steps:

            key = (machine_id, step)

            if key not in raw_lookup:
                raise ValueError(
                    f"Missing raw observation: machine={machine_id}, "
                    f"step={step}"
                )

            future_records.append(raw_lookup[key])

        cpu_targets.append(
            max(record[0] for record in future_records)
        )

        memory_targets.append(
            max(record[1] for record in future_records)
        )

    y_cpu = pd.Series(cpu_targets, dtype=float)
    y_memory = pd.Series(memory_targets, dtype=float)

    # Reconstruct the exact 55-feature matrix.

    metadata_columns = [
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
        "target",
    ]

    feature_columns = [
        column
        for column in features.columns
        if column not in metadata_columns
    ]

    X = features[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError("Missing or non-numeric historical features.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values in historical features.")

    groups = features["machineID"]

    print(f"Feature rows: {len(features)}")
    print(f"Machines: {groups.nunique()}")
    print(f"Historical features: {len(feature_columns)}")

    # Match the original training script's exact split call.

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    _, test_idx = next(
        splitter.split(X, y_cpu, groups)
    )

    test_features = features.iloc[test_idx].copy()

    test_features = test_features.reset_index(drop=True)

    predictions = pd.read_csv(PREDICTION_PATH)

    if len(test_features) != len(predictions):
        raise ValueError(
            f"Row count mismatch: reconstructed={len(test_features)}, "
            f"predictions={len(predictions)}"
        )

    # Normalize IDs in saved predictions too.

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
    )

    expected_ids = test_features["machineID"].to_numpy()
    saved_ids = predictions["machineID"].to_numpy()

    if not np.array_equal(expected_ids, saved_ids):
        mismatch_positions = np.where(expected_ids != saved_ids)[0]

        print("\nFirst mismatched positions:")

        for position in mismatch_positions[:10]:
            print(
                f"Position {position}: "
                f"expected machine={expected_ids[position]}, "
                f"saved machine={saved_ids[position]}"
            )

        raise ValueError(
            "Machine IDs or row ordering still do not match."
        )

    # Verify actual targets against the saved predictions.

    checks = [
        ("actual_peak_cpu", y_cpu.iloc[test_idx].to_numpy()),
        (
            "actual_peak_memory",
            y_memory.iloc[test_idx].to_numpy(),
        ),
    ]

    for column, expected_values in checks:

        if column not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {column}"
            )

        saved_values = predictions[column].to_numpy(dtype=float)

        if not np.allclose(
            expected_values,
            saved_values,
            atol=1e-5,
            rtol=0,
        ):

            raise ValueError(
                f"Target verification failed for {column}. "
                "Stopping to avoid incorrect window matching."
            )

    print("\nSUCCESS: Test rows and saved targets verified.")
    print("The reconstructed test split matches the saved predictions.")

    matched = test_features[
        [
            "machineID",
            "history_start_step",
            "history_end_step",
            "target_start_step",
            "target_end_step",
        ]
    ].copy()

    # Keep machineID only from test_features to avoid duplicate columns.

    prediction_columns = predictions.drop(
        columns=["machineID"]
    )

    matched = pd.concat(
        [
            matched.reset_index(drop=True),
            prediction_columns.reset_index(drop=True),
        ],
        axis=1,
    )

    return raw, matched


def inspect_machine(raw, matched, machine_id):

    machine_rows = matched[
        matched["machineID"] == machine_id
    ].copy()

    if machine_rows.empty:
        print(f"\nMachine {machine_id}: no test predictions found.")
        return

    # Rank windows by the largest CPU or memory underprediction.

    machine_rows["cpu_underprediction"] = (
        machine_rows["actual_peak_cpu"]
        - machine_rows["predicted_peak_cpu"]
    )

    machine_rows["memory_underprediction"] = (
        machine_rows["actual_peak_memory"]
        - machine_rows["predicted_peak_memory"]
    )

    machine_rows["largest_underprediction"] = machine_rows[
        ["cpu_underprediction", "memory_underprediction"]
    ].max(axis=1)

    machine_rows = machine_rows[
        machine_rows["largest_underprediction"] > 0
    ].sort_values(
        "largest_underprediction",
        ascending=False,
    )

    print("\n" + "=" * 90)
    print(f"MACHINE {machine_id} — WORST UNDERPREDICTIONS")
    print("=" * 90)

    if machine_rows.empty:
        print("No underpredictions found.")
        return

    for rank, (_, prediction) in enumerate(
        machine_rows.head(3).iterrows(),
        start=1,
    ):

        history_start = int(prediction["history_start_step"])
        history_end = int(prediction["history_end_step"])

        target_start = int(prediction["target_start_step"])
        target_end = int(prediction["target_end_step"])

        print(f"\nWINDOW {rank}")

        print(
            f"CPU actual={prediction['actual_peak_cpu']:.2f}% | "
            f"predicted={prediction['predicted_peak_cpu']:.2f}% | "
            f"error={prediction['cpu_underprediction']:.2f}"
        )

        print(
            f"Memory actual={prediction['actual_peak_memory']:.2f}% | "
            f"predicted={prediction['predicted_peak_memory']:.2f}% | "
            f"error={prediction['memory_underprediction']:.2f}"
        )

        print("\nHistorical 12 observations + Future 6 observations:")

        selected = raw[
            (raw["machineID"] == machine_id)
            & (
                raw["step"].between(
                    history_start,
                    history_end,
                )
                | raw["step"].between(
                    target_start,
                    target_end,
                )
            )
        ].copy()

        selected = selected.sort_values("step")

        selected["phase"] = np.where(
            selected["step"].between(
                history_start,
                history_end,
            ),
            "HISTORY",
            "FUTURE",
        )

        expected_count = 18

        if len(selected) != expected_count:
            print(
                f"WARNING: Expected {expected_count} rows, "
                f"found {len(selected)}."
            )

        print(
            selected[
                [
                    "step",
                    "phase",
                    "util_CPU",
                    "util_memory",
                ]
            ].to_string(
                index=False,
                formatters={
                    "util_CPU": "{:.2f}".format,
                    "util_memory": "{:.2f}".format,
                },
            )
        )


def main():

    raw, matched = load_and_reconstruct()

    for machine_id in PROBLEM_MACHINES:
        inspect_machine(raw, matched, machine_id)

    print("\nInspection completed.")
    print("No model files or training data were modified.")


if __name__ == "__main__":
    main()
