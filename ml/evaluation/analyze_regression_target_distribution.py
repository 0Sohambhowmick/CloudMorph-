
"""
CloudMorph - Regression Target Distribution Analysis

Measures how frequently future CPU and memory demand reaches
elevated and high utilization levels across machine-aware splits.

Read-only analysis. Does not modify or retrain models.
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

RANDOM_STATE = 42


def main():

    print("Loading datasets...")

    features = pd.read_csv(FEATURE_PATH)
    raw = pd.read_csv(RAW_PATH)

    features["machineID"] = features["machineID"].astype(str)
    raw["machineID"] = raw["machineID"].astype(str)

    raw["step"] = pd.to_numeric(
        raw["step"],
        errors="raise",
    ).astype(int)

    for column in ["util_CPU", "util_memory"]:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="raise",
        )

    # Reconstruct future peak targets exactly as in training.

    print("Reconstructing future peak targets...")

    lookup = {
        (row.machineID, row.step): (
            float(row.util_CPU),
            float(row.util_memory),
        )
        for row in raw.itertuples(index=False)
    }

    cpu_targets = []
    memory_targets = []

    for row in features.itertuples(index=False):

        machine_id = row.machineID

        start_step = int(row.target_start_step)
        end_step = int(row.target_end_step)

        steps = list(range(start_step, end_step + 1))

        if len(steps) != 6:
            raise ValueError(
                f"Expected 6 future steps for machine {machine_id}."
            )

        future = []

        for step in steps:

            key = (machine_id, step)

            if key not in lookup:
                raise ValueError(
                    f"Missing raw data for machine {machine_id}, "
                    f"step {step}."
                )

            future.append(lookup[key])

        cpu_targets.append(max(value[0] for value in future))
        memory_targets.append(max(value[1] for value in future))

    y_cpu = pd.Series(cpu_targets, index=features.index)
    y_memory = pd.Series(memory_targets, index=features.index)

    groups = features["machineID"]

    # Reconstruct exact original machine-aware split.

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
        raise ValueError("Invalid historical feature values.")

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_validation_idx, test_idx = next(
        outer_splitter.split(X, y_cpu, groups)
    )

    X_train_validation = X.iloc[train_validation_idx]
    y_train_validation = y_cpu.iloc[train_validation_idx]
    groups_train_validation = groups.iloc[train_validation_idx]

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

    train_idx = train_validation_idx[train_relative_idx]
    validation_idx = train_validation_idx[validation_relative_idx]

    splits = {
        "TRAIN": train_idx,
        "VALIDATION": validation_idx,
        "TEST": test_idx,
        "ALL DATA": np.arange(len(features)),
    }

    def print_distribution(name, indices, values, resource):

        series = values.iloc[indices]

        categories = pd.cut(
            series,
            bins=[-np.inf, 35, 70, 85, np.inf],
            labels=["LOW", "NORMAL", "ELEVATED", "HIGH"],
            right=False,
        )

        counts = categories.value_counts(
            sort=False,
        )

        total = len(series)

        print(f"\n{resource} — {name}")
        print("-" * 65)
        print(f"Samples: {total:,}")

        for category in ["LOW", "NORMAL", "ELEVATED", "HIGH"]:

            count = int(counts.get(category, 0))
            percentage = count / total * 100 if total else 0

            print(
                f"{category:<12} {count:>8,} "
                f"({percentage:>6.2f}%)"
            )

        print(
            f"Mean: {series.mean():.2f}% | "
            f"Median: {series.median():.2f}% | "
            f"Maximum: {series.max():.2f}%"
        )

    print("\n" + "=" * 75)
    print("FUTURE CPU AND MEMORY TARGET DISTRIBUTION")
    print("=" * 75)

    for split_name, indices in splits.items():

        print(f"\n{'=' * 75}")
        print(f"SPLIT: {split_name}")
        print(f"{'=' * 75}")

        print_distribution(
            split_name,
            indices,
            y_cpu,
            "CPU",
        )

        print_distribution(
            split_name,
            indices,
            y_memory,
            "MEMORY",
        )

        cpu_high = y_cpu.iloc[indices] >= 85
        memory_high = y_memory.iloc[indices] >= 85

        either_high = cpu_high | memory_high
        both_high = cpu_high & memory_high

        total = len(indices)

        print("\nCOMBINED HIGH-DEMAND WINDOWS")
        print("-" * 65)

        print(
            f"CPU >=85% OR Memory >=85%: "
            f"{either_high.sum():,} "
            f"({either_high.mean() * 100:.2f}%)"
        )

        print(
            f"CPU >=85% AND Memory >=85%: "
            f"{both_high.sum():,} "
            f"({both_high.mean() * 100:.2f}%)"
        )

    print("\nAnalysis completed.")
    print("No training, model or dataset files were modified.")


if __name__ == "__main__":
    main()
