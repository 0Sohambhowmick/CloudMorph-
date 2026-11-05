
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

RAW_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "ml"
    / "models"
    / "memory_high_demand_classifier"
    / "test_predictions.csv"
)

OUTPUT_DIR = ROOT / "ml" / "evaluation"


# Machines with missed high-memory windows.
MISSED_MACHINES = ["341", "436", "730", "995", "449"]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("CLOUDMORPH MEMORY MACHINE DISTRIBUTION AUDIT")
    print("=" * 75)

    features = pd.read_csv(FEATURE_PATH)

    raw = pd.read_csv(RAW_PATH)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    # -----------------------------------------------------
    # 1. DATA PREPARATION
    # -----------------------------------------------------

    features["machineID"] = (
        features["machineID"].astype(str)
    )

    raw["machineID"] = (
        raw["machineID"].astype(str)
    )

    raw["util_memory"] = pd.to_numeric(
        raw["util_memory"],
        errors="raise",
    )

    raw["util_CPU"] = pd.to_numeric(
        raw["util_CPU"],
        errors="raise",
    )

    raw["step"] = pd.to_numeric(
        raw["step"],
        errors="raise",
    ).astype(int)

    X = features.drop(
        columns=[
            "machineID",
            "history_start_step",
            "history_end_step",
            "target_start_step",
            "target_end_step",
            "target",
        ],
        errors="ignore",
    )

    X = X.apply(pd.to_numeric, errors="raise")

    groups = features["machineID"]

    # -----------------------------------------------------
    # 2. RECONSTRUCT EXACT MACHINE SPLIT
    # -----------------------------------------------------

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_validation_idx, test_idx = next(
        outer.split(X, groups=groups)
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=43,
    )

    train_relative_idx, validation_relative_idx = next(
        inner.split(
            X.iloc[train_validation_idx],
            groups=groups.iloc[train_validation_idx],
        )
    )

    train_idx = train_validation_idx[
        train_relative_idx
    ]

    validation_idx = train_validation_idx[
        validation_relative_idx
    ]

    train_machines = set(
        groups.iloc[train_idx]
    )

    validation_machines = set(
        groups.iloc[validation_idx]
    )

    test_machines = set(
        groups.iloc[test_idx]
    )

    print("\n1. MACHINE SPLIT SUMMARY")

    print(f"Training machines: {len(train_machines)}")
    print(f"Validation machines: {len(validation_machines)}")
    print(f"Test machines: {len(test_machines)}")

    print(
        "Machine overlap:",
        bool(
            (train_machines & validation_machines)
            or (train_machines & test_machines)
            or (validation_machines & test_machines)
        ),
    )

    # -----------------------------------------------------
    # 3. RAW MACHINE-LEVEL MEMORY STATISTICS
    # -----------------------------------------------------

    print("\n2. MACHINE-LEVEL MEMORY DISTRIBUTION")

    machine_stats = (
        raw.groupby("machineID")
        .agg(
            observations=("util_memory", "count"),
            memory_mean=("util_memory", "mean"),
            memory_median=("util_memory", "median"),
            memory_std=("util_memory", "std"),
            memory_min=("util_memory", "min"),
            memory_max=("util_memory", "max"),
            memory_p90=("util_memory", lambda x: x.quantile(0.90)),
            memory_p95=("util_memory", lambda x: x.quantile(0.95)),
            high_memory_observations=(
                "util_memory",
                lambda x: int((x >= 85).sum()),
            ),
            elevated_memory_observations=(
                "util_memory",
                lambda x: int((x >= 70).sum()),
            ),
        )
        .reset_index()
    )

    machine_stats["high_memory_percentage"] = (
        machine_stats["high_memory_observations"]
        / machine_stats["observations"]
        * 100
    )

    machine_stats["machine_split"] = (
        machine_stats["machineID"].map(
            lambda machine:
                "TRAIN" if machine in train_machines
                else "VALIDATION" if machine in validation_machines
                else "TEST"
        )
    )

    print(
        machine_stats.groupby("machine_split")[
            [
                "memory_mean",
                "memory_median",
                "memory_std",
                "memory_p90",
                "memory_p95",
                "memory_max",
                "high_memory_percentage",
            ]
        ]
        .agg(["mean", "median"])
        .round(3)
        .to_string()
    )

    # -----------------------------------------------------
    # 4. MISSED MACHINES VS OTHER TEST MACHINES
    # -----------------------------------------------------

    print("\n3. MISSED MACHINES VS OTHER TEST MACHINES")

    machine_stats["machine_category"] = (
        machine_stats.apply(
            lambda row:
                "MISSED_MACHINE"
                if (
                    row["machineID"] in MISSED_MACHINES
                    and row["machine_split"] == "TEST"
                )
                else "OTHER_TEST_MACHINE"
                if row["machine_split"] == "TEST"
                else row["machine_split"],
            axis=1,
        )
    )

    relevant = machine_stats[
        machine_stats["machine_category"].isin(
            [
                "MISSED_MACHINE",
                "OTHER_TEST_MACHINE",
            ]
        )
    ]

    comparison = (
        relevant.groupby("machine_category")[
            [
                "memory_mean",
                "memory_median",
                "memory_std",
                "memory_p90",
                "memory_p95",
                "memory_max",
                "high_memory_percentage",
            ]
        ]
        .agg(["mean", "median"])
        .round(3)
    )

    print(comparison.to_string())

    # -----------------------------------------------------
    # 5. SPECIFIC MISSED MACHINES
    # -----------------------------------------------------

    print("\n4. INDIVIDUAL MISSED MACHINE STATISTICS")

    missed_stats = machine_stats[
        machine_stats["machineID"].isin(MISSED_MACHINES)
    ].copy()

    print(
        missed_stats[
            [
                "machineID",
                "machine_split",
                "observations",
                "memory_mean",
                "memory_median",
                "memory_std",
                "memory_p90",
                "memory_p95",
                "memory_max",
                "high_memory_percentage",
            ]
        ]
        .round(3)
        .sort_values("machineID")
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # 6. TEST MACHINE PREDICTION PERFORMANCE
    # -----------------------------------------------------

    print("\n5. TEST MACHINE PREDICTION PERFORMANCE")

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
    )

    prediction_summary = (
        predictions.groupby("machineID")
        .apply(
            lambda group: pd.Series({
                "test_windows": len(group),
                "actual_high": int(
                    group["actual_high_memory"].sum()
                ),
                "detected_high": int(
                    group["predicted_high_memory"].sum()
                ),
                "missed_high": int(
                    (
                        (group["actual_high_memory"] == 1)
                        & (group["predicted_high_memory"] == 0)
                    ).sum()
                ),
                "mean_probability": group[
                    "predicted_probability"
                ].mean(),
                "mean_actual_peak": group[
                    "actual_peak_memory"
                ].mean(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    prediction_summary = prediction_summary.merge(
        machine_stats[
            [
                "machineID",
                "memory_mean",
                "memory_p95",
                "memory_max",
                "high_memory_percentage",
            ]
        ],
        on="machineID",
        how="left",
    )

    print(
        prediction_summary[
            prediction_summary["actual_high"] > 0
        ]
        .round(3)
        .sort_values("missed_high", ascending=False)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # 7. SAVE RESULTS
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    machine_stats.to_csv(
        OUTPUT_DIR
        / "memory_machine_distribution_all.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "memory_machine_distribution_comparison.csv"
    )

    missed_stats.to_csv(
        OUTPUT_DIR
        / "memory_missed_machine_statistics.csv",
        index=False,
    )

    prediction_summary.to_csv(
        OUTPUT_DIR
        / "memory_machine_prediction_performance.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("MACHINE DISTRIBUTION AUDIT COMPLETED")
    print("=" * 75)

    print("\nGenerated files:")
    print("1. memory_machine_distribution_all.csv")
    print("2. memory_machine_distribution_comparison.csv")
    print("3. memory_missed_machine_statistics.csv")
    print("4. memory_machine_prediction_performance.csv")


if __name__ == "__main__":
    main()
