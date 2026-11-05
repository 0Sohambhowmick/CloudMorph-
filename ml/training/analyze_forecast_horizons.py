
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

OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"

JSON_PATH = (
    OUTPUT_DIR / "forecast_horizon_benchmark.json"
)

CSV_PATH = (
    OUTPUT_DIR / "forecast_horizon_machine_results.csv"
)

TARGETS = [
    "util_CPU",
    "util_memory",
]

# Each observation represents five minutes.
HORIZONS = {
    "5_minutes": 1,
    "15_minutes": 3,
    "30_minutes": 6,
    "60_minutes": 12,
}

# Use the same chronological proportions as our original
# per-machine temporal experiment.
TRAIN_END = 100
VALIDATION_END = 122


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required = [
        "machineID",
        "timestamp",
        *TARGETS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.sort_values(
        ["machineID", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_machine_timelines(df):

    expected_observations = 144

    counts = df.groupby("machineID").size()

    if not (counts == expected_observations).all():
        raise ValueError(
            "Some machines do not have exactly 144 observations."
        )

    for machine_id, group in df.groupby("machineID"):

        timestamps = group["timestamp"].to_numpy()

        if not np.all(np.diff(timestamps) == 300):
            raise ValueError(
                f"Unexpected timestamp interval for machine {machine_id}"
            )


# ============================================================
# HORIZON EVALUATION
# ============================================================

def evaluate_horizon(df, horizon):

    machine_results = []

    for machine_id, group in df.groupby("machineID"):

        group = group.sort_values("timestamp")

        for target in TARGETS:

            values = group[target].to_numpy(
                dtype=np.float64
            )

            # Target observations must remain in the test region.
            # Predictions use the observation immediately preceding
            # each target as the persistence forecast.

            actual = values[
                VALIDATION_END + horizon:
            ]

            predicted = values[
                VALIDATION_END:
                len(values) - horizon
            ]

            # Verify equal lengths.
            if len(actual) != len(predicted):
                raise ValueError(
                    "Actual and predicted arrays have different lengths."
                )

            errors = actual - predicted

            mae = float(
                np.mean(np.abs(errors))
            )

            rmse = float(
                np.sqrt(np.mean(errors ** 2))
            )

            machine_results.append({
                "machineID": int(machine_id),
                "target": target,
                "horizon_steps": horizon,
                "horizon_minutes": horizon * 5,
                "samples": int(len(actual)),
                "MAE": mae,
                "RMSE": rmse,
            })

    results_df = pd.DataFrame(machine_results)

    summary = {}

    for target in TARGETS:

        target_df = results_df[
            results_df["target"] == target
        ]

        summary[target] = {
            # Aggregate over all machine-level test predictions.
            "MAE": float(
                np.average(
                    target_df["MAE"],
                    weights=target_df["samples"],
                )
            ),

            "RMSE": float(
                np.sqrt(
                    np.average(
                        target_df["RMSE"] ** 2,
                        weights=target_df["samples"],
                    )
                )
            ),

            "machines": int(
                target_df["machineID"].nunique()
            ),

            "total_predictions": int(
                target_df["samples"].sum()
            ),
        }

    return summary, results_df


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CLOUDMORPH - FORECAST HORIZON BENCHMARK")
    print("=" * 65)

    df = load_data()

    validate_machine_timelines(df)

    print("\nDataset:")
    print("Records:", len(df))
    print("Machines:", df["machineID"].nunique())
    print("Observations per machine: 144")

    print("\nTemporal split:")
    print("Training target indices: 12 to 99")
    print("Validation target indices: 100 to 121")
    print("Test region starts at index: 122")

    benchmark = {}
    all_machine_results = []

    print("\nRunning persistence benchmarks...\n")

    for name, horizon in HORIZONS.items():

        summary, machine_results = evaluate_horizon(
            df,
            horizon,
        )

        benchmark[name] = {
            "horizon_steps": horizon,
            "horizon_minutes": horizon * 5,
            "results": summary,
        }

        all_machine_results.append(
            machine_results.assign(
                horizon=name
            )
        )

        print(f"\n{horizon * 5}-MINUTE FORECAST")

        for target, metrics in summary.items():

            print(f"\n{target}")

            print(
                f"  MAE: {metrics['MAE']:.4f}"
            )

            print(
                f"  RMSE: {metrics['RMSE']:.4f}"
            )

            print(
                f"  Predictions: {metrics['total_predictions']}"
            )

    final_df = pd.concat(
        all_machine_results,
        ignore_index=True,
    )

    final_df.to_csv(
        CSV_PATH,
        index=False,
    )

    report = {
        "dataset": {
            "records": int(len(df)),
            "machines": int(df["machineID"].nunique()),
            "observations_per_machine": 144,
            "interval_seconds": 300,
        },

        "split": {
            "train_end_index_exclusive": TRAIN_END,
            "validation_end_index_exclusive": VALIDATION_END,
            "test_start_index": VALIDATION_END,
        },

        "method": (
            "Persistence baseline: predict future utilization "
            "using the latest observation available at forecast origin."
        ),

        "benchmarks": benchmark,
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print("\nSaved reports:")
    print(JSON_PATH)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
