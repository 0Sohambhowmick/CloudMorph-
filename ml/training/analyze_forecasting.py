
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

REPORT_PATH = (
    OUTPUT_DIR
    / "forecasting_diagnostic.json"
)

CSV_REPORT_PATH = (
    OUTPUT_DIR
    / "machine_forecasting_stability.csv"
)

TARGETS = [
    "util_CPU",
    "util_memory",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "machineID",
        "timestamp",
        *TARGETS,
    ]

    missing = [
        column
        for column in required_columns
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
# GLOBAL STABILITY ANALYSIS
# ============================================================

def analyze_global_stability(df):

    results = {}

    for target in TARGETS:

        # Difference between consecutive observations
        # for the same machine.
        changes = (
            df.groupby("machineID")[target]
            .diff()
            .dropna()
        )

        absolute_changes = changes.abs()

        results[target] = {
            "mean_utilization": float(
                df[target].mean()
            ),

            "std_utilization": float(
                df[target].std()
            ),

            "mean_absolute_change": float(
                absolute_changes.mean()
            ),

            "median_absolute_change": float(
                absolute_changes.median()
            ),

            "p90_absolute_change": float(
                absolute_changes.quantile(0.90)
            ),

            "p95_absolute_change": float(
                absolute_changes.quantile(0.95)
            ),

            "max_absolute_change": float(
                absolute_changes.max()
            ),

            "percentage_change_below_1": float(
                (absolute_changes < 1).mean() * 100
            ),

            "percentage_change_below_2": float(
                (absolute_changes < 2).mean() * 100
            ),

            "percentage_change_below_5": float(
                (absolute_changes < 5).mean() * 100
            ),
        }

    return results


# ============================================================
# MACHINE-LEVEL ANALYSIS
# ============================================================

def analyze_machine_stability(df):

    machine_results = []

    for machine_id, group in df.groupby("machineID"):

        group = group.sort_values("timestamp")

        row = {
            "machineID": int(machine_id),
            "observations": int(len(group)),
        }

        for target in TARGETS:

            changes = (
                group[target]
                .diff()
                .dropna()
                .abs()
            )

            row[f"{target}_mean_change"] = float(
                changes.mean()
            )

            row[f"{target}_std_change"] = float(
                changes.std()
            )

        machine_results.append(row)

    return pd.DataFrame(machine_results)


# ============================================================
# PERSISTENCE BASELINE ANALYSIS
# ============================================================

def analyze_persistence(df):

    results = {}

    for target in TARGETS:

        grouped = df.groupby("machineID")[target]

        actual = df[target]

        previous = grouped.shift(1)

        valid = previous.notna()

        actual = actual[valid]
        previous = previous[valid]

        mae = np.mean(
            np.abs(actual - previous)
        )

        rmse = np.sqrt(
            np.mean((actual - previous) ** 2)
        )

        results[target] = {
            "persistence_mae": float(mae),
            "persistence_rmse": float(rmse),
        }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CLOUDMORPH - FORECASTING DIAGNOSTIC")
    print("=" * 65)

    df = load_data()

    print("\nDataset information:")
    print("Total records:", len(df))
    print("Unique machines:", df["machineID"].nunique())

    print(
        "Timestamp range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    global_results = analyze_global_stability(df)

    machine_results = analyze_machine_stability(df)

    persistence_results = analyze_persistence(df)

    machine_results.to_csv(
        CSV_REPORT_PATH,
        index=False,
    )

    report = {
        "dataset": {
            "records": int(len(df)),
            "machines": int(df["machineID"].nunique()),
            "unique_timestamps": int(
                df["timestamp"].nunique()
            ),
        },

        "global_stability": global_results,

        "persistence_baseline": persistence_results,

        "machine_stability": {
            "machines": int(len(machine_results)),

            "mean_cpu_change": float(
                machine_results["util_CPU_mean_change"].mean()
            ),

            "mean_memory_change": float(
                machine_results["util_memory_mean_change"].mean()
            ),

            "median_cpu_change": float(
                machine_results["util_CPU_mean_change"].median()
            ),

            "median_memory_change": float(
                machine_results["util_memory_mean_change"].median()
            ),
        },
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

    print("\nGLOBAL STABILITY RESULTS")

    for target, values in global_results.items():

        print(f"\n{target}")

        for metric, value in values.items():

            print(
                f"  {metric}: {value:.4f}"
            )

    print("\nPERSISTENCE BASELINE")

    for target, values in persistence_results.items():

        print(f"\n{target}")

        print(
            "  MAE:",
            round(values["persistence_mae"], 4),
        )

        print(
            "  RMSE:",
            round(values["persistence_rmse"], 4),
        )

    print("\nMACHINE STABILITY SUMMARY")

    print(
        json.dumps(
            report["machine_stability"],
            indent=4,
        )
    )

    print("\nREPORTS SAVED:")
    print(REPORT_PATH)
    print(CSV_REPORT_PATH)

    print("\nFORECASTING DIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()
