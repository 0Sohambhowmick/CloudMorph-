
"""
CloudMorph - Resource Demand Regression Error Analysis

Analyzes saved test predictions by:
- Actual utilization range
- Individual machine
- Large prediction errors
- Prediction bias

Does not retrain or modify models.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "resource_demand_regressor"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "resource_demand_regressor"
)

RANGE_REPORT_PATH = OUTPUT_DIR / "utilization_range_analysis.csv"
MACHINE_REPORT_PATH = OUTPUT_DIR / "machine_error_analysis.csv"
SUMMARY_PATH = OUTPUT_DIR / "error_analysis_summary.json"


def analyze_target(df, actual_column, predicted_column, error_column):

    actual = df[actual_column]
    predicted = df[predicted_column]
    absolute_error = df[error_column]

    signed_error = predicted - actual

    ranges = [
        ("LOW", 0, 35),
        ("NORMAL", 35, 70),
        ("ELEVATED", 70, 85),
        ("HIGH", 85, 101),
    ]

    range_results = []

    for name, lower, upper in ranges:

        mask = (actual >= lower) & (actual < upper)

        if mask.sum() == 0:
            continue

        range_results.append({
            "workload_range": name,
            "actual_min": lower,
            "actual_max_exclusive": upper,
            "samples": int(mask.sum()),
            "mae": float(absolute_error[mask].mean()),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        (actual[mask] - predicted[mask]) ** 2
                    )
                )
            ),
            "mean_signed_error": float(signed_error[mask].mean()),
            "errors_above_5_points": int(
                (absolute_error[mask] > 5).sum()
            ),
            "errors_above_10_points": int(
                (absolute_error[mask] > 10).sum()
            ),
        })

    range_df = pd.DataFrame(range_results)

    machine_df = pd.DataFrame({
        "machineID": df["machineID"],
        "actual": actual,
        "predicted": predicted,
        "absolute_error": absolute_error,
        "signed_error": signed_error,
    })

    machine_report = (
        machine_df
        .groupby("machineID")
        .agg(
            samples=("absolute_error", "count"),
            mae=("absolute_error", "mean"),
            max_error=("absolute_error", "max"),
            mean_signed_error=("signed_error", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    summary = {
        "samples": int(len(df)),
        "overall_mae": float(absolute_error.mean()),
        "overall_rmse": float(
            np.sqrt(np.mean((actual - predicted) ** 2))
        ),
        "mean_signed_error": float(signed_error.mean()),
        "errors_above_5_points": int(
            (absolute_error > 5).sum()
        ),
        "errors_above_10_points": int(
            (absolute_error > 10).sum()
        ),
        "percentage_errors_above_5_points": float(
            (absolute_error > 5).mean() * 100
        ),
        "percentage_errors_above_10_points": float(
            (absolute_error > 10).mean() * 100
        ),
        "range_analysis": range_results,
        "worst_10_machines": (
            machine_report.head(10).to_dict(orient="records")
        ),
    }

    return range_df, machine_report, summary


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "machineID",
        "actual_peak_cpu",
        "predicted_peak_cpu",
        "cpu_absolute_error",
        "actual_peak_memory",
        "predicted_peak_memory",
        "memory_absolute_error",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    numeric_columns = required_columns[1:]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Prediction file contains missing or invalid values."
        )

    # Confirm saved absolute errors are consistent.

    cpu_error = np.abs(
        df["actual_peak_cpu"] - df["predicted_peak_cpu"]
    )

    memory_error = np.abs(
        df["actual_peak_memory"] - df["predicted_peak_memory"]
    )

    if not np.allclose(
        cpu_error,
        df["cpu_absolute_error"],
        atol=1e-5,
    ):
        raise ValueError("CPU absolute error mismatch.")

    if not np.allclose(
        memory_error,
        df["memory_absolute_error"],
        atol=1e-5,
    ):
        raise ValueError("Memory absolute error mismatch.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CPU analysis.

    cpu_ranges, cpu_machines, cpu_summary = analyze_target(
        df,
        "actual_peak_cpu",
        "predicted_peak_cpu",
        "cpu_absolute_error",
    )

    # Memory analysis.

    memory_ranges, memory_machines, memory_summary = analyze_target(
        df,
        "actual_peak_memory",
        "predicted_peak_memory",
        "memory_absolute_error",
    )

    # Save range analysis.

    range_report = pd.concat(
        [
            cpu_ranges.assign(resource="CPU"),
            memory_ranges.assign(resource="MEMORY"),
        ],
        ignore_index=True,
    )

    range_report.to_csv(
        RANGE_REPORT_PATH,
        index=False,
    )

    # Save machine analysis.

    machine_report = pd.concat(
        [
            cpu_machines.assign(resource="CPU"),
            memory_machines.assign(resource="MEMORY"),
        ],
        ignore_index=True,
    )

    machine_report.to_csv(
        MACHINE_REPORT_PATH,
        index=False,
    )

    # Save summary.

    summary = {
        "project": "CloudMorph",
        "task": "Resource demand regression error analysis",
        "test_samples": int(len(df)),
        "unique_test_machines": int(df["machineID"].nunique()),
        "cpu": cpu_summary,
        "memory": memory_summary,
        "notes": [
            "Analysis uses saved unseen-machine test predictions.",
            "Workload ranges are based on actual future peak utilization.",
            "Positive signed error means overprediction.",
            "Negative signed error means underprediction.",
            "This analysis does not establish safe resizing outcomes.",
        ],
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # Print useful results.

    for name, result in [
        ("CPU", cpu_summary),
        ("MEMORY", memory_summary),
    ]:

        print("\n" + "=" * 60)
        print(f"{name} ERROR ANALYSIS")
        print("=" * 60)

        print(f"Overall MAE: {result['overall_mae']:.4f}")
        print(f"Overall RMSE: {result['overall_rmse']:.4f}")
        print(
            f"Mean signed error: "
            f"{result['mean_signed_error']:.4f}"
        )

        print(
            f"Errors above 5 points: "
            f"{result['percentage_errors_above_5_points']:.2f}%"
        )

        print(
            f"Errors above 10 points: "
            f"{result['percentage_errors_above_10_points']:.2f}%"
        )

        print("\nPerformance by actual utilization range:")

        print(
            pd.DataFrame(
                result["range_analysis"]
            ).to_string(index=False)
        )

        print("\n10 machines with highest MAE:")

        print(
            pd.DataFrame(
                result["worst_10_machines"]
            ).to_string(index=False)
        )

    print("\nAnalysis completed.")
    print(f"Range report: {RANGE_REPORT_PATH}")
    print(f"Machine report: {MACHINE_REPORT_PATH}")
    print(f"Summary report: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
