
"""
CloudMorph - Cross-Machine Spike Analysis

Checks whether the workload spikes observed around steps 103-108
are isolated or appear across many machines.

Read-only analysis. Does not modify datasets or models.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

START_STEP = 98
END_STEP = 110


def main():

    print("Loading original workload dataset...")

    df = pd.read_csv(RAW_PATH)

    required_columns = [
        "machineID",
        "step",
        "util_CPU",
        "util_memory",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    df["machineID"] = df["machineID"].astype(str)

    for column in ["step", "util_CPU", "util_memory"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = df[
        df["step"].between(START_STEP, END_STEP)
    ].copy()

    if df.empty:
        raise ValueError("No records found in the requested step range.")

    print("\nCROSS-MACHINE WORKLOAD STATISTICS")
    print("=" * 105)

    summary = (
        df.groupby("step")
        .agg(
            machines=("machineID", "nunique"),
            cpu_mean=("util_CPU", "mean"),
            cpu_median=("util_CPU", "median"),
            cpu_p90=("util_CPU", lambda x: x.quantile(0.90)),
            cpu_p95=("util_CPU", lambda x: x.quantile(0.95)),
            memory_mean=("util_memory", "mean"),
            memory_median=("util_memory", "median"),
            memory_p90=("util_memory", lambda x: x.quantile(0.90)),
            memory_p95=("util_memory", lambda x: x.quantile(0.95)),
            cpu_ge_70=("util_CPU", lambda x: (x >= 70).sum()),
            memory_ge_70=("util_memory", lambda x: (x >= 70).sum()),
            cpu_ge_85=("util_CPU", lambda x: (x >= 85).sum()),
            memory_ge_85=("util_memory", lambda x: (x >= 85).sum()),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                column: "{:.2f}".format
                for column in summary.select_dtypes(
                    include="float"
                ).columns
            },
        )
    )

    print("\nMACHINES WITH HIGH CPU OR MEMORY")
    print("=" * 80)

    high_rows = df[
        (df["util_CPU"] >= 70)
        | (df["util_memory"] >= 70)
    ].copy()

    if high_rows.empty:
        print("No machines crossed 70% utilization.")
    else:
        print(
            high_rows.sort_values(
                ["step", "util_memory", "util_CPU"],
                ascending=[True, False, False],
            )[
                [
                    "step",
                    "machineID",
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

    print("\nBASELINE VS SPIKE PERIOD")
    print("=" * 80)

    baseline = df[df["step"].between(98, 102)]
    spike_period = df[df["step"].between(103, 108)]

    for label, period in [
        ("BASELINE (98-102)", baseline),
        ("SPIKE PERIOD (103-108)", spike_period),
    ]:

        print(f"\n{label}")

        print(
            f"Records: {len(period)} | "
            f"Machines: {period['machineID'].nunique()}"
        )

        print(
            f"CPU mean: {period['util_CPU'].mean():.2f}% | "
            f"CPU median: {period['util_CPU'].median():.2f}% | "
            f"CPU >=70%: {(period['util_CPU'] >= 70).sum()}"
        )

        print(
            f"Memory mean: {period['util_memory'].mean():.2f}% | "
            f"Memory median: {period['util_memory'].median():.2f}% | "
            f"Memory >=70%: {(period['util_memory'] >= 70).sum()}"
        )

    print("\nAnalysis completed.")
    print("No model or dataset files were modified.")


if __name__ == "__main__":
    main()
