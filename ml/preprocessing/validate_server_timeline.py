
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "datasets" / "raw"
    / "alibaba_2017" / "alibaba-trace-2017"
)

METADATA_DIR = ROOT / "datasets" / "metadata"

SERVER_FILE = DATA_DIR / "server_usage.csv"
COVERAGE_FILE = METADATA_DIR / "server_usage_coverage.csv"

COLUMNS = [
    "timestamp",
    "machineID",
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]


def main():
    print("CLOUDMORPH - SERVER TIMELINE VALIDATION")

    # Read server observations
    df = pd.read_csv(
        SERVER_FILE,
        header=None,
        names=COLUMNS,
    )

    coverage = pd.read_csv(COVERAGE_FILE)

    # Identify machines with all 144 observations
    complete_ids = set(
        coverage.loc[
            coverage["complete_history"],
            "entity_id"
        ].astype(int)
    )

    complete_df = df[
        df["machineID"].isin(complete_ids)
    ].copy()

    complete_df = complete_df.sort_values(
        ["machineID", "timestamp"]
    )

    # Check global timestamp intervals
    timestamps = sorted(
        df["timestamp"].dropna().unique()
    )

    intervals = pd.Series(timestamps).diff().dropna()

    interval_report = {
        "unique_timestamps": len(timestamps),
        "minimum_interval_seconds": float(intervals.min()),
        "maximum_interval_seconds": float(intervals.max()),
        "all_intervals_exactly_300_seconds": bool(
            (intervals == 300).all()
        ),
    }

    # Check missing measurements
    measurement_columns = [
        "util_CPU",
        "util_memory",
        "util_disk",
        "load1",
        "load5",
        "load15",
    ]

    missing_report = (
        df[measurement_columns]
        .isnull()
        .sum()
        .to_dict()
    )

    missing_rows = df[
        df[measurement_columns].isnull().any(axis=1)
    ].copy()

    missing_rows["is_complete_history_machine"] = (
        missing_rows["machineID"].isin(complete_ids)
    )

    # Check timestamp gaps for complete-history machines
    complete_df["previous_timestamp"] = (
        complete_df.groupby("machineID")["timestamp"].shift(1)
    )

    complete_df["interval_seconds"] = (
        complete_df["timestamp"]
        - complete_df["previous_timestamp"]
    )

    bad_intervals = complete_df[
        complete_df["interval_seconds"].notna()
        & (complete_df["interval_seconds"] != 300)
    ]

    report = {
        "total_server_rows": len(df),
        "complete_history_machines": len(complete_ids),
        "complete_history_rows": len(complete_df),
        "global_timestamp_check": interval_report,
        "missing_measurements": missing_report,
        "rows_with_missing_measurements": len(missing_rows),
        "missing_rows_in_complete_machines": int(
            missing_rows[
                "is_complete_history_machine"
            ].sum()
        ),
        "unexpected_intervals_in_complete_machines": len(
            bad_intervals
        ),
    }

    print("\nVALIDATION SUMMARY")
    print(json.dumps(report, indent=2))

    print("\nMISSING RECORD DETAILS")
    if missing_rows.empty:
        print("No missing records found.")
    else:
        print(
            missing_rows.to_string(index=False)
        )

    output_path = (
        METADATA_DIR / "server_timeline_validation.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nReport saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
