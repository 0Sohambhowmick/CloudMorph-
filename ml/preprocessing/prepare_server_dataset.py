
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    ROOT / "datasets" / "raw"
    / "alibaba_2017" / "alibaba-trace-2017"
)

METADATA_DIR = ROOT / "datasets" / "metadata"
PROCESSED_DIR = ROOT / "datasets" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = RAW_DIR / "server_usage.csv"
COVERAGE_FILE = METADATA_DIR / "server_usage_coverage.csv"

OUTPUT_FILE = PROCESSED_DIR / "server_usage_clean.csv"
MANIFEST_FILE = METADATA_DIR / "server_dataset_manifest.json"

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

FEATURE_COLUMNS = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

EXPECTED_OBSERVATIONS = 144
EXPECTED_INTERVAL_SECONDS = 300


def main():
    print("=" * 65)
    print("CLOUDMORPH - SERVER DATASET PREPARATION")
    print("=" * 65)

    # Load only machines with complete historical coverage.
    coverage = pd.read_csv(COVERAGE_FILE)

    complete_machines = set(
        coverage.loc[
            coverage["complete_history"],
            "entity_id"
        ].astype(int)
    )

    print(f"\nComplete-history machines: {len(complete_machines)}")

    # Load raw server measurements.
    df = pd.read_csv(
        SOURCE_FILE,
        header=None,
        names=COLUMNS,
    )

    print(f"Original records: {len(df):,}")

    # Select complete-history machines.
    df = df[
        df["machineID"].isin(complete_machines)
    ].copy()

    # Check required fields.
    if df[["timestamp", "machineID"] + FEATURE_COLUMNS].isnull().any().any():
        raise ValueError(
            "Missing values found in selected server records."
        )

    # Validate numerical values.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    # Validate utilization values.
    utilization_columns = [
        "util_CPU",
        "util_memory",
        "util_disk",
    ]

    for column in utilization_columns:
        if not df[column].between(0, 100).all():
            raise ValueError(
                f"Invalid utilization values in {column}."
            )

    # Sort chronologically for each machine.
    df = df.sort_values(
        ["machineID", "timestamp"]
    ).reset_index(drop=True)

    # Validate each machine has exactly 144 records.
    counts = df.groupby("machineID").size()

    if not (counts == EXPECTED_OBSERVATIONS).all():
        raise ValueError(
            "Some selected machines do not have exactly 144 records."
        )

    # Validate five-minute interval for every machine.
    intervals = (
        df.groupby("machineID")["timestamp"]
        .diff()
        .dropna()
    )

    if not (intervals == EXPECTED_INTERVAL_SECONDS).all():
        raise ValueError(
            "Unexpected timestamp intervals found."
        )

    # Add elapsed time within each machine's history.
    df["step"] = (
        df.groupby("machineID").cumcount()
    )

    # Keep a clear, consistent column order.
    final_columns = [
        "machineID",
        "timestamp",
        "step",
        "util_CPU",
        "util_memory",
        "util_disk",
        "load1",
        "load5",
        "load15",
    ]

    df = df[final_columns]

    # Save the processed dataset.
    df.to_csv(OUTPUT_FILE, index=False)

    manifest = {
        "dataset": "Alibaba Cluster Trace 2017",
        "source_file": "server_usage.csv",
        "processed_file": str(OUTPUT_FILE),
        "selection_rule": (
            "Machines with exactly 144 valid observations "
            "and 300-second intervals"
        ),
        "machine_count": int(df["machineID"].nunique()),
        "record_count": int(len(df)),
        "observations_per_machine": EXPECTED_OBSERVATIONS,
        "interval_seconds": EXPECTED_INTERVAL_SECONDS,
        "features": final_columns,
        "missing_values": int(df.isnull().sum().sum()),
        "timestamp_min": int(df["timestamp"].min()),
        "timestamp_max": int(df["timestamp"].max()),
    }

    with open(
        MANIFEST_FILE, "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=4)

    print("\nDATASET PREPARATION SUCCESSFUL")
    print(json.dumps(manifest, indent=2))

    print("\nCPU and memory summary:")
    print(
        df[["util_CPU", "util_memory"]]
        .describe()
        .round(3)
        .to_string()
    )

    print("\nProcessed dataset saved:")
    print(OUTPUT_FILE)

    print("\nDataset manifest saved:")
    print(MANIFEST_FILE)


if __name__ == "__main__":
    main()
