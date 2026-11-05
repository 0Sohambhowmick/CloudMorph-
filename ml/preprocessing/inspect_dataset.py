
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "alibaba_2017"
    / "alibaba-trace-2017"
)

# Official Alibaba 2017 column definitions
DATASETS = {
    "server_usage.csv": [
        "timestamp",
        "machineID",
        "util_CPU",
        "util_memory",
        "util_disk",
        "load1",
        "load5",
        "load15",
    ],
    "container_usage.csv": [
        "ts",
        "instance_id",
        "cpu_util",
        "mem_util",
        "disk_util",
        "load1",
        "load5",
        "load15",
        "avg_cpi",
        "avg_mpki",
        "max_cpi",
        "max_mpki",
    ],
    "batch_instance.csv": [
        "start_timestamp",
        "end_timestamp",
        "job_id",
        "task_id",
        "machineID",
        "status",
        "seq_no",
        "total_seq_no",
        "real_cpu_max",
        "real_cpu_avg",
        "real_mem_max",
        "real_mem_avg",
    ],
}


def inspect_dataset(filename, columns):
    file_path = DATASET_DIR / filename

    print("\n" + "=" * 65)
    print(f"DATASET: {filename}")
    print("=" * 65)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    # Read only the first 10,000 rows for initial inspection
    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
        nrows=10000,
    )

    print(f"Rows inspected: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nStatistical summary:")
    print(df.describe(include="all").to_string())


def main():
    print("CLOUDMORPH - ALIBABA DATASET INSPECTION")
    print(f"Dataset directory: {DATASET_DIR}")

    for filename, columns in DATASETS.items():
        inspect_dataset(filename, columns)

    print("\n" + "=" * 65)
    print("INITIAL DATASET INSPECTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
