
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "alibaba_2017"
    / "alibaba-trace-2017"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

DATASETS = {
    "server_usage.csv": [
        "timestamp", "machineID", "util_CPU", "util_memory",
        "util_disk", "load1", "load5", "load15"
    ],
    "container_usage.csv": [
        "ts", "instance_id", "cpu_util", "mem_util",
        "disk_util", "load1", "load5", "load15",
        "avg_cpi", "avg_mpki", "max_cpi", "max_mpki"
    ],
}

UTILIZATION_COLUMNS = {
    "server_usage.csv": ["util_CPU", "util_memory", "util_disk"],
    "container_usage.csv": ["cpu_util", "mem_util", "disk_util"],
}


def profile_file(filename, columns):
    path = DATASET_DIR / filename

    print("\n" + "=" * 65)
    print(f"FULL DATASET PROFILE: {filename}")
    print("=" * 65)

    if not path.exists():
        print(f"File not found: {path}")
        return None

    total_rows = 0
    missing = {column: 0 for column in columns}
    non_finite = {column: 0 for column in columns}
    out_of_range = {
        column: 0
        for column in UTILIZATION_COLUMNS[filename]
    }

    numeric_columns = None
    sums = {}
    minima = {}
    maxima = {}
    valid_counts = {}

    timestamp_column = (
        "timestamp" if filename == "server_usage.csv" else "ts"
    )

    timestamp_min = None
    timestamp_max = None

    reader = pd.read_csv(
        path,
        header=None,
        names=columns,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        if numeric_columns is None:
            numeric_columns = chunk.select_dtypes(
                include="number"
            ).columns.tolist()

            for column in numeric_columns:
                sums[column] = 0.0
                minima[column] = float("inf")
                maxima[column] = float("-inf")
                valid_counts[column] = 0

        missing_values = chunk.isnull().sum()

        for column in columns:
            missing[column] += int(missing_values[column])

        for column in numeric_columns:
            values = pd.to_numeric(
                chunk[column], errors="coerce"
            )

            array = values.to_numpy(dtype="float64")
            finite_mask = np.isfinite(array)
            finite_values = array[finite_mask]

            non_finite[column] += int(
                (~finite_mask & ~np.isnan(array)).sum()
            )

            if len(finite_values):
                sums[column] += float(finite_values.sum())
                minima[column] = min(
                    minima[column], float(finite_values.min())
                )
                maxima[column] = max(
                    maxima[column], float(finite_values.max())
                )
                valid_counts[column] += len(finite_values)

        for column in UTILIZATION_COLUMNS[filename]:
            values = pd.to_numeric(
                chunk[column], errors="coerce"
            )

            out_of_range[column] += int(
                ((values < 0) | (values > 100)).sum()
            )

        timestamps = pd.to_numeric(
            chunk[timestamp_column], errors="coerce"
        ).dropna()

        if not timestamps.empty:
            current_min = float(timestamps.min())
            current_max = float(timestamps.max())

            timestamp_min = (
                current_min if timestamp_min is None
                else min(timestamp_min, current_min)
            )
            timestamp_max = (
                current_max if timestamp_max is None
                else max(timestamp_max, current_max)
            )

        if chunk_number % 10 == 0:
            print(
                f"Processed {total_rows:,} rows..."
            )

    statistics = {}

    for column in numeric_columns:
        count = valid_counts[column]

        statistics[column] = {
            "min": minima[column] if count else None,
            "max": maxima[column] if count else None,
            "mean": sums[column] / count if count else None,
            "valid_count": count,
        }

    result = {
        "filename": filename,
        "total_rows": total_rows,
        "chunk_size": CHUNK_SIZE,
        "missing_values": missing,
        "non_finite_values": non_finite,
        "utilization_out_of_range_0_100": out_of_range,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "statistics": statistics,
    }

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Timestamp range: {timestamp_min} to {timestamp_max}")

    print("\nMissing values:")
    print(json.dumps(missing, indent=2))

    print("\nNon-finite values:")
    print(json.dumps(non_finite, indent=2))

    print("\nUtilization values outside 0-100:")
    print(json.dumps(out_of_range, indent=2))

    print("\nNumeric statistics:")
    print(pd.DataFrame(statistics).T.to_string())

    return result


def main():
    print("CLOUDMORPH - FULL ALIBABA DATASET PROFILING")
    print(f"Reading dataset from: {DATASET_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    results = {}

    for filename, columns in DATASETS.items():
        result = profile_file(filename, columns)

        if result:
            results[filename] = result

    report_path = OUTPUT_DIR / "dataset_profile.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 65)
    print("FULL DATASET PROFILING COMPLETED")
    print(f"Report saved to: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
