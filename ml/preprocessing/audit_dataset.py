
from pathlib import Path
import sqlite3
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT / "datasets" / "raw"
    / "alibaba_2017" / "alibaba-trace-2017"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

DATASETS = {
    "server_usage.csv": {
        "columns": [
            "timestamp", "machineID", "util_CPU", "util_memory",
            "util_disk", "load1", "load5", "load15"
        ],
        "time_col": "timestamp",
        "id_col": "machineID",
    },
    "container_usage.csv": {
        "columns": [
            "ts", "instance_id", "cpu_util", "mem_util",
            "disk_util", "load1", "load5", "load15",
            "avg_cpi", "avg_mpki", "max_cpi", "max_mpki"
        ],
        "time_col": "ts",
        "id_col": "instance_id",
    },
}


def audit_file(filename, config):
    path = DATASET_DIR / filename
    time_col = config["time_col"]
    id_col = config["id_col"]
    columns = config["columns"]

    print("\n" + "=" * 65)
    print(f"AUDITING: {filename}")
    print("=" * 65)

    # Temporary on-disk database prevents keeping every key in RAM.
    db_path = OUTPUT_DIR / f"{filename}.audit.sqlite"

    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)

    connection.execute("""
        CREATE TABLE record_keys (
            timestamp REAL NOT NULL,
            entity_id INTEGER NOT NULL,
            PRIMARY KEY (timestamp, entity_id)
        )
    """)

    total_rows = 0
    duplicate_keys = 0
    unique_entities = set()
    timestamp_counts = {}
    timestamp_min = None
    timestamp_max = None
    previous_timestamp = None
    ordering_breaks = 0

    try:
        reader = pd.read_csv(
            path,
            header=None,
            names=columns,
            chunksize=CHUNK_SIZE,
        )

        for chunk in reader:
            total_rows += len(chunk)

            # Rows missing either key cannot be used for time-series identity.
            valid = chunk.dropna(subset=[time_col, id_col]).copy()

            timestamps = pd.to_numeric(
                valid[time_col], errors="coerce"
            )
            entity_ids = pd.to_numeric(
                valid[id_col], errors="coerce"
            )

            valid = valid[
                timestamps.notna()
                & entity_ids.notna()
            ].copy()

            valid[time_col] = timestamps.loc[valid.index]
            valid[id_col] = entity_ids.loc[valid.index]

            valid = valid[
                (valid[id_col] > 0)
                & (valid[id_col] % 1 == 0)
            ]

            unique_entities.update(
                valid[id_col].astype("int64").unique().tolist()
            )

            counts = valid[time_col].value_counts()

            for timestamp, count in counts.items():
                timestamp_counts[float(timestamp)] = (
                    timestamp_counts.get(float(timestamp), 0)
                    + int(count)
                )

            if not valid.empty:
                current_min = float(valid[time_col].min())
                current_max = float(valid[time_col].max())

                timestamp_min = (
                    current_min if timestamp_min is None
                    else min(timestamp_min, current_min)
                )
                timestamp_max = (
                    current_max if timestamp_max is None
                    else max(timestamp_max, current_max)
                )

                # Check whether file order is chronological.
                values = valid[time_col].tolist()

                for value in values:
                    if (
                        previous_timestamp is not None
                        and value < previous_timestamp
                    ):
                        ordering_breaks += 1

                    previous_timestamp = value

                keys = list(zip(
                    valid[time_col].astype(float),
                    valid[id_col].astype("int64"),
                ))

                before = connection.total_changes

                connection.executemany(
                    """
                    INSERT OR IGNORE INTO record_keys
                    (timestamp, entity_id)
                    VALUES (?, ?)
                    """,
                    keys,
                )

                inserted = connection.total_changes - before
                duplicate_keys += len(keys) - inserted

            connection.commit()

        unique_timestamps = len(timestamp_counts)

        report = {
            "filename": filename,
            "total_rows": total_rows,
            "unique_entities": len(unique_entities),
            "unique_timestamps": unique_timestamps,
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
            "duplicate_entity_timestamp_keys": duplicate_keys,
            "chronological_order_breaks": ordering_breaks,
            "records_per_timestamp": {
                "minimum": min(timestamp_counts.values())
                if timestamp_counts else None,
                "maximum": max(timestamp_counts.values())
                if timestamp_counts else None,
                "average": (
                    sum(timestamp_counts.values())
                    / unique_timestamps
                    if unique_timestamps else None
                ),
            },
        }

        print(json.dumps(report, indent=2))
        return report

    finally:
        connection.close()
        if db_path.exists():
            db_path.unlink()


def main():
    results = {}

    print("CLOUDMORPH - DATASET QUALITY AUDIT")

    for filename, config in DATASETS.items():
        results[filename] = audit_file(filename, config)

    output_file = OUTPUT_DIR / "dataset_audit.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nAUDIT COMPLETED")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
