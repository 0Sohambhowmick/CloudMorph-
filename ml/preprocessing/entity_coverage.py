
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


def analyze_coverage(filename, config):
    path = DATASET_DIR / filename
    time_col = config["time_col"]
    id_col = config["id_col"]

    db_path = OUTPUT_DIR / "coverage_temp.sqlite"

    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)

    connection.execute("""
        CREATE TABLE observations (
            entity_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            PRIMARY KEY (entity_id, timestamp)
        )
    """)

    try:
        print("\n" + "=" * 65)
        print(f"ENTITY COVERAGE: {filename}")
        print("=" * 65)

        total_rows = 0

        reader = pd.read_csv(
            path,
            header=None,
            names=config["columns"],
            chunksize=CHUNK_SIZE,
        )

        for chunk in reader:
            total_rows += len(chunk)

            valid = chunk[[id_col, time_col]].copy()

            valid[id_col] = pd.to_numeric(
                valid[id_col], errors="coerce"
            )
            valid[time_col] = pd.to_numeric(
                valid[time_col], errors="coerce"
            )

            valid = valid.dropna()

            valid = valid[
                (valid[id_col] > 0)
                & (valid[id_col] % 1 == 0)
                & (valid[time_col] % 1 == 0)
            ]

            records = [
                (int(entity), int(timestamp))
                for entity, timestamp in zip(
                    valid[id_col], valid[time_col]
                )
            ]

            connection.executemany(
                """
                INSERT OR IGNORE INTO observations
                (entity_id, timestamp)
                VALUES (?, ?)
                """,
                records,
            )

            connection.commit()

        query = """
            SELECT
                entity_id,
                COUNT(*) AS observation_count,
                MIN(timestamp) AS first_timestamp,
                MAX(timestamp) AS last_timestamp
            FROM observations
            GROUP BY entity_id
        """

        coverage = pd.read_sql_query(query, connection)

        # Calculate timestamp gaps for each entity.
        gap_query = """
            WITH ordered AS (
                SELECT
                    entity_id,
                    timestamp,
                    LAG(timestamp) OVER (
                        PARTITION BY entity_id
                        ORDER BY timestamp
                    ) AS previous_timestamp
                FROM observations
            )
            SELECT
                entity_id,
                SUM(
                    CASE
                        WHEN previous_timestamp IS NOT NULL
                        AND timestamp - previous_timestamp > 300
                        THEN 1 ELSE 0
                    END
                ) AS gaps_over_5_minutes
            FROM ordered
            GROUP BY entity_id
        """

        gaps = pd.read_sql_query(gap_query, connection)

        coverage = coverage.merge(
            gaps, on="entity_id", how="left"
        )

        coverage["gaps_over_5_minutes"] = (
            coverage["gaps_over_5_minutes"].fillna(0).astype(int)
        )

        expected_observations = 144

        coverage["coverage_percent"] = (
            coverage["observation_count"]
            / expected_observations
            * 100
        ).clip(upper=100)

        coverage["coverage_percent"] = (
            coverage["coverage_percent"].round(2)
        )

        coverage["complete_history"] = (
            coverage["observation_count"]
            == expected_observations
        )

        coverage = coverage.sort_values(
            "observation_count", ascending=False
        )

        output_name = filename.replace(".csv", "_coverage.csv")
        output_path = OUTPUT_DIR / output_name

        coverage.to_csv(output_path, index=False)

        summary = {
            "dataset": filename,
            "total_source_rows": total_rows,
            "unique_entities": int(len(coverage)),
            "expected_observations_per_entity": expected_observations,
            "complete_history_entities": int(
                coverage["complete_history"].sum()
            ),
            "entities_with_incomplete_history": int(
                (~coverage["complete_history"]).sum()
            ),
            "average_observations_per_entity": round(
                float(coverage["observation_count"].mean()), 2
            ),
            "average_coverage_percent": round(
                float(coverage["coverage_percent"].mean()), 2
            ),
            "entities_with_gaps_over_5_minutes": int(
                (coverage["gaps_over_5_minutes"] > 0).sum()
            ),
            "coverage_report": str(output_path),
        }

        print(json.dumps(summary, indent=2))

        print("\nTop 10 entities by historical observations:")
        print(coverage.head(10).to_string(index=False))

        print("\nCoverage report saved:")
        print(output_path)

        return summary

    finally:
        connection.close()

        if db_path.exists():
            db_path.unlink()


def main():
    results = {}

    print("CLOUDMORPH - ENTITY HISTORY COVERAGE ANALYSIS")

    for filename, config in DATASETS.items():
        results[filename] = analyze_coverage(
            filename, config
        )

    report_path = OUTPUT_DIR / "entity_coverage_summary.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nENTITY COVERAGE ANALYSIS COMPLETED")
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
