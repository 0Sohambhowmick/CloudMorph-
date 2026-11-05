
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "datasets/processed/server_usage_clean.csv"

LSTM_FILE = (
    ROOT
    / "ml/models/lstm_sequence_classifier/test_predictions.csv"
)

OUTPUT_DIR = ROOT / "ml/models/lstm_false_alert_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]


def main():
    logger.info("Loading raw workload data and LSTM predictions...")

    raw = pd.read_csv(RAW_FILE)
    predictions = pd.read_csv(LSTM_FILE)

    for column in [
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_end_step",
    ]:
        predictions[column] = predictions[column].astype(int)

    raw["machineID"] = raw["machineID"].astype(int)
    raw["step"] = raw["step"].astype(int)

    logger.info("Identifying all LSTM false-positive alerts...")

    false_alerts = predictions[
        (predictions["risk_alert"] == "HIGH_RISK")
        & (predictions["target"] != "HIGH_DEMAND")
    ].copy()

    if false_alerts.empty:
        raise ValueError("No LSTM false-positive alerts found.")

    logger.info("Total false positives: %s", len(false_alerts))

    raw_groups = {
        int(machine): group.set_index("step").sort_index()
        for machine, group in raw.groupby("machineID")
    }

    records = []

    logger.info("Reconstructing historical and future windows...")

    for _, prediction in false_alerts.iterrows():
        machine = int(prediction["machineID"])
        start = int(prediction["history_start_step"])
        history_end = int(prediction["history_end_step"])
        target_end = int(prediction["target_end_step"])

        machine_data = raw_groups[machine]

        history = machine_data.loc[start:history_end]
        future = machine_data.loc[history_end + 1:target_end]

        if len(history) != 12 or len(future) != 6:
            logger.warning(
                "Skipping incomplete window: machine=%s start=%s",
                machine,
                start,
            )
            continue

        record = {
            "machineID": machine,
            "history_start_step": start,
            "history_end_step": history_end,
            "target_end_step": target_end,
            "actual_workload": prediction["target"],
            "predicted_workload": prediction["predicted_workload"],
            "high_demand_probability": float(
                prediction["high_demand_probability"]
            ),
            "risk_threshold": float(prediction["risk_threshold"]),
        }

        for feature in FEATURES:
            values = history[feature].to_numpy(dtype=float)

            record[f"{feature}_current"] = float(values[-1])
            record[f"{feature}_mean"] = float(values.mean())
            record[f"{feature}_min"] = float(values.min())
            record[f"{feature}_max"] = float(values.max())
            record[f"{feature}_recent_mean"] = float(
                values[-3:].mean()
            )
            record[f"{feature}_previous_mean"] = float(
                values[-6:-3].mean()
            )
            record[f"{feature}_trend"] = float(
                np.polyfit(np.arange(len(values)), values, 1)[0]
            )

            record[f"{feature}_future_max"] = float(
                future[feature].max()
            )
            record[f"{feature}_future_mean"] = float(
                future[feature].mean()
            )

        record["memory_recent_change"] = (
            record["util_memory_recent_mean"]
            - record["util_memory_previous_mean"]
        )

        record["cpu_recent_change"] = (
            record["util_CPU_recent_mean"]
            - record["util_CPU_previous_mean"]
        )

        record["memory_future_peak_change"] = (
            record["util_memory_future_max"]
            - record["util_memory_current"]
        )

        record["cpu_future_peak_change"] = (
            record["util_CPU_future_max"]
            - record["util_CPU_current"]
        )

        records.append(record)

    cases = pd.DataFrame(records)

    if cases.empty:
        raise ValueError("No complete false-alert windows reconstructed.")

    # --------------------------------------------------
    # 1. Save individual false-positive cases
    # --------------------------------------------------

    cases.to_csv(
        OUTPUT_DIR / "lstm_false_alert_cases.csv",
        index=False,
    )

    # --------------------------------------------------
    # 2. Machine-level analysis
    # --------------------------------------------------

    machine_summary = (
        cases.groupby("machineID")
        .agg(
            false_alerts=("machineID", "size"),
            average_probability=(
                "high_demand_probability", "mean"
            ),
            maximum_probability=(
                "high_demand_probability", "max"
            ),
            actual_workload=("actual_workload", "first"),
            average_memory=("util_memory_mean", "mean"),
            average_cpu=("util_CPU_mean", "mean"),
            average_memory_trend=("util_memory_trend", "mean"),
            first_history_step=("history_start_step", "min"),
            last_history_step=("history_start_step", "max"),
        )
        .reset_index()
        .sort_values("false_alerts", ascending=False)
    )

    machine_summary.to_csv(
        OUTPUT_DIR / "lstm_false_alert_machine_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. Compare CPU and memory characteristics
    # --------------------------------------------------

    feature_columns = [
        "util_CPU_current",
        "util_CPU_mean",
        "util_CPU_recent_mean",
        "util_CPU_trend",
        "util_CPU_future_max",
        "util_memory_current",
        "util_memory_mean",
        "util_memory_recent_mean",
        "util_memory_trend",
        "util_memory_future_max",
        "memory_recent_change",
        "cpu_recent_change",
        "memory_future_peak_change",
        "cpu_future_peak_change",
        "high_demand_probability",
    ]

    feature_summary = (
        cases[feature_columns]
        .agg(["mean", "median", "min", "max"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    feature_summary.to_csv(
        OUTPUT_DIR / "lstm_false_alert_feature_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # 4. Probability bands
    # --------------------------------------------------

    cases["probability_band"] = pd.cut(
        cases["high_demand_probability"],
        bins=[0, 0.2, 0.5, 0.8, 0.95, 1.000001],
        labels=[
            "0-20%",
            "20-50%",
            "50-80%",
            "80-95%",
            "95-100%",
        ],
        include_lowest=True,
    )

    probability_summary = (
        cases.groupby("probability_band", observed=False)
        .agg(
            false_alerts=("machineID", "size"),
            average_probability=(
                "high_demand_probability", "mean"
            ),
            average_memory=("util_memory_mean", "mean"),
            average_cpu=("util_CPU_mean", "mean"),
        )
        .reset_index()
    )

    probability_summary.to_csv(
        OUTPUT_DIR / "lstm_false_alert_probability_bands.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Consecutive false-alert windows
    # --------------------------------------------------

    cases = cases.sort_values(
        ["machineID", "history_start_step"]
    )

    cases["previous_start"] = (
        cases.groupby("machineID")["history_start_step"].shift(1)
    )

    cases["new_alert_sequence"] = (
        cases["previous_start"].isna()
        | (
            cases["history_start_step"]
            - cases["previous_start"]
            != 1
        )
    )

    cases["alert_sequence_id"] = (
        cases.groupby("machineID")["new_alert_sequence"]
        .cumsum()
    )

    sequence_summary = (
        cases.groupby(["machineID", "alert_sequence_id"])
        .agg(
            consecutive_alert_windows=("machineID", "size"),
            first_step=("history_start_step", "min"),
            last_step=("history_start_step", "max"),
            maximum_probability=(
                "high_demand_probability", "max"
            ),
            actual_workload=("actual_workload", "first"),
        )
        .reset_index()
        .sort_values(
            "consecutive_alert_windows",
            ascending=False,
        )
    )

    sequence_summary.to_csv(
        OUTPUT_DIR / "lstm_false_alert_sequences.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. Final report
    # --------------------------------------------------

    report = {
        "false_positive_windows": int(len(cases)),
        "unique_machines": int(cases["machineID"].nunique()),
        "average_high_probability": float(
            cases["high_demand_probability"].mean()
        ),
        "median_high_probability": float(
            cases["high_demand_probability"].median()
        ),
        "probability_above_95_percent": int(
            (cases["high_demand_probability"] >= 0.95).sum()
        ),
        "average_historical_memory": float(
            cases["util_memory_mean"].mean()
        ),
        "average_historical_cpu": float(
            cases["util_CPU_mean"].mean()
        ),
        "average_future_memory_peak": float(
            cases["util_memory_future_max"].mean()
        ),
        "average_future_cpu_peak": float(
            cases["util_CPU_future_max"].mean()
        ),
        "average_memory_recent_change": float(
            cases["memory_recent_change"].mean()
        ),
        "average_cpu_recent_change": float(
            cases["cpu_recent_change"].mean()
        ),
        "alert_sequences": int(
            len(sequence_summary)
        ),
        "longest_consecutive_alert_sequence": int(
            sequence_summary["consecutive_alert_windows"].max()
        ),
    }

    with open(
        OUTPUT_DIR / "lstm_false_alert_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    logger.info("LSTM FALSE-ALERT ANALYSIS COMPLETED")
    logger.info(
        "False-positive windows: %s",
        report["false_positive_windows"],
    )
    logger.info(
        "Unique machines: %s",
        report["unique_machines"],
    )
    logger.info(
        "Average probability: %.4f",
        report["average_high_probability"],
    )
    logger.info(
        "Average historical memory: %.2f",
        report["average_historical_memory"],
    )
    logger.info(
        "Average future memory peak: %.2f",
        report["average_future_memory_peak"],
    )
    logger.info(
        "Alert sequences: %s",
        report["alert_sequences"],
    )
    logger.info(
        "Longest alert sequence: %s",
        report["longest_consecutive_alert_sequence"],
    )
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
