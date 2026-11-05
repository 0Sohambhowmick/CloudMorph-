
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

RISK_PATH = (
    ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
    / "risk_evaluation"
    / "risk_predictions.csv"
)

RAW_DATA_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = (
    ROOT
    / "ml"
    / "models"
    / "extratrees_false_negative_analysis"
)

THRESHOLD = 0.30
HIGH_LABEL = "HIGH_DEMAND"

KEYS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
]

UTILIZATION_COLUMNS = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    if not RISK_PATH.exists():
        raise FileNotFoundError(
            f"Missing risk predictions: {RISK_PATH}"
        )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {RAW_DATA_PATH}"
        )

    predictions = pd.read_csv(RISK_PATH)

    raw = pd.read_csv(RAW_DATA_PATH)

    required_prediction_columns = (
        KEYS
        + [
            "actual_workload",
            "high_demand_probability",
        ]
    )

    missing = (
        set(required_prediction_columns)
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    required_raw_columns = (
        ["machineID", "step"]
        + UTILIZATION_COLUMNS
    )

    missing = (
        set(required_raw_columns)
        - set(raw.columns)
    )

    if missing:
        raise ValueError(
            f"Missing raw dataset columns: {missing}"
        )

    for column in KEYS[1:]:
        predictions[column] = (
            predictions[column].astype(int)
        )

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
    )

    raw["machineID"] = (
        raw["machineID"].astype(str)
    )

    raw["step"] = raw["step"].astype(int)

    raw = raw.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    return predictions, raw


# =========================================================
# IDENTIFY FALSE NEGATIVES
# =========================================================

def identify_false_negatives(predictions):

    predictions = predictions.copy()

    predictions["high_demand_probability"] = (
        pd.to_numeric(
            predictions["high_demand_probability"],
            errors="coerce",
        )
    )

    if not np.isfinite(
        predictions["high_demand_probability"]
    ).all():

        raise ValueError(
            "Invalid probability values found."
        )

    # Recalculate alerts from the validated threshold.
    # Do not trust the saved risk_alert column.
    predictions["recalculated_risk_alert"] = (
        predictions["high_demand_probability"]
        >= THRESHOLD
    ).astype(int)

    predictions["actual_high"] = (
        predictions["actual_workload"]
        == HIGH_LABEL
    ).astype(int)

    predictions["true_positive"] = (
        (predictions["actual_high"] == 1)
        & (predictions["recalculated_risk_alert"] == 1)
    )

    predictions["false_positive"] = (
        (predictions["actual_high"] == 0)
        & (predictions["recalculated_risk_alert"] == 1)
    )

    predictions["false_negative"] = (
        (predictions["actual_high"] == 1)
        & (predictions["recalculated_risk_alert"] == 0)
    )

    false_negatives = predictions[
        predictions["false_negative"]
    ].copy()

    false_positives = predictions[
        predictions["false_positive"]
    ].copy()

    true_positives = predictions[
        predictions["true_positive"]
    ].copy()

    print("\nALERT RECONSTRUCTION")
    print("-" * 60)

    print(
        "Actual HIGH windows:",
        int(predictions["actual_high"].sum())
    )

    print(
        "True positives:",
        len(true_positives)
    )

    print(
        "False positives:",
        len(false_positives)
    )

    print(
        "False negatives:",
        len(false_negatives)
    )

    return (
        predictions,
        false_negatives,
        false_positives,
        true_positives,
    )


# =========================================================
# HISTORICAL AND FUTURE UTILIZATION ANALYSIS
# =========================================================

def analyze_utilization(
    false_negatives,
    raw,
):

    records = []

    raw_grouped = {
        str(machine_id): group.sort_values("step")
        for machine_id, group in raw.groupby("machineID")
    }

    for _, prediction in false_negatives.iterrows():

        machine_id = str(
            prediction["machineID"]
        )

        machine_data = raw_grouped.get(
            machine_id
        )

        if machine_data is None:
            continue

        history = machine_data[
            (
                machine_data["step"]
                >= prediction["history_start_step"]
            )
            &
            (
                machine_data["step"]
                <= prediction["history_end_step"]
            )
        ]

        future = machine_data[
            (
                machine_data["step"]
                >= prediction["target_start_step"]
            )
            &
            (
                machine_data["step"]
                <= prediction["target_end_step"]
            )
        ]

        if history.empty or future.empty:
            continue

        record = {
            "machineID": machine_id,

            "history_start_step": int(
                prediction["history_start_step"]
            ),

            "history_end_step": int(
                prediction["history_end_step"]
            ),

            "target_start_step": int(
                prediction["target_start_step"]
            ),

            "target_end_step": int(
                prediction["target_end_step"]
            ),

            "high_demand_probability": float(
                prediction["high_demand_probability"]
            ),

            "risk_threshold": THRESHOLD,

            "historical_observations": len(history),

            "future_observations": len(future),
        }

        for column in UTILIZATION_COLUMNS:

            history_values = pd.to_numeric(
                history[column],
                errors="coerce",
            ).dropna()

            future_values = pd.to_numeric(
                future[column],
                errors="coerce",
            ).dropna()

            if history_values.empty or future_values.empty:
                continue

            record[f"{column}_history_mean"] = float(
                history_values.mean()
            )

            record[f"{column}_history_max"] = float(
                history_values.max()
            )

            record[f"{column}_recent_mean"] = float(
                history_values.tail(6).mean()
            )

            record[f"{column}_early_mean"] = float(
                history_values.head(6).mean()
            )

            record[f"{column}_recent_change"] = float(
                history_values.tail(6).mean()
                - history_values.head(6).mean()
            )

            record[f"{column}_future_mean"] = float(
                future_values.mean()
            )

            record[f"{column}_future_max"] = float(
                future_values.max()
            )

            record[f"{column}_future_min"] = float(
                future_values.min()
            )

        records.append(record)

    return pd.DataFrame(records)


# =========================================================
# MACHINE-LEVEL SUMMARY
# =========================================================

def create_machine_summary(utilization):

    if utilization.empty:
        return pd.DataFrame()

    summary = (
        utilization.groupby("machineID")
        .agg(
            false_negative_windows=(
                "machineID",
                "size",
            ),

            average_high_probability=(
                "high_demand_probability",
                "mean",
            ),

            average_historical_cpu=(
                "util_CPU_history_mean",
                "mean",
            ),

            average_historical_memory=(
                "util_memory_history_mean",
                "mean",
            ),

            average_future_cpu_peak=(
                "util_CPU_future_max",
                "mean",
            ),

            average_future_memory_peak=(
                "util_memory_future_max",
                "mean",
            ),

            average_memory_recent_change=(
                "util_memory_recent_change",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "false_negative_windows",
            ascending=False,
        )
    )

    return summary


# =========================================================
# COMPARISON WITH DETECTED HIGH WINDOWS
# =========================================================

def create_pattern_comparison(
    utilization,
    predictions,
    raw,
):

    detected = predictions[
        predictions["true_positive"]
    ].copy()

    # Analyze detected windows using the same utilization
    # extraction logic, allowing a like-for-like comparison.
    detected_utilization = analyze_utilization(
        detected,
        raw,
    )

    if utilization.empty or detected_utilization.empty:
        return pd.DataFrame()

    metrics = [
        "util_CPU_history_mean",
        "util_memory_history_mean",
        "util_CPU_recent_change",
        "util_memory_recent_change",
        "util_CPU_future_max",
        "util_memory_future_max",
    ]

    records = []

    for metric in metrics:

        if (
            metric not in utilization.columns
            or metric not in detected_utilization.columns
        ):
            continue

        records.append({

            "metric": metric,

            "false_negative_mean": float(
                utilization[metric].mean()
            ),

            "true_positive_mean": float(
                detected_utilization[metric].mean()
            ),

            "false_negative_median": float(
                utilization[metric].median()
            ),

            "true_positive_median": float(
                detected_utilization[metric].median()
            ),

        })

    return pd.DataFrame(records)


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("EXTRA TREES FALSE NEGATIVE ANALYSIS")
    print("=" * 75)

    predictions, raw = load_data()

    print(
        f"\nPrediction windows: {len(predictions):,}"
    )

    print(
        f"Raw dataset rows: {len(raw):,}"
    )

    (
        predictions,
        false_negatives,
        false_positives,
        true_positives,
    ) = identify_false_negatives(predictions)

    # Save false-negative windows.
    false_negatives.to_csv(
        OUTPUT_DIR / "false_negative_windows.csv",
        index=False,
    )

    # Save false-positive windows.
    false_positives.to_csv(
        OUTPUT_DIR / "false_positive_windows.csv",
        index=False,
    )

    # Save true-positive windows.
    true_positives.to_csv(
        OUTPUT_DIR / "true_positive_windows.csv",
        index=False,
    )

    # Extract historical and future utilization.
    utilization = analyze_utilization(
        false_negatives,
        raw,
    )

    utilization.to_csv(
        OUTPUT_DIR / "false_negative_utilization_analysis.csv",
        index=False,
    )

    # Machine-level analysis.
    machine_summary = create_machine_summary(
        utilization
    )

    machine_summary.to_csv(
        OUTPUT_DIR / "false_negative_machine_summary.csv",
        index=False,
    )

    # Compare missed and detected high-demand windows.
    comparison = create_pattern_comparison(
        utilization,
        predictions,
        raw,
    )

    comparison.to_csv(
        OUTPUT_DIR / "false_negative_pattern_comparison.csv",
        index=False,
    )

    # -----------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------

    total_high = int(
        predictions["actual_high"].sum()
    )

    total_tp = len(true_positives)

    total_fn = len(false_negatives)

    total_fp = len(false_positives)

    summary = {

        "model": "Extra Trees",

        "risk_threshold": THRESHOLD,

        "test_windows": int(len(predictions)),

        "actual_high_windows": total_high,

        "true_positives": int(total_tp),

        "false_positives": int(total_fp),

        "false_negatives": int(total_fn),

        "recall": float(
            total_tp / total_high
            if total_high else 0
        ),

        "precision": float(
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp else 0
        ),

        "false_negative_windows_analyzed": int(
            len(utilization)
        ),

        "machines_with_false_negatives": int(
            utilization["machineID"].nunique()
            if not utilization.empty else 0
        ),

        "important_note": (
            "Risk alerts were recalculated from "
            "high_demand_probability >= 0.30. "
            "The saved risk_alert column was not trusted."
        ),

    }

    with open(
        OUTPUT_DIR / "false_negative_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # CONSOLE REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("FALSE NEGATIVE ANALYSIS RESULTS")
    print("=" * 75)

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print("\nTOP MACHINES WITH MISSED WINDOWS")

    if not machine_summary.empty:

        print(
            machine_summary.head(15).to_string(
                index=False
            )
        )

    print("\nUTILIZATION PATTERN COMPARISON")

    if not comparison.empty:

        print(
            comparison.to_string(
                index=False
            )
        )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
