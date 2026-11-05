
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "ml" / "models"

RISK_PATH = (
    MODEL_DIR
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
    MODEL_DIR
    / "extratrees_memory_transition_analysis"
)

THRESHOLD = 0.30
HIGH_LABEL = "HIGH_DEMAND"

UTILIZATION_COLUMNS = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]


# =========================================================
# LOAD AND VALIDATE DATA
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

    prediction_columns = [
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
        "actual_workload",
        "high_demand_probability",
    ]

    raw_columns = [
        "machineID",
        "step",
    ] + UTILIZATION_COLUMNS

    missing_predictions = (
        set(prediction_columns)
        - set(predictions.columns)
    )

    missing_raw = (
        set(raw_columns)
        - set(raw.columns)
    )

    if missing_predictions:
        raise ValueError(
            f"Missing prediction columns: {missing_predictions}"
        )

    if missing_raw:
        raise ValueError(
            f"Missing raw dataset columns: {missing_raw}"
        )

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
    )

    raw["machineID"] = (
        raw["machineID"].astype(str)
    )

    step_columns = [
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
    ]

    for column in step_columns:

        predictions[column] = (
            pd.to_numeric(
                predictions[column],
                errors="raise",
            ).astype(int)
        )

    raw["step"] = (
        pd.to_numeric(
            raw["step"],
            errors="raise",
        ).astype(int)
    )

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

    if (
        (predictions["high_demand_probability"] < 0).any()
        or
        (predictions["high_demand_probability"] > 1).any()
    ):

        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if raw.duplicated(
        ["machineID", "step"]
    ).any():

        raise ValueError(
            "Duplicate machineID and step combinations in raw data."
        )

    raw = raw.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    return predictions, raw


# =========================================================
# RECONSTRUCT EXTRA TREES RESULTS
# =========================================================

def reconstruct_predictions(predictions):

    predictions = predictions.copy()

    predictions["actual_high"] = (
        predictions["actual_workload"]
        == HIGH_LABEL
    ).astype(int)

    # Recalculate alerts using the validated threshold.
    # Do not depend on the saved risk_alert column.
    predictions["recalculated_alert"] = (
        predictions["high_demand_probability"]
        >= THRESHOLD
    ).astype(int)

    predictions["result_category"] = "NOT_HIGH"

    predictions.loc[
        (
            predictions["actual_high"] == 1
        )
        &
        (
            predictions["recalculated_alert"] == 1
        ),
        "result_category",
    ] = "TRUE_POSITIVE"

    predictions.loc[
        (
            predictions["actual_high"] == 1
        )
        &
        (
            predictions["recalculated_alert"] == 0
        ),
        "result_category",
    ] = "FALSE_NEGATIVE"

    return predictions


# =========================================================
# EXTRACT UTILIZATION TRANSITIONS
# =========================================================

def extract_transition_features(
    predictions,
    raw,
):

    records = []

    raw_grouped = {
        str(machine_id): group.sort_values("step")
        for machine_id, group in raw.groupby("machineID")
    }

    for _, row in predictions.iterrows():

        machine_id = str(row["machineID"])

        machine_data = raw_grouped.get(machine_id)

        if machine_data is None:
            continue

        history = machine_data[
            (
                machine_data["step"]
                >= row["history_start_step"]
            )
            &
            (
                machine_data["step"]
                <= row["history_end_step"]
            )
        ]

        future = machine_data[
            (
                machine_data["step"]
                >= row["target_start_step"]
            )
            &
            (
                machine_data["step"]
                <= row["target_end_step"]
            )
        ]

        if history.empty or future.empty:
            continue

        record = {

            "machineID": machine_id,

            "history_start_step": int(
                row["history_start_step"]
            ),

            "history_end_step": int(
                row["history_end_step"]
            ),

            "target_start_step": int(
                row["target_start_step"]
            ),

            "target_end_step": int(
                row["target_end_step"]
            ),

            "actual_workload": row["actual_workload"],

            "result_category": row["result_category"],

            "high_demand_probability": float(
                row["high_demand_probability"]
            ),

            "historical_observations": len(history),

            "future_observations": len(future),

        }

        for column in UTILIZATION_COLUMNS:

            historical_values = pd.to_numeric(
                history[column],
                errors="coerce",
            ).dropna()

            future_values = pd.to_numeric(
                future[column],
                errors="coerce",
            ).dropna()

            if (
                historical_values.empty
                or future_values.empty
            ):
                continue

            early_values = (
                historical_values.head(6)
            )

            recent_values = (
                historical_values.tail(6)
            )

            record[
                f"{column}_history_mean"
            ] = float(
                historical_values.mean()
            )

            record[
                f"{column}_history_max"
            ] = float(
                historical_values.max()
            )

            record[
                f"{column}_early_mean"
            ] = float(
                early_values.mean()
            )

            record[
                f"{column}_recent_mean"
            ] = float(
                recent_values.mean()
            )

            record[
                f"{column}_recent_change"
            ] = float(
                recent_values.mean()
                - early_values.mean()
            )

            record[
                f"{column}_future_mean"
            ] = float(
                future_values.mean()
            )

            record[
                f"{column}_future_max"
            ] = float(
                future_values.max()
            )

            record[
                f"{column}_future_change"
            ] = float(
                future_values.mean()
                - historical_values.mean()
            )

        records.append(record)

    return pd.DataFrame(records)


# =========================================================
# MEMORY TRANSITION SUMMARY
# =========================================================

def create_memory_transition_summary(features):

    if features.empty:
        return pd.DataFrame()

    metrics = [
        "util_memory_history_mean",
        "util_memory_history_max",
        "util_memory_early_mean",
        "util_memory_recent_mean",
        "util_memory_recent_change",
        "util_memory_future_mean",
        "util_memory_future_max",
        "util_memory_future_change",
        "util_CPU_history_mean",
        "util_CPU_recent_change",
        "util_CPU_future_max",
    ]

    records = []

    categories = [
        "FALSE_NEGATIVE",
        "TRUE_POSITIVE",
    ]

    for category in categories:

        group = features[
            features["result_category"] == category
        ]

        if group.empty:
            continue

        for metric in metrics:

            if metric not in group.columns:
                continue

            values = group[metric].dropna()

            if values.empty:
                continue

            records.append({

                "result_category": category,

                "metric": metric,

                "window_count": len(group),

                "mean": float(values.mean()),

                "median": float(values.median()),

                "minimum": float(values.min()),

                "maximum": float(values.max()),

                "standard_deviation": float(
                    values.std()
                ),

            })

    return pd.DataFrame(records)


# =========================================================
# MACHINE-LEVEL TRANSITION SUMMARY
# =========================================================

def create_machine_summary(features):

    if features.empty:
        return pd.DataFrame()

    missed = features[
        features["result_category"] == "FALSE_NEGATIVE"
    ]

    if missed.empty:
        return pd.DataFrame()

    summary = (
        missed.groupby("machineID")
        .agg(

            false_negative_windows=(
                "result_category",
                "size",
            ),

            average_high_probability=(
                "high_demand_probability",
                "mean",
            ),

            average_history_memory=(
                "util_memory_history_mean",
                "mean",
            ),

            average_recent_memory=(
                "util_memory_recent_mean",
                "mean",
            ),

            average_memory_change=(
                "util_memory_recent_change",
                "mean",
            ),

            average_future_memory_peak=(
                "util_memory_future_max",
                "mean",
            ),

            average_history_cpu=(
                "util_CPU_history_mean",
                "mean",
            ),

            average_cpu_change=(
                "util_CPU_recent_change",
                "mean",
            ),

            average_future_cpu_peak=(
                "util_CPU_future_max",
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
# TEMPORAL MISSED-EVENT DISTRIBUTION
# =========================================================

def create_temporal_distribution(features):

    if features.empty:
        return pd.DataFrame()

    summary = (
        features.groupby(
            ["target_start_step", "result_category"]
        )
        .agg(

            window_count=(
                "machineID",
                "size",
            ),

            unique_machines=(
                "machineID",
                "nunique",
            ),

            average_memory_history=(
                "util_memory_history_mean",
                "mean",
            ),

            average_memory_change=(
                "util_memory_recent_change",
                "mean",
            ),

            average_future_memory_peak=(
                "util_memory_future_max",
                "mean",
            ),

            average_cpu_change=(
                "util_CPU_recent_change",
                "mean",
            ),

        )
        .reset_index()
        .sort_values(
            ["target_start_step", "result_category"]
        )
    )

    return summary


# =========================================================
# MEMORY VS CPU ANALYSIS
# =========================================================

def create_memory_cpu_analysis(features):

    if features.empty:
        return pd.DataFrame()

    metrics = [
        "util_memory_history_mean",
        "util_memory_recent_change",
        "util_memory_future_max",
        "util_CPU_history_mean",
        "util_CPU_recent_change",
        "util_CPU_future_max",
    ]

    records = []

    for category in [
        "FALSE_NEGATIVE",
        "TRUE_POSITIVE",
    ]:

        group = features[
            features["result_category"] == category
        ]

        if group.empty:
            continue

        for metric in metrics:

            values = group[metric].dropna()

            if values.empty:
                continue

            records.append({

                "result_category": category,

                "metric": metric,

                "mean": float(values.mean()),

                "median": float(values.median()),

                "sample_count": int(len(values)),

            })

    return pd.DataFrame(records)


# =========================================================
# SUMMARY STATISTICS
# =========================================================

def build_json_summary(
    predictions,
    features,
):

    missed = features[
        features["result_category"] == "FALSE_NEGATIVE"
    ]

    detected = features[
        features["result_category"] == "TRUE_POSITIVE"
    ]

    summary = {

        "model": "Extra Trees",

        "threshold": THRESHOLD,

        "test_windows": int(len(predictions)),

        "actual_high_windows": int(
            predictions["actual_high"].sum()
        ),

        "false_negative_windows": int(
            predictions["result_category"]
            .eq("FALSE_NEGATIVE")
            .sum()
        ),

        "true_positive_windows": int(
            predictions["result_category"]
            .eq("TRUE_POSITIVE")
            .sum()
        ),

        "false_negative_machines": int(
            missed["machineID"].nunique()
            if not missed.empty else 0
        ),

        "true_positive_machines": int(
            detected["machineID"].nunique()
            if not detected.empty else 0
        ),

        "false_negative_average_memory_change": (
            float(
                missed["util_memory_recent_change"].mean()
            )
            if not missed.empty else None
        ),

        "true_positive_average_memory_change": (
            float(
                detected["util_memory_recent_change"].mean()
            )
            if not detected.empty else None
        ),

        "false_negative_average_future_memory_peak": (
            float(
                missed["util_memory_future_max"].mean()
            )
            if not missed.empty else None
        ),

        "true_positive_average_future_memory_peak": (
            float(
                detected["util_memory_future_max"].mean()
            )
            if not detected.empty else None
        ),

        "important_notes": [

            "Alerts were reconstructed using probability >= 0.30.",

            "Only historical observations were used for historical "
            "transition features.",

            "Future utilization is used only for retrospective "
            "analysis, not as model input.",

            "Rows represent overlapping prediction windows, "
            "not independent incidents.",

            "These are historical Alibaba Cluster Trace experiments, "
            "not live cloud provider measurements.",

        ],

    }

    return summary


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("EXTRA TREES MEMORY TRANSITION ANALYSIS")
    print("=" * 75)

    predictions, raw = load_data()

    print(
        f"\nPrediction windows: {len(predictions):,}"
    )

    print(
        f"Raw dataset rows: {len(raw):,}"
    )

    predictions = reconstruct_predictions(
        predictions
    )

    print(
        "\nExtracting historical and future utilization..."
    )

    features = extract_transition_features(
        predictions,
        raw,
    )

    if features.empty:
        raise ValueError(
            "No utilization features could be extracted."
        )

    print(
        f"Analyzed windows: {len(features):,}"
    )

    # -----------------------------------------------------
    # CREATE ANALYSES
    # -----------------------------------------------------

    memory_summary = (
        create_memory_transition_summary(features)
    )

    machine_summary = (
        create_machine_summary(features)
    )

    temporal_summary = (
        create_temporal_distribution(features)
    )

    memory_cpu_summary = (
        create_memory_cpu_analysis(features)
    )

    json_summary = build_json_summary(
        predictions,
        features,
    )

    # -----------------------------------------------------
    # SAVE CSV FILES
    # -----------------------------------------------------

    memory_summary.to_csv(
        OUTPUT_DIR / "memory_transition_summary.csv",
        index=False,
    )

    machine_summary.to_csv(
        OUTPUT_DIR / "machine_memory_transition_summary.csv",
        index=False,
    )

    temporal_summary.to_csv(
        OUTPUT_DIR / "temporal_missed_event_distribution.csv",
        index=False,
    )

    memory_cpu_summary.to_csv(
        OUTPUT_DIR / "memory_vs_cpu_analysis.csv",
        index=False,
    )

    features.to_csv(
        OUTPUT_DIR / "all_window_transition_features.csv",
        index=False,
    )

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    with open(
        OUTPUT_DIR / "memory_transition_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # CONSOLE REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("MEMORY TRANSITION SUMMARY")
    print("=" * 75)

    print(
        memory_summary.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("TOP MACHINES WITH MISSED WINDOWS")
    print("=" * 75)

    if not machine_summary.empty:

        print(
            machine_summary.head(15).to_string(
                index=False
            )
        )

    print("\n" + "=" * 75)
    print("TEMPORAL MISSED-EVENT DISTRIBUTION")
    print("=" * 75)

    missed_temporal = temporal_summary[
        temporal_summary["result_category"]
        == "FALSE_NEGATIVE"
    ]

    print(
        missed_temporal.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("MEMORY VS CPU ANALYSIS")
    print("=" * 75)

    print(
        memory_cpu_summary.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(
        json.dumps(
            json_summary,
            indent=4,
        )
    )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
