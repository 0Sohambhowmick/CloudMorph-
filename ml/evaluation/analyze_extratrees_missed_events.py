
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
    / "extratrees_missed_event_analysis"
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
# LOAD DATA
# =========================================================

def load_data():

    if not RISK_PATH.exists():
        raise FileNotFoundError(RISK_PATH)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(RAW_DATA_PATH)

    predictions = pd.read_csv(RISK_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)

    required_prediction_columns = [
        "machineID",
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
        "actual_workload",
        "high_demand_probability",
    ]

    missing = (
        set(required_prediction_columns)
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    required_raw_columns = [
        "machineID",
        "step",
    ] + UTILIZATION_COLUMNS

    missing = (
        set(required_raw_columns)
        - set(raw.columns)
    )

    if missing:
        raise ValueError(
            f"Missing raw columns: {missing}"
        )

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
    )

    raw["machineID"] = (
        raw["machineID"].astype(str)
    )

    for column in [
        "history_start_step",
        "history_end_step",
        "target_start_step",
        "target_end_step",
    ]:

        predictions[column] = (
            predictions[column].astype(int)
        )

    raw["step"] = raw["step"].astype(int)

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
            "Invalid high-demand probabilities."
        )

    if (
        (predictions["high_demand_probability"] < 0).any()
        or
        (predictions["high_demand_probability"] > 1).any()
    ):

        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    raw = raw.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    return predictions, raw


# =========================================================
# RECONSTRUCT ALERTS
# =========================================================

def reconstruct_predictions(predictions):

    predictions = predictions.copy()

    predictions["actual_high"] = (
        predictions["actual_workload"]
        == HIGH_LABEL
    ).astype(int)

    # Recalculate alerts from the selected threshold.
    predictions["risk_alert"] = (
        predictions["high_demand_probability"]
        >= THRESHOLD
    ).astype(int)

    predictions["false_negative"] = (
        (predictions["actual_high"] == 1)
        & (predictions["risk_alert"] == 0)
    )

    predictions["true_positive"] = (
        (predictions["actual_high"] == 1)
        & (predictions["risk_alert"] == 1)
    )

    return predictions


# =========================================================
# BUILD MISSED EVENTS
# =========================================================

def build_missed_events(predictions):

    missed = predictions[
        predictions["false_negative"]
    ].copy()

    if missed.empty:
        return pd.DataFrame()

    missed = missed.sort_values(
        ["machineID", "target_start_step"]
    ).reset_index(drop=True)

    events = []

    event_id = 0

    for machine_id, group in missed.groupby(
        "machineID",
        sort=False,
    ):

        group = group.sort_values(
            "target_start_step"
        )

        current_event = []
        previous_end = None

        for _, row in group.iterrows():

            start = int(row["target_start_step"])
            end = int(row["target_end_step"])

            # Merge overlapping or directly adjacent
            # future target windows into one event.
            if (
                current_event
                and previous_end is not None
                and start > previous_end + 1
            ):

                event_id += 1

                events.append(
                    create_event_record(
                        event_id,
                        machine_id,
                        current_event,
                    )
                )

                current_event = []

            current_event.append(row.to_dict())

            previous_end = max(
                end,
                previous_end
                if previous_end is not None
                else end,
            )

        if current_event:

            event_id += 1

            events.append(
                create_event_record(
                    event_id,
                    machine_id,
                    current_event,
                )
            )

    return pd.DataFrame(events)


def create_event_record(
    event_id,
    machine_id,
    windows,
):

    df = pd.DataFrame(windows)

    return {

        "event_id": event_id,

        "machineID": machine_id,

        "missed_windows": len(df),

        "event_start_step": int(
            df["target_start_step"].min()
        ),

        "event_end_step": int(
            df["target_end_step"].max()
        ),

        "event_span_steps": int(
            df["target_end_step"].max()
            - df["target_start_step"].min()
            + 1
        ),

        "average_high_probability": float(
            df["high_demand_probability"].mean()
        ),

        "maximum_high_probability": float(
            df["high_demand_probability"].max()
        ),

        "maximum_history_cpu": float(
            df["history_end_step"].max()
        ),

    }


# =========================================================
# EXTRACT EVENT UTILIZATION
# =========================================================

def analyze_event_utilization(events, raw):

    records = []

    grouped = {
        str(machine_id): group.sort_values("step")
        for machine_id, group in raw.groupby("machineID")
    }

    for _, event in events.iterrows():

        machine_id = str(event["machineID"])

        machine_data = grouped.get(machine_id)

        if machine_data is None:
            continue

        start = int(event["event_start_step"])
        end = int(event["event_end_step"])

        # Inspect 12 historical observations before the
        # missed event and all observations in its event span.
        history_start = start - 12

        history = machine_data[
            (machine_data["step"] >= history_start)
            &
            (machine_data["step"] < start)
        ]

        future = machine_data[
            (machine_data["step"] >= start)
            &
            (machine_data["step"] <= end)
        ]

        if history.empty or future.empty:
            continue

        record = {

            "event_id": int(event["event_id"]),

            "machineID": machine_id,

            "missed_windows": int(
                event["missed_windows"]
            ),

            "event_start_step": start,

            "event_end_step": end,

            "event_span_steps": int(
                event["event_span_steps"]
            ),

        }

        for column in UTILIZATION_COLUMNS:

            historical = pd.to_numeric(
                history[column],
                errors="coerce",
            ).dropna()

            event_values = pd.to_numeric(
                future[column],
                errors="coerce",
            ).dropna()

            if historical.empty or event_values.empty:
                continue

            record[
                f"{column}_pre_event_mean"
            ] = float(historical.mean())

            record[
                f"{column}_pre_event_max"
            ] = float(historical.max())

            record[
                f"{column}_event_mean"
            ] = float(event_values.mean())

            record[
                f"{column}_event_max"
            ] = float(event_values.max())

            record[
                f"{column}_event_min"
            ] = float(event_values.min())

            record[
                f"{column}_event_change"
            ] = float(
                event_values.mean()
                - historical.mean()
            )

        # Event drivers are inferred from actual future
        # utilization peaks, not from model probabilities.
        cpu_peak = record.get(
            "util_CPU_event_max",
            0,
        )

        memory_peak = record.get(
            "util_memory_event_max",
            0,
        )

        cpu_high = cpu_peak >= 85
        memory_high = memory_peak >= 85

        if cpu_high and memory_high:
            driver = "CPU_AND_MEMORY"

        elif cpu_high:
            driver = "CPU"

        elif memory_high:
            driver = "MEMORY"

        else:
            driver = "OTHER_OR_LABEL_WINDOW"

        record["event_driver"] = driver

        records.append(record)

    return pd.DataFrame(records)


# =========================================================
# CHECK WHETHER OTHER WINDOWS WERE DETECTED
# =========================================================

def analyze_event_detection(events, predictions):

    records = []

    for _, event in events.iterrows():

        machine_id = str(event["machineID"])

        start = int(event["event_start_step"])
        end = int(event["event_end_step"])

        machine_predictions = predictions[
            predictions["machineID"] == machine_id
        ]

        overlapping = machine_predictions[
            (
                machine_predictions["target_start_step"]
                <= end
            )
            &
            (
                machine_predictions["target_end_step"]
                >= start
            )
        ]

        detected = overlapping[
            overlapping["risk_alert"] == 1
        ]

        records.append({

            "event_id": int(event["event_id"]),

            "machineID": machine_id,

            "missed_windows": int(
                event["missed_windows"]
            ),

            "overlapping_test_windows": len(overlapping),

            "detected_overlapping_windows": len(detected),

            "missed_overlapping_windows": int(
                len(overlapping) - len(detected)
            ),

            "any_overlapping_detection": bool(
                not detected.empty
            ),

            "maximum_overlapping_probability": (
                float(
                    overlapping[
                        "high_demand_probability"
                    ].max()
                )
                if not overlapping.empty
                else None
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
    print("EXTRA TREES MISSED EVENT ANALYSIS")
    print("=" * 75)

    predictions, raw = load_data()

    predictions = reconstruct_predictions(
        predictions
    )

    missed_windows = predictions[
        predictions["false_negative"]
    ].copy()

    print(
        f"\nTotal test windows: {len(predictions):,}"
    )

    print(
        f"Missed HIGH_DEMAND windows: {len(missed_windows)}"
    )

    events = build_missed_events(
        predictions
    )

    if events.empty:

        print(
            "\nNo missed events found."
        )

        return

    utilization = analyze_event_utilization(
        events,
        raw,
    )

    detection = analyze_event_detection(
        events,
        predictions,
    )

    # -----------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------

    events.to_csv(
        OUTPUT_DIR / "missed_event_summary.csv",
        index=False,
    )

    utilization.to_csv(
        OUTPUT_DIR / "missed_event_utilization.csv",
        index=False,
    )

    detection.to_csv(
        OUTPUT_DIR / "missed_event_detection.csv",
        index=False,
    )

    # -----------------------------------------------------
    # EVENT DRIVER SUMMARY
    # -----------------------------------------------------

    if not utilization.empty:

        driver_summary = (
            utilization["event_driver"]
            .value_counts()
            .rename_axis("event_driver")
            .reset_index(name="event_count")
        )

    else:

        driver_summary = pd.DataFrame(
            columns=[
                "event_driver",
                "event_count",
            ]
        )

    driver_summary.to_csv(
        OUTPUT_DIR / "missed_event_driver_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # MACHINE SUMMARY
    # -----------------------------------------------------

    machine_summary = (
        events.groupby("machineID")
        .agg(
            missed_events=(
                "event_id",
                "count",
            ),

            missed_windows=(
                "missed_windows",
                "sum",
            ),

            average_event_span=(
                "event_span_steps",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "missed_windows",
            ascending=False,
        )
    )

    machine_summary.to_csv(
        OUTPUT_DIR / "missed_event_machine_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # JSON REPORT
    # -----------------------------------------------------

    summary = {

        "model": "Extra Trees",

        "threshold": THRESHOLD,

        "test_windows": int(len(predictions)),

        "actual_high_windows": int(
            predictions["actual_high"].sum()
        ),

        "missed_windows": int(
            len(missed_windows)
        ),

        "independent_missed_events": int(
            len(events)
        ),

        "machines_with_missed_events": int(
            events["machineID"].nunique()
        ),

        "events_with_overlapping_detection": int(
            detection["any_overlapping_detection"].sum()
        ),

        "events_without_overlapping_detection": int(
            (~detection["any_overlapping_detection"]).sum()
        ),

        "event_driver_counts": {
            str(row["event_driver"]): int(
                row["event_count"]
            )
            for _, row in driver_summary.iterrows()
        },

        "important_note": (
            "Events are groups of overlapping or adjacent "
            "missed target windows for the same machine. "
            "Event drivers use future utilization peaks. "
            "Overlapping detection is not equivalent to "
            "early warning or event-level recall."
        ),

    }

    with open(
        OUTPUT_DIR / "missed_event_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # CONSOLE OUTPUT
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("MISSED EVENT SUMMARY")
    print("=" * 75)

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print("\nTOP MISSED EVENTS")

    print(
        events.sort_values(
            "missed_windows",
            ascending=False,
        ).head(15).to_string(index=False)
    )

    print("\nEVENT DRIVER SUMMARY")

    print(
        driver_summary.to_string(index=False)
    )

    print("\nOVERLAPPING DETECTION SUMMARY")

    print(
        detection.head(15).to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
