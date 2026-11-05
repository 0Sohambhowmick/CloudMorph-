
"""
CloudMorph - Event-Level Memory Classifier Evaluation

Evaluates high-memory events using the original test machines.

An event is a continuous period during which raw memory
utilization is >= 85%.

A prediction is considered:
1. EARLY: Alert occurs before the event starts.
2. DURING: Alert occurs during the event.
3. LATE: Alert occurs after the event.
4. MISSED: No alert occurs in the associated evaluation window.

Important:
Prediction windows overlap. This script uses their historical
window end step as the time at which the prediction is available.

It does not retrain or modify any model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT
    / "ml"
    / "data"
    / "optimization_training_dataset.csv"
)

RAW_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "ml"
    / "models"
    / "memory_high_demand_classifier"
    / "test_predictions.csv"
)

OUTPUT_DIR = ROOT / "ml" / "evaluation"

MEMORY_THRESHOLD = 85.0

# A prediction describes the next six observations.
PREDICTION_HORIZON_STEPS = 6

# The historical data uses 5-minute steps.
MIN_EVENT_GAP_STEPS = 1


# ---------------------------------------------------------
# RECONSTRUCT EXACT TEST SPLIT
# ---------------------------------------------------------

def reconstruct_test_rows(features):

    features = features.copy()

    features["machineID"] = (
        features["machineID"].astype(str)
    )

    X = features.drop(
        columns=[
            "machineID",
            "history_start_step",
            "history_end_step",
            "target_start_step",
            "target_end_step",
            "target",
        ],
        errors="ignore",
    )

    X = X.apply(
        pd.to_numeric,
        errors="raise",
    )

    groups = features["machineID"]

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_validation_idx, test_idx = next(
        outer.split(X, groups=groups)
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=43,
    )

    inner.split(
        X.iloc[train_validation_idx],
        groups=groups.iloc[train_validation_idx],
    )

    return (
        features.iloc[test_idx].reset_index(drop=True)
    )


# ---------------------------------------------------------
# DETECT CONTINUOUS HIGH-MEMORY EVENTS
# ---------------------------------------------------------

def detect_events(raw):

    events = []

    raw = raw.sort_values(
        ["machineID", "step"]
    )

    for machine_id, group in raw.groupby("machineID"):

        group = group.sort_values("step").reset_index(
            drop=True
        )

        high = (
            group["util_memory"].to_numpy()
            >= MEMORY_THRESHOLD
        )

        steps = group["step"].to_numpy()

        values = group["util_memory"].to_numpy()

        active_indices = np.where(high)[0]

        if len(active_indices) == 0:
            continue

        current_indices = [active_indices[0]]

        for index in active_indices[1:]:

            previous_index = current_indices[-1]

            step_gap = (
                steps[index] - steps[previous_index]
            )

            if step_gap <= MIN_EVENT_GAP_STEPS:

                current_indices.append(index)

            else:

                events.append(
                    build_event(
                        machine_id,
                        current_indices,
                        steps,
                        values,
                    )
                )

                current_indices = [index]

        if current_indices:

            events.append(
                build_event(
                    machine_id,
                    current_indices,
                    steps,
                    values,
                )
            )

    return pd.DataFrame(events)


def build_event(machine_id, indices, steps, values):

    event_steps = steps[indices]

    event_values = values[indices]

    return {
        "machineID": str(machine_id),
        "event_start_step": int(event_steps[0]),
        "event_end_step": int(event_steps[-1]),
        "event_duration_steps": int(
            event_steps[-1] - event_steps[0] + 1
        ),
        "event_duration_minutes": float(
            (event_steps[-1] - event_steps[0] + 1) * 5
        ),
        "event_peak_memory": float(
            np.max(event_values)
        ),
        "event_mean_memory": float(
            np.mean(event_values)
        ),
    }


# ---------------------------------------------------------
# CLASSIFY EVENT DETECTION
# ---------------------------------------------------------

def evaluate_events(events, test_rows, predictions):

    test_rows = test_rows.copy()

    test_rows["predicted_high_memory"] = (
        predictions["predicted_high_memory"].to_numpy()
    )

    test_rows["predicted_probability"] = (
        predictions["predicted_probability"].to_numpy()
    )

    test_rows["machineID"] = (
        test_rows["machineID"].astype(str)
    )

    test_rows["history_end_step"] = pd.to_numeric(
        test_rows["history_end_step"],
        errors="raise",
    ).astype(int)

    test_rows["target_start_step"] = pd.to_numeric(
        test_rows["target_start_step"],
        errors="raise",
    ).astype(int)

    test_rows["target_end_step"] = pd.to_numeric(
        test_rows["target_end_step"],
        errors="raise",
    ).astype(int)

    results = []

    for event_id, event in events.iterrows():

        machine_id = event["machineID"]

        event_start = int(event["event_start_step"])

        event_end = int(event["event_end_step"])

        event_predictions = test_rows[
            test_rows["machineID"] == machine_id
        ].copy()

        # Prediction target horizon overlaps the event.

        relevant = event_predictions[
            (
                event_predictions["target_start_step"]
                <= event_end
            )
            &
            (
                event_predictions["target_end_step"]
                >= event_start
            )
        ].copy()

        # A prediction is only available after its history
        # has been observed.

        alerts = relevant[
            relevant["predicted_high_memory"] == 1
        ].copy()

        if alerts.empty:

            status = "MISSED"

            first_alert_time = None

            lead_steps = None

            lead_minutes = None

        else:

            first_alert = alerts.sort_values(
                "history_end_step"
            ).iloc[0]

            first_alert_time = int(
                first_alert["history_end_step"]
            )

            lead_steps = event_start - first_alert_time

            lead_minutes = lead_steps * 5

            if lead_steps > 0:

                status = "EARLY"

            elif lead_steps == 0:

                status = "AT_ONSET"

            elif first_alert_time <= event_end:

                status = "DURING"

            else:

                status = "LATE"

        results.append({
            "event_id": int(event_id + 1),
            "machineID": machine_id,
            "event_start_step": event_start,
            "event_end_step": event_end,
            "event_duration_minutes": event[
                "event_duration_minutes"
            ],
            "event_peak_memory": event[
                "event_peak_memory"
            ],
            "event_mean_memory": event[
                "event_mean_memory"
            ],
            "relevant_prediction_windows": len(relevant),
            "alert_windows": len(alerts),
            "first_alert_history_end_step": first_alert_time,
            "lead_time_steps": lead_steps,
            "lead_time_minutes": lead_minutes,
            "detection_status": status,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def print_summary(results):

    print("\n" + "=" * 75)

    print("EVENT-LEVEL MEMORY CLASSIFIER EVALUATION")

    print("=" * 75)

    print("\n1. EVENT COUNTS")

    print(f"Total events: {len(results)}")

    print(
        results["detection_status"]
        .value_counts()
        .to_string()
    )

    print("\n2. EVENT DETECTION RATE")

    detected = results[
        results["detection_status"] != "MISSED"
    ]

    early = results[
        results["detection_status"] == "EARLY"
    ]

    during = results[
        results["detection_status"].isin(
            ["AT_ONSET", "DURING"]
        )
    ]

    late = results[
        results["detection_status"] == "LATE"
    ]

    missed = results[
        results["detection_status"] == "MISSED"
    ]

    total = len(results)

    print(
        "Any detection: "
        f"{len(detected)}/{total} "
        f"({len(detected) / total * 100:.2f}%)"
    )

    print(
        "Early detection: "
        f"{len(early)}/{total} "
        f"({len(early) / total * 100:.2f}%)"
    )

    print(
        "At onset/during: "
        f"{len(during)}/{total} "
        f"({len(during) / total * 100:.2f}%)"
    )

    print(
        "Late detection: "
        f"{len(late)}/{total} "
        f"({len(late) / total * 100:.2f}%)"
    )

    print(
        "Completely missed: "
        f"{len(missed)}/{total} "
        f"({len(missed) / total * 100:.2f}%)"
    )

    print("\n3. EARLY WARNING LEAD TIME")

    if not early.empty:

        print(
            early["lead_time_minutes"]
            .describe()
            .round(2)
            .to_string()
        )

    else:

        print("No events detected before onset.")

    print("\n4. EVENT DETAILS")

    print(
        results[
            [
                "event_id",
                "machineID",
                "event_start_step",
                "event_end_step",
                "event_peak_memory",
                "detection_status",
                "lead_time_minutes",
                "alert_windows",
            ]
        ]
        .to_string(index=False)
    )

    print("\n5. MACHINE-WISE EVENT RESULTS")

    print(
        results.groupby("machineID").agg(
            total_events=("event_id", "count"),
            early_events=(
                "detection_status",
                lambda x: (x == "EARLY").sum(),
            ),
            during_events=(
                "detection_status",
                lambda x: x.isin(
                    ["AT_ONSET", "DURING"]
                ).sum(),
            ),
            late_events=(
                "detection_status",
                lambda x: (x == "LATE").sum(),
            ),
            missed_events=(
                "detection_status",
                lambda x: (x == "MISSED").sum(),
            ),
        ).to_string()
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = pd.read_csv(FEATURE_PATH)

    raw = pd.read_csv(RAW_PATH)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    features["machineID"] = (
        features["machineID"].astype(str)
    )

    raw["machineID"] = (
        raw["machineID"].astype(str)
    )

    raw["step"] = pd.to_numeric(
        raw["step"],
        errors="raise",
    ).astype(int)

    raw["util_memory"] = pd.to_numeric(
        raw["util_memory"],
        errors="raise",
    )

    # Reconstruct the original test split.

    test_rows = reconstruct_test_rows(features)

    if len(test_rows) != len(predictions):

        raise ValueError(
            "Test row count does not match saved predictions."
        )

    if not np.array_equal(
        test_rows["machineID"].to_numpy(),
        predictions["machineID"].astype(str).to_numpy(),
    ):

        raise ValueError(
            "Test machine order does not match saved predictions."
        )

    print(
        "\nSUCCESS: Original test rows and machine order verified."
    )

    # Restrict raw event detection to held-out test machines.

    test_machines = set(
        test_rows["machineID"]
    )

    raw_test = raw[
        raw["machineID"].isin(test_machines)
    ].copy()

    events = detect_events(raw_test)

    print(
        f"\nHigh-memory threshold: {MEMORY_THRESHOLD}%"
    )

    print(
        f"Test machines: {len(test_machines)}"
    )

    print(
        f"Distinct high-memory events: {len(events)}"
    )

    results = evaluate_events(
        events,
        test_rows,
        predictions,
    )

    if results.empty:

        raise ValueError(
            "No high-memory events found in test machines."
        )

    print_summary(results)

    # Save outputs.

    events.to_csv(
        OUTPUT_DIR / "memory_test_high_events.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR / "memory_event_level_results.csv",
        index=False,
    )

    summary = (
        results["detection_status"]
        .value_counts()
        .rename_axis("detection_status")
        .reset_index(name="event_count")
    )

    summary.to_csv(
        OUTPUT_DIR / "memory_event_level_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)

    print("EVENT-LEVEL EVALUATION COMPLETED")

    print("=" * 75)

    print("\nGenerated files:")

    print("1. memory_test_high_events.csv")

    print("2. memory_event_level_results.csv")

    print("3. memory_event_level_summary.csv")


if __name__ == "__main__":
    main()
