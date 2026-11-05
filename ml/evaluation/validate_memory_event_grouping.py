
"""
CloudMorph - Memory Event Grouping Validation

Compares high-memory event definitions using:
    Gap 0: No below-threshold observations allowed.
    Gap 1: Allow one below-threshold observation.
    Gap 2: Allow two below-threshold observations.

Uses the original held-out test machines and saved predictions.

No model retraining or threshold tuning.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT / "ml" / "data"
    / "optimization_training_dataset.csv"
)

RAW_PATH = (
    ROOT / "datasets" / "processed"
    / "server_usage_clean.csv"
)

PREDICTIONS_PATH = (
    ROOT / "ml" / "models"
    / "memory_high_demand_classifier"
    / "test_predictions.csv"
)

OUTPUT_DIR = ROOT / "ml" / "evaluation"

MEMORY_THRESHOLD = 85.0

GAP_SETTINGS = [0, 1, 2]

STEP_MINUTES = 5


# ---------------------------------------------------------
# RECONSTRUCT TEST DATA
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
    ).apply(pd.to_numeric, errors="raise")

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

    return features.iloc[test_idx].reset_index(drop=True)


# ---------------------------------------------------------
# BUILD EVENTS WITH ALLOWED GAPS
# ---------------------------------------------------------

def build_events(raw_test, allowed_gap):

    events = []

    for machine_id, group in raw_test.groupby("machineID"):

        group = group.sort_values("step").reset_index(
            drop=True
        )

        steps = group["step"].to_numpy()

        memory = group["util_memory"].to_numpy()

        high_indices = np.where(
            memory >= MEMORY_THRESHOLD
        )[0]

        if len(high_indices) == 0:
            continue

        current_indices = [high_indices[0]]

        for index in high_indices[1:]:

            previous_index = current_indices[-1]

            previous_step = steps[previous_index]

            current_step = steps[index]

            # Count actual missing observations between
            # two high-memory observations.

            gap = current_step - previous_step - 1

            if gap <= allowed_gap:

                current_indices.append(index)

            else:

                events.append(
                    create_event(
                        machine_id,
                        current_indices,
                        steps,
                        memory,
                        allowed_gap,
                    )
                )

                current_indices = [index]

        if current_indices:

            events.append(
                create_event(
                    machine_id,
                    current_indices,
                    steps,
                    memory,
                    allowed_gap,
                )
            )

    return pd.DataFrame(events)


def create_event(
    machine_id,
    indices,
    steps,
    memory,
    allowed_gap,
):

    event_steps = steps[indices]

    event_memory = memory[indices]

    return {
        "machineID": str(machine_id),
        "allowed_gap": allowed_gap,
        "event_start_step": int(event_steps[0]),
        "event_end_step": int(event_steps[-1]),
        "event_observation_count": len(indices),
        "event_duration_steps": int(
            event_steps[-1] - event_steps[0] + 1
        ),
        "event_duration_minutes": float(
            (event_steps[-1] - event_steps[0] + 1)
            * STEP_MINUTES
        ),
        "event_peak_memory": float(
            np.max(event_memory)
        ),
        "event_mean_high_memory": float(
            np.mean(event_memory)
        ),
    }


# ---------------------------------------------------------
# MATCH EVENTS TO SAVED PREDICTIONS
# ---------------------------------------------------------

def evaluate_events(events, test_rows, predictions):

    test_rows = test_rows.copy()

    test_rows["machineID"] = (
        test_rows["machineID"].astype(str)
    )

    predictions = predictions.copy()

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
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

    if len(test_rows) != len(predictions):
        raise ValueError("Test row alignment mismatch.")

    if not np.array_equal(
        test_rows["machineID"].to_numpy(),
        predictions["machineID"].to_numpy(),
    ):
        raise ValueError("Test machine ordering mismatch.")

    test_rows["predicted_high_memory"] = (
        predictions["predicted_high_memory"].to_numpy()
    )

    test_rows["predicted_probability"] = (
        predictions["predicted_probability"].to_numpy()
    )

    results = []

    for event_id, event in events.iterrows():

        machine_id = event["machineID"]

        event_start = int(event["event_start_step"])

        event_end = int(event["event_end_step"])

        machine_predictions = test_rows[
            test_rows["machineID"] == machine_id
        ].copy()

        relevant = machine_predictions[
            (
                machine_predictions["target_start_step"]
                <= event_end
            )
            &
            (
                machine_predictions["target_end_step"]
                >= event_start
            )
        ].copy()

        alerts = relevant[
            relevant["predicted_high_memory"] == 1
        ].copy()

        if alerts.empty:

            status = "MISSED"

            first_alert_step = None

            lead_minutes = None

        else:

            first_alert = alerts.sort_values(
                "history_end_step"
            ).iloc[0]

            first_alert_step = int(
                first_alert["history_end_step"]
            )

            lead_minutes = (
                event_start - first_alert_step
            ) * STEP_MINUTES

            if lead_minutes > 0:

                status = "EARLY"

            elif lead_minutes == 0:

                status = "AT_ONSET"

            elif first_alert_step <= event_end:

                status = "DURING"

            else:

                status = "LATE"

        results.append({
            **event.to_dict(),
            "event_id": int(event_id + 1),
            "relevant_prediction_windows": len(relevant),
            "alert_windows": len(alerts),
            "first_alert_history_end_step": first_alert_step,
            "lead_time_minutes": lead_minutes,
            "detection_status": status,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# SUMMARIZE RESULTS
# ---------------------------------------------------------

def summarize_results(results, gap):

    total = len(results)

    counts = results["detection_status"].value_counts()

    early = int(counts.get("EARLY", 0))

    at_onset = int(counts.get("AT_ONSET", 0))

    during = int(counts.get("DURING", 0))

    late = int(counts.get("LATE", 0))

    missed = int(counts.get("MISSED", 0))

    any_detected = total - missed

    early_lead = results.loc[
        results["detection_status"] == "EARLY",
        "lead_time_minutes",
    ].dropna()

    return {
        "allowed_gap": gap,
        "total_events": total,
        "early_events": early,
        "at_onset_events": at_onset,
        "during_events": during,
        "late_events": late,
        "missed_events": missed,
        "any_detection_rate_pct": (
            any_detected / total * 100
            if total else 0
        ),
        "early_detection_rate_pct": (
            early / total * 100
            if total else 0
        ),
        "early_lead_mean_minutes": (
            float(early_lead.mean())
            if len(early_lead) else None
        ),
        "early_lead_median_minutes": (
            float(early_lead.median())
            if len(early_lead) else None
        ),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 75)

    print("CLOUDMORPH MEMORY EVENT GROUPING VALIDATION")

    print("=" * 75)

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

    predictions["machineID"] = (
        predictions["machineID"].astype(str)
    )

    test_rows = reconstruct_test_rows(features)

    if len(test_rows) != len(predictions):
        raise ValueError("Test prediction row count mismatch.")

    if not np.array_equal(
        test_rows["machineID"].to_numpy(),
        predictions["machineID"].to_numpy(),
    ):
        raise ValueError("Test prediction order mismatch.")

    print(
        "\nSUCCESS: Exact held-out test rows verified."
    )

    test_machines = set(
        test_rows["machineID"]
    )

    raw_test = raw[
        raw["machineID"].isin(test_machines)
    ].copy()

    all_results = []

    summary_rows = []

    machine_summary_rows = []

    for gap in GAP_SETTINGS:

        print("\n" + "-" * 75)

        print(
            f"EVENT GROUPING: ALLOW {gap} BELOW-THRESHOLD OBSERVATIONS"
        )

        print("-" * 75)

        events = build_events(
            raw_test,
            allowed_gap=gap,
        )

        results = evaluate_events(
            events,
            test_rows,
            predictions,
        )

        summary = summarize_results(
            results,
            gap,
        )

        summary_rows.append(summary)

        results["allowed_gap"] = gap

        all_results.append(results)

        print(
            f"Total events: {summary['total_events']}"
        )

        print(
            f"Early: {summary['early_events']}"
        )

        print(
            f"At onset: {summary['at_onset_events']}"
        )

        print(
            f"During: {summary['during_events']}"
        )

        print(
            f"Missed: {summary['missed_events']}"
        )

        print(
            "Any detection rate: "
            f"{summary['any_detection_rate_pct']:.2f}%"
        )

        print(
            "Early detection rate: "
            f"{summary['early_detection_rate_pct']:.2f}%"
        )

        print(
            "Mean early lead time: "
            f"{summary['early_lead_mean_minutes']}"
        )

        print("\nMachine-wise results:")

        machine_summary = (
            results.groupby("machineID")
            .agg(
                total_events=("event_id", "count"),
                early_events=(
                    "detection_status",
                    lambda x: (x == "EARLY").sum(),
                ),
                at_onset_events=(
                    "detection_status",
                    lambda x: (x == "AT_ONSET").sum(),
                ),
                during_events=(
                    "detection_status",
                    lambda x: (x == "DURING").sum(),
                ),
                missed_events=(
                    "detection_status",
                    lambda x: (x == "MISSED").sum(),
                ),
            )
            .reset_index()
        )

        machine_summary["allowed_gap"] = gap

        print(
            machine_summary.to_string(index=False)
        )

        machine_summary_rows.append(
            machine_summary
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summary_rows)

    machine_df = pd.concat(
        machine_summary_rows,
        ignore_index=True,
    )

    combined_results.to_csv(
        OUTPUT_DIR
        / "memory_event_grouping_details.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "memory_event_grouping_comparison.csv",
        index=False,
    )

    machine_df.to_csv(
        OUTPUT_DIR
        / "memory_event_grouping_machine_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 75)

    print("FINAL EVENT GROUPING COMPARISON")

    print("=" * 75)

    print(
        summary_df.round(3).to_string(index=False)
    )

    print("\nGenerated files:")

    print("1. memory_event_grouping_details.csv")

    print("2. memory_event_grouping_comparison.csv")

    print("3. memory_event_grouping_machine_comparison.csv")

    print("\nEVENT GROUPING VALIDATION COMPLETED")


if __name__ == "__main__":
    main()
