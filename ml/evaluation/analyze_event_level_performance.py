
from pathlib import Path
import logging

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT / "datasets" / "processed" / "server_usage_clean.csv"

PREDICTION_DIR = (
    ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
    / "temporal_feature_evaluation"
)

OUTPUT_DIR = (
    ROOT
    / "ml"
    / "models"
    / "optimization_classifier"
    / "event_level_evaluation"
)

BASELINE_PATH = PREDICTION_DIR / "baseline_predictions.csv"
ENHANCED_PATH = PREDICTION_DIR / "enhanced_predictions.csv"

HIGH_THRESHOLD = 85.0
STEP_MINUTES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_data():
    logger.info("Loading raw workload data and predictions...")

    raw = pd.read_csv(RAW_DATA_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    enhanced = pd.read_csv(ENHANCED_PATH)

    required_raw = {
        "machineID", "step", "util_CPU", "util_memory"
    }

    required_predictions = {
        "machineID",
        "target_start_step",
        "target_end_step",
        "target",
        "risk_alert",
    }

    if not required_raw.issubset(raw.columns):
        raise ValueError(
            f"Missing raw columns: {required_raw - set(raw.columns)}"
        )

    for name, frame in [
        ("baseline", baseline),
        ("enhanced", enhanced),
    ]:
        missing = required_predictions - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} predictions missing columns: {missing}"
            )

    return raw, baseline, enhanced


# ---------------------------------------------------------
# BUILD ACTUAL HIGH-DEMAND EVENTS FROM RAW DATA
# ---------------------------------------------------------

def build_events(raw):
    logger.info("Identifying actual high-demand events...")

    raw = raw.copy()

    raw["is_high"] = (
        (raw["util_CPU"] >= HIGH_THRESHOLD)
        | (raw["util_memory"] >= HIGH_THRESHOLD)
    )

    raw = raw.sort_values(["machineID", "step"])

    events = []
    event_id = 0

    for machine_id, group in raw.groupby("machineID"):
        high_steps = (
            group.loc[group["is_high"], "step"]
            .astype(int)
            .sort_values()
            .tolist()
        )

        if not high_steps:
            continue

        start = previous = high_steps[0]

        for step in high_steps[1:]:
            if step == previous + 1:
                previous = step
                continue

            event_id += 1

            events.append({
                "event_id": event_id,
                "machineID": machine_id,
                "event_start_step": start,
                "event_end_step": previous,
                "event_duration_minutes": (
                    previous - start + 1
                ) * STEP_MINUTES,
            })

            start = previous = step

        event_id += 1

        events.append({
            "event_id": event_id,
            "machineID": machine_id,
            "event_start_step": start,
            "event_end_step": previous,
            "event_duration_minutes": (
                previous - start + 1
            ) * STEP_MINUTES,
        })

    result = pd.DataFrame(events)

    if result.empty:
        raise ValueError("No high-demand events found.")

    logger.info("Identified %s actual events.", len(result))

    return result


# ---------------------------------------------------------
# MATCH MODEL ALERTS TO ACTUAL EVENTS
# ---------------------------------------------------------

def evaluate_events(events, predictions, model_name):
    logger.info("Evaluating %s alerts against actual events...", model_name)

    predictions = predictions.copy()

    predictions["machineID"] = predictions["machineID"].astype(str)
    events = events.copy()
    events["machineID"] = events["machineID"].astype(str)

    predictions["is_alert"] = (
        predictions["risk_alert"]
        .astype(str)
        .str.upper()
        .isin(["HIGH_RISK", "TRUE", "1", "YES"])
    )

    alerts = predictions[predictions["is_alert"]].copy()

    event_results = []

    for event in events.itertuples(index=False):
        machine_alerts = alerts[
            alerts["machineID"] == event.machineID
        ]

        # An alert is associated with an event if its
        # predicted target interval overlaps the actual event.
        matched = machine_alerts[
            (
                machine_alerts["target_start_step"]
                <= event.event_end_step
            )
            &
            (
                machine_alerts["target_end_step"]
                >= event.event_start_step
            )
        ].copy()

        detected = not matched.empty

        if detected:
            # Earliest forecast window associated with this event.
            first_alert = matched.sort_values(
                "target_start_step"
            ).iloc[0]

            lead_steps = (
                event.event_start_step
                 - (first_alert["target_start_step"] - 1)
            )

            lead_minutes = lead_steps * STEP_MINUTES

            alert_count = len(matched)
        else:
            lead_minutes = None
            alert_count = 0

        event_results.append({
            "model": model_name,
            "event_id": event.event_id,
            "machineID": event.machineID,
            "event_start_step": event.event_start_step,
            "event_end_step": event.event_end_step,
            "event_duration_minutes": event.event_duration_minutes,
            "detected": detected,
            "matched_alert_count": alert_count,
            "lead_time_minutes": lead_minutes,
        })

    event_df = pd.DataFrame(event_results)

    # Count alert windows that do not overlap any actual event.
    unmatched_alerts = []

    for alert in alerts.itertuples(index=False):
        overlaps = events[
            (events["machineID"] == alert.machineID)
            &
            (
                events["event_start_step"]
                <= alert.target_end_step
            )
            &
            (
                events["event_end_step"]
                >= alert.target_start_step
            )
        ]

        if overlaps.empty:
            unmatched_alerts.append({
                "model": model_name,
                "machineID": alert.machineID,
                "target_start_step": alert.target_start_step,
                "target_end_step": alert.target_end_step,
                "high_demand_probability": getattr(
                    alert, "high_demand_probability", None
                ),
            })

    false_alert_df = pd.DataFrame(unmatched_alerts)

    return event_df, false_alert_df


# ---------------------------------------------------------
# SUMMARIZE RESULTS
# ---------------------------------------------------------

def summarize(event_results, false_alerts):
    summaries = []

    for model_name, group in event_results.groupby("model"):
        total = len(group)
        detected = int(group["detected"].sum())
        missed = total - detected

        detected_events = group[group["detected"]]

        lead_times = detected_events[
            "lead_time_minutes"
        ].dropna()

        summaries.append({
            "model": model_name,
            "total_actual_events": total,
            "detected_events": detected,
            "missed_events": missed,
            "event_recall_percent": (
                detected / total * 100 if total else 0
            ),
            "mean_lead_time_minutes": (
                lead_times.mean() if len(lead_times) else None
            ),
            "median_lead_time_minutes": (
                lead_times.median() if len(lead_times) else None
            ),
            "events_with_early_warning": int(
                (lead_times > 0).sum()
            ),
            "events_alerted_at_or_after_start": int(
                (lead_times <= 0).sum()
            ),
            "unmatched_alert_windows": int(
                (false_alerts["model"] == model_name).sum()
            ) if not false_alerts.empty else 0,
        })

    return pd.DataFrame(summaries)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, baseline, enhanced = load_data()

    # Determine the exact test forecast coverage.
    evaluation_start = min(
        baseline["target_start_step"].min(),
        enhanced["target_start_step"].min(),
    )

    evaluation_end = max(
        baseline["target_end_step"].max(),
        enhanced["target_end_step"].max(),
    )

    logger.info(
        "Evaluation period: steps %s to %s",
        evaluation_start,
        evaluation_end,
    )

    # Keep only the raw observations covered by test forecasts.
    raw = raw[
        (raw["step"] >= evaluation_start)
        & (raw["step"] <= evaluation_end)
    ].copy()

    # Rebuild actual events only within the evaluation period.
    events = build_events(raw)

    baseline_events, baseline_false = evaluate_events(
        events, baseline, "BASELINE"
    )

    enhanced_events, enhanced_false = evaluate_events(
        events, enhanced, "ENHANCED"
    )

    all_events = pd.concat(
        [baseline_events, enhanced_events],
        ignore_index=True,
    )

    all_false = pd.concat(
        [baseline_false, enhanced_false],
        ignore_index=True,
    )

    summary = summarize(all_events, all_false)

    comparison = (
        all_events.pivot(
            index="event_id",
            columns="model",
            values="detected",
        )
        .reset_index()
    )

    comparison = comparison.rename_axis(None, axis=1)

    comparison["agreement"] = comparison.apply(
        lambda row:
            "BOTH_DETECTED"
            if row.get("BASELINE") and row.get("ENHANCED")
            else "BASELINE_ONLY"
            if row.get("BASELINE")
            else "ENHANCED_ONLY"
            if row.get("ENHANCED")
            else "BOTH_MISSED",
        axis=1,
    )

    comparison = comparison.merge(
        events,
        on="event_id",
        how="left",
    )

    events.to_csv(
        OUTPUT_DIR / "actual_high_demand_events.csv",
        index=False,
    )

    all_events.to_csv(
        OUTPUT_DIR / "event_detection_details.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "event_level_model_comparison.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "event_agreement_comparison.csv",
        index=False,
    )

    all_false.to_csv(
        OUTPUT_DIR / "unmatched_alert_windows.csv",
        index=False,
    )

    logger.info(
        "\nEVENT-LEVEL MODEL COMPARISON\n%s",
        summary.to_string(index=False),
    )

    logger.info(
        "\nEVENT AGREEMENT\n%s",
        comparison["agreement"].value_counts().to_string(),
    )

    logger.info("Evaluation period: %s to %s", evaluation_start, evaluation_end)
    logger.info("Actual events in evaluation period: %s", len(events))
    logger.info("EVENT-LEVEL ANALYSIS COMPLETED")
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
