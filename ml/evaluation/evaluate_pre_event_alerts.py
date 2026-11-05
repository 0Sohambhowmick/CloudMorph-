
from pathlib import Path
import logging

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "temporal_feature_evaluation"
)

EVENT_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "event_level_evaluation"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "pre_event_evaluation"
)

BASELINE_PATH = MODEL_DIR / "baseline_predictions.csv"
ENHANCED_PATH = MODEL_DIR / "enhanced_predictions.csv"
EVENT_PATH = EVENT_DIR / "actual_high_demand_events.csv"

STEP_MINUTES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_data():
    baseline = pd.read_csv(BASELINE_PATH)
    enhanced = pd.read_csv(ENHANCED_PATH)
    events = pd.read_csv(EVENT_PATH)

    return baseline, enhanced, events


def evaluate_model(events, predictions, model_name):
    predictions = predictions.copy()
    predictions["machineID"] = predictions["machineID"].astype(str)

    predictions["is_alert"] = (
        predictions["risk_alert"]
        .astype(str)
        .str.upper()
        .isin(["HIGH_RISK", "TRUE", "1", "YES"])
    )

    results = []

    for event in events.itertuples(index=False):
        machine = predictions[
            predictions["machineID"] == str(event.machineID)
        ].copy()

        # Only forecasts whose target interval includes the
        # event's first high-demand observation, and which
        # were issued before that observation.
        eligible = machine[
            (
                machine["target_start_step"]
                < event.event_start_step
            )
            &
            (
                machine["target_end_step"]
                >= event.event_start_step
            )
        ].copy()

        alerted = eligible[eligible["is_alert"]].copy()

        detected = not alerted.empty

        if detected:
            # Earliest issued alert among eligible forecasts.
            first_alert = alerted.sort_values(
                "target_start_step"
            ).iloc[0]

            prediction_available_step = (
                first_alert["target_start_step"] - 1
            )

            lead_time = (
                event.event_start_step
                - prediction_available_step
            ) * STEP_MINUTES

            probability = first_alert[
                "high_demand_probability"
            ]

        else:
            lead_time = None
            probability = (
                eligible["high_demand_probability"].max()
                if not eligible.empty
                else None
            )

        results.append({
            "model": model_name,
            "event_id": event.event_id,
            "machineID": event.machineID,
            "event_start_step": event.event_start_step,
            "eligible_forecast_windows": len(eligible),
            "detected_before_event": detected,
            "early_warning_lead_minutes": lead_time,
            "alert_probability": probability,
        })

    return pd.DataFrame(results)


def summarize(results):
    rows = []

    for model, group in results.groupby("model"):
        detected = group[group["detected_before_event"]]
        leads = detected["early_warning_lead_minutes"].dropna()

        total = len(group)
        detected_count = len(detected)

        rows.append({
            "model": model,
            "total_events": total,
            "events_with_early_alert": detected_count,
            "events_without_early_alert": total - detected_count,
            "early_event_recall_percent": (
                detected_count / total * 100 if total else 0
            ),
            "mean_lead_minutes": (
                leads.mean() if len(leads) else None
            ),
            "median_lead_minutes": (
                leads.median() if len(leads) else None
            ),
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline, enhanced, events = load_data()

    logger.info("Evaluating %s actual events.", len(events))

    baseline_results = evaluate_model(
        events, baseline, "BASELINE"
    )

    enhanced_results = evaluate_model(
        events, enhanced, "ENHANCED"
    )

    details = pd.concat(
        [baseline_results, enhanced_results],
        ignore_index=True,
    )

    summary = summarize(details)

    comparison = details.pivot(
        index="event_id",
        columns="model",
        values="detected_before_event",
    ).reset_index()

    comparison.columns.name = None

    comparison["agreement"] = comparison.apply(
        lambda row:
            "BOTH_EARLY_ALERT"
            if row["BASELINE"] and row["ENHANCED"]
            else "BASELINE_ONLY"
            if row["BASELINE"]
            else "ENHANCED_ONLY"
            if row["ENHANCED"]
            else "BOTH_MISSED",
        axis=1,
    )

    details.to_csv(
        OUTPUT_DIR / "pre_event_details.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "pre_event_summary.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "pre_event_agreement.csv",
        index=False,
    )

    logger.info(
        "\nPRE-EVENT PERFORMANCE\n%s",
        summary.to_string(index=False),
    )

    logger.info(
        "\nEARLY ALERT AGREEMENT\n%s",
        comparison["agreement"].value_counts().to_string(),
    )

    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("PRE-EVENT EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
