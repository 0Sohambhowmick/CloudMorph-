
from pathlib import Path
import logging
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT / "ml" / "models" / "optimization_classifier"
)

PRE_EVENT = BASE / "pre_event_evaluation" / "pre_event_details.csv"
EVENT_DETAILS = BASE / "event_level_evaluation" / "event_detection_details.csv"

OUTPUT = BASE / "early_late_analysis"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    pre = pd.read_csv(PRE_EVENT)
    event = pd.read_csv(EVENT_DETAILS)

    pre["machineID"] = pre["machineID"].astype(str)
    event["machineID"] = event["machineID"].astype(str)

    pre = pre.rename(columns={
        "detected_before_event": "early_detection",
        "early_warning_lead_minutes": "early_lead_minutes",
        "alert_probability": "early_alert_probability",
    })

    event = event.rename(columns={
        "detected": "event_detection",
        "lead_time_minutes": "event_lead_minutes",
    })

    merged = pre.merge(
        event[
            [
                "model",
                "event_id",
                "event_detection",
                "event_lead_minutes",
                "matched_alert_count",
            ]
        ],
        on=["model", "event_id"],
        how="left",
        validate="one_to_one",
    )

    def classify(row):
        if row["early_detection"]:
            return "EARLY_DETECTION"

        if row["event_detection"]:
            return "LATE_ONLY_DETECTION"

        return "NEVER_DETECTED"

    merged["detection_category"] = merged.apply(
        classify, axis=1
    )

    summary = (
        merged.groupby(
            ["model", "detection_category"]
        )
        .agg(
            event_count=("event_id", "count"),
            mean_probability=("early_alert_probability", "mean"),
            mean_event_lead_minutes=("event_lead_minutes", "mean"),
        )
        .reset_index()
    )

    comparison = merged.pivot(
        index="event_id",
        columns="model",
        values="detection_category",
    ).reset_index()

    comparison.columns.name = None

    merged.to_csv(
        OUTPUT / "early_late_event_details.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT / "early_late_summary.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT / "early_late_model_comparison.csv",
        index=False,
    )

    logger.info("\nEARLY/LATE SUMMARY\n%s", summary.to_string(index=False))
    logger.info("\nMODEL COMPARISON\n%s", comparison.to_string(index=False))
    logger.info("Saved reports to %s", OUTPUT)


if __name__ == "__main__":
    main()
