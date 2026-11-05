
from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "datasets" / "processed" / "server_usage_clean.csv"

PREDICTION_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "temporal_feature_evaluation"
)

EVENT_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "event_level_evaluation"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "missed_event_analysis"
)

BASELINE_PATH = PREDICTION_DIR / "baseline_predictions.csv"
ENHANCED_PATH = PREDICTION_DIR / "enhanced_predictions.csv"

AGREEMENT_PATH = EVENT_DIR / "event_agreement_comparison.csv"

HIGH_THRESHOLD = 85.0
HISTORY_WINDOW = 12
STEP_MINUTES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------

def load_data():
    logger.info("Loading raw data, events and model predictions...")

    raw = pd.read_csv(RAW_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    enhanced = pd.read_csv(ENHANCED_PATH)
    agreement = pd.read_csv(AGREEMENT_PATH)

    required_raw = {
        "machineID", "step", "util_CPU", "util_memory"
    }

    if not required_raw.issubset(raw.columns):
        raise ValueError("Required raw dataset columns are missing.")

    for name, frame in [
        ("baseline", baseline),
        ("enhanced", enhanced),
    ]:
        required = {
            "machineID",
            "target_start_step",
            "target_end_step",
            "high_demand_probability",
            "risk_alert",
        }

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} prediction file missing: {missing}"
            )

    return raw, baseline, enhanced, agreement


# ---------------------------------------------------------
# HISTORICAL FEATURE ANALYSIS
# ---------------------------------------------------------

def calculate_history_features(history):
    history = history.sort_values("step")

    features = {}

    signals = ["util_CPU", "util_memory"]

    for signal in signals:
        values = history[signal].astype(float).to_numpy()

        features[f"{signal}_current"] = values[-1]
        features[f"{signal}_mean"] = values.mean()
        features[f"{signal}_min"] = values.min()
        features[f"{signal}_max"] = values.max()
        features[f"{signal}_std"] = values.std()
        features[f"{signal}_recent_mean"] = values[-3:].mean()

        features[f"{signal}_previous_mean"] = (
            values[-6:-3].mean()
        )

        features[f"{signal}_recent_change"] = (
            values[-3:].mean() - values[-6:-3].mean()
        )

        features[f"{signal}_last_change"] = (
            values[-1] - values[-2]
        )

        features[f"{signal}_trend_per_step"] = (
            np.polyfit(np.arange(len(values)), values, 1)[0]
        )

    return features


# ---------------------------------------------------------
# MODEL PROBABILITY DURING EVENT
# ---------------------------------------------------------

def get_model_event_metrics(
    predictions,
    machine_id,
    event_start,
    event_end,
    model_name,
):
    machine_predictions = predictions[
        predictions["machineID"].astype(str)
        == str(machine_id)
    ].copy()

    overlapping = machine_predictions[
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

    if overlapping.empty:
        return {
            f"{model_name}_overlapping_windows": 0,
            f"{model_name}_max_high_probability": None,
            f"{model_name}_mean_high_probability": None,
            f"{model_name}_minimum_threshold": None,
            f"{model_name}_alert_windows": 0,
        }

    return {
        f"{model_name}_overlapping_windows": len(overlapping),
        f"{model_name}_max_high_probability": (
            overlapping["high_demand_probability"].max()
        ),
        f"{model_name}_mean_high_probability": (
            overlapping["high_demand_probability"].mean()
        ),
        f"{model_name}_minimum_threshold": (
            overlapping["risk_threshold"].min()
            if "risk_threshold" in overlapping.columns
            else None
        ),
        f"{model_name}_alert_windows": int(
            overlapping["risk_alert"]
            .astype(str)
            .str.upper()
            .isin(["HIGH_RISK", "TRUE", "1", "YES"])
            .sum()
        ),
    }


# ---------------------------------------------------------
# INVESTIGATE EACH MISSED EVENT
# ---------------------------------------------------------

def analyze_missed_events(
    raw,
    baseline,
    enhanced,
    missed_events,
):
    results = []

    raw["machineID"] = raw["machineID"].astype(str)

    for event in missed_events.itertuples(index=False):
        machine_id = str(event.machineID)
        start = int(event.event_start_step)
        end = int(event.event_end_step)

        machine_data = raw[
            raw["machineID"] == machine_id
        ].sort_values("step")

        history = machine_data[
            (machine_data["step"] >= start - HISTORY_WINDOW)
            &
            (machine_data["step"] < start)
        ]

        future = machine_data[
            (machine_data["step"] >= start)
            &
            (machine_data["step"] <= end)
        ]

        if len(history) != HISTORY_WINDOW:
            logger.warning(
                "Skipping machine %s event at step %s: "
                "expected %s history rows, found %s",
                machine_id,
                start,
                HISTORY_WINDOW,
                len(history),
            )
            continue

        if future.empty:
            continue

        features = calculate_history_features(history)

        cpu_peak = future["util_CPU"].max()
        memory_peak = future["util_memory"].max()

        cpu_high = cpu_peak >= HIGH_THRESHOLD
        memory_high = memory_peak >= HIGH_THRESHOLD

        if cpu_high and memory_high:
            driver = "CPU_AND_MEMORY"
        elif cpu_high:
            driver = "CPU"
        elif memory_high:
            driver = "MEMORY"
        else:
            driver = "CHECK_EVENT_LABEL"

        row = {
            "event_id": event.event_id,
            "machineID": machine_id,
            "event_start_step": start,
            "event_end_step": end,
            "event_duration_minutes": (
                end - start + 1
            ) * STEP_MINUTES,

            "event_driver": driver,

            "event_cpu_peak": cpu_peak,
            "event_memory_peak": memory_peak,

            "event_cpu_mean": future["util_CPU"].mean(),
            "event_memory_mean": future["util_memory"].mean(),

            "history_rows": len(history),
        }

        row.update(features)

        row.update(
            get_model_event_metrics(
                baseline,
                machine_id,
                start,
                end,
                "baseline",
            )
        )

        row.update(
            get_model_event_metrics(
                enhanced,
                machine_id,
                start,
                end,
                "enhanced",
            )
        )

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------
# SUMMARIES
# ---------------------------------------------------------

def create_summaries(details):
    if details.empty:
        raise ValueError("No missed events were analyzed.")

    driver_summary = (
        details.groupby("event_driver")
        .agg(
            event_count=("event_id", "count"),
            average_duration=("event_duration_minutes", "mean"),
            average_cpu_peak=("event_cpu_peak", "mean"),
            average_memory_peak=("event_memory_peak", "mean"),
            average_history_cpu=("util_CPU_mean", "mean"),
            average_history_memory=("util_memory_mean", "mean"),
            average_cpu_recent_change=("util_CPU_recent_change", "mean"),
            average_memory_recent_change=("util_memory_recent_change", "mean"),
        )
        .reset_index()
    )

    details["memory_rise_category"] = pd.cut(
        details["util_memory_recent_change"],
        bins=[-np.inf, 0, 5, 15, np.inf],
        labels=[
            "DECLINING_OR_STABLE",
            "SMALL_RISE",
            "MODERATE_RISE",
            "SHARP_RISE",
        ],
    )

    memory_summary = (
        details.groupby(
            "memory_rise_category",
            observed=False,
        )
        .agg(
            event_count=("event_id", "count"),
            average_memory_peak=("event_memory_peak", "mean"),
            average_memory_history=("util_memory_mean", "mean"),
            average_baseline_probability=(
                "baseline_max_high_probability", "mean"
            ),
            average_enhanced_probability=(
                "enhanced_max_high_probability", "mean"
            ),
        )
        .reset_index()
    )

    return driver_summary, memory_summary


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, baseline, enhanced, agreement = load_data()

    missed = agreement[
        agreement["agreement"] == "BOTH_MISSED"
    ].copy()

    if missed.empty:
        logger.info("No events missed by both models.")
        return

    logger.info("Investigating %s jointly missed events.", len(missed))

    details = analyze_missed_events(
        raw,
        baseline,
        enhanced,
        missed,
    )

    if details.empty:
        raise ValueError("No event details were generated.")

    driver_summary, memory_summary = create_summaries(details)

    details.to_csv(
        OUTPUT_DIR / "missed_event_details.csv",
        index=False,
    )

    driver_summary.to_csv(
        OUTPUT_DIR / "missed_event_driver_summary.csv",
        index=False,
    )

    memory_summary.to_csv(
        OUTPUT_DIR / "missed_event_memory_summary.csv",
        index=False,
    )

    logger.info(
        "\nMISSED EVENT DRIVER SUMMARY\n%s",
        driver_summary.to_string(index=False),
    )

    logger.info(
        "\nMEMORY RISE SUMMARY\n%s",
        memory_summary.to_string(index=False),
    )

    logger.info(
        "\nMISSED EVENT PROBABILITIES\n%s",
        details[
            [
                "event_id",
                "machineID",
                "event_start_step",
                "event_driver",
                "event_duration_minutes",
                "baseline_max_high_probability",
                "enhanced_max_high_probability",
            ]
        ].to_string(index=False),
    )

    logger.info("=" * 65)
    logger.info("MISSED EVENT ANALYSIS COMPLETED")
    logger.info("Analyzed events: %s", len(details))
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
