
"""
CloudMorph - Baseline vs Enhanced Model Disagreement Analysis

Uses the already saved final test predictions.
No training, retraining, threshold tuning or production changes.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "ml" / "models"
    / "optimization_classifier"
    / "temporal_feature_evaluation"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models"
    / "optimization_classifier"
    / "model_disagreement"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphDisagreement")


def load_predictions():

    baseline = pd.read_csv(
        INPUT_DIR / "baseline_predictions.csv"
    )

    enhanced = pd.read_csv(
        INPUT_DIR / "enhanced_predictions.csv"
    )

    keys = [
        "machineID",
        "history_start_step",
        "target",
    ]

    if baseline.duplicated(keys).any():
        raise ValueError("Duplicate baseline prediction keys.")

    if enhanced.duplicated(keys).any():
        raise ValueError("Duplicate enhanced prediction keys.")

    merged = baseline.merge(
        enhanced,
        on=keys,
        suffixes=("_baseline", "_enhanced"),
        validate="one_to_one",
    )

    if len(merged) != len(baseline) or len(merged) != len(enhanced):
        raise ValueError("Prediction files do not align.")

    return merged


def add_prediction_flags(df):

    actual_high = df["target"] == "HIGH_DEMAND"

    baseline_alert = (
        df["risk_alert_baseline"] == "HIGH_RISK"
    )

    enhanced_alert = (
        df["risk_alert_enhanced"] == "HIGH_RISK"
    )

    df["baseline_alert"] = baseline_alert
    df["enhanced_alert"] = enhanced_alert

    df["alert_agreement"] = np.select(
        [
            baseline_alert & enhanced_alert,
            baseline_alert & ~enhanced_alert,
            ~baseline_alert & enhanced_alert,
        ],
        [
            "BOTH_ALERT",
            "BASELINE_ONLY_ALERT",
            "ENHANCED_ONLY_ALERT",
        ],
        default="NEITHER_ALERT",
    )

    df["baseline_risk_outcome"] = np.select(
        [
            actual_high & baseline_alert,
            actual_high & ~baseline_alert,
            ~actual_high & baseline_alert,
        ],
        [
            "TRUE_POSITIVE",
            "FALSE_NEGATIVE",
            "FALSE_POSITIVE",
        ],
        default="TRUE_NEGATIVE",
    )

    df["enhanced_risk_outcome"] = np.select(
        [
            actual_high & enhanced_alert,
            actual_high & ~enhanced_alert,
            ~actual_high & enhanced_alert,
        ],
        [
            "TRUE_POSITIVE",
            "FALSE_NEGATIVE",
            "FALSE_POSITIVE",
        ],
        default="TRUE_NEGATIVE",
    )

    df["prediction_changed"] = (
        df["predicted_workload_baseline"]
        != df["predicted_workload_enhanced"]
    )

    df["risk_probability_difference"] = (
        df["high_demand_probability_enhanced"]
        - df["high_demand_probability_baseline"]
    )

    return df


def create_summary(df):

    actual_high = df["target"] == "HIGH_DEMAND"

    rows = []

    for group in [
        "BOTH_ALERT",
        "BASELINE_ONLY_ALERT",
        "ENHANCED_ONLY_ALERT",
        "NEITHER_ALERT",
    ]:

        subset = df[
            df["alert_agreement"] == group
        ]

        rows.append({
            "alert_agreement": group,
            "total_cases": len(subset),
            "actual_high_cases": int(
                (subset["target"] == "HIGH_DEMAND").sum()
            ),
            "actual_non_high_cases": int(
                (subset["target"] != "HIGH_DEMAND").sum()
            ),
            "mean_baseline_probability": (
                subset["high_demand_probability_baseline"].mean()
            ),
            "mean_enhanced_probability": (
                subset["high_demand_probability_enhanced"].mean()
            ),
        })

    return pd.DataFrame(rows)


def create_outcome_comparison(df):

    rows = []

    for model in ["baseline", "enhanced"]:

        column = f"{model}_risk_outcome"

        counts = df[column].value_counts()

        rows.append({
            "model": model.upper(),
            "true_positives": int(counts.get("TRUE_POSITIVE", 0)),
            "false_positives": int(counts.get("FALSE_POSITIVE", 0)),
            "false_negatives": int(counts.get("FALSE_NEGATIVE", 0)),
            "true_negatives": int(counts.get("TRUE_NEGATIVE", 0)),
        })

    return pd.DataFrame(rows)


def main():

    logger.info("Loading saved predictions...")

    df = load_predictions()

    df = add_prediction_flags(df)

    summary = create_summary(df)

    outcome_comparison = create_outcome_comparison(df)

    # High-demand cases detected by baseline but not enhanced.
    baseline_only_high = df[
        (df["target"] == "HIGH_DEMAND")
        & df["baseline_alert"]
        & ~df["enhanced_alert"]
    ].copy()

    # High-demand cases detected by enhanced but not baseline.
    enhanced_only_high = df[
        (df["target"] == "HIGH_DEMAND")
        & ~df["baseline_alert"]
        & df["enhanced_alert"]
    ].copy()

    # High-demand cases missed by both.
    both_missed = df[
        (df["target"] == "HIGH_DEMAND")
        & ~df["baseline_alert"]
        & ~df["enhanced_alert"]
    ].copy()

    # False alerts generated by either model.
    disagreement_false_alerts = df[
        (df["target"] != "HIGH_DEMAND")
        & (
            df["baseline_alert"]
            | df["enhanced_alert"]
        )
    ].copy()

    # Cases where workload classification changed.
    changed_predictions = df[
        df["prediction_changed"]
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "alert_agreement_summary.csv": summary,
        "risk_outcome_comparison.csv": outcome_comparison,
        "baseline_only_high_cases.csv": baseline_only_high,
        "enhanced_only_high_cases.csv": enhanced_only_high,
        "both_models_missed_cases.csv": both_missed,
        "disagreement_false_alerts.csv": disagreement_false_alerts,
        "changed_workload_predictions.csv": changed_predictions,
        "all_model_disagreements.csv": df,
    }

    for filename, dataframe in outputs.items():

        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

        logger.info(
            "Saved %s (%s rows)",
            filename,
            len(dataframe),
        )

    logger.info("ALERT AGREEMENT")

    logger.info(
        "\n%s",
        summary.to_string(index=False),
    )

    logger.info("RISK OUTCOME COMPARISON")

    logger.info(
        "\n%s",
        outcome_comparison.to_string(index=False),
    )

    logger.info(
        "Baseline-only high cases: %s",
        len(baseline_only_high),
    )

    logger.info(
        "Enhanced-only high cases: %s",
        len(enhanced_only_high),
    )

    logger.info(
        "Missed by both models: %s",
        len(both_missed),
    )

    logger.info("=" * 65)
    logger.info("MODEL DISAGREEMENT ANALYSIS COMPLETED")
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
