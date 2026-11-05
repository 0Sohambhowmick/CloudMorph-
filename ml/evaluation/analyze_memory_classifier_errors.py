
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "memory_high_demand_classifier"
)

PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"

OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("\n" + "=" * 65)
    print("CLOUDMORPH MEMORY CLASSIFIER ERROR ANALYSIS")
    print("=" * 65)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    df = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "machineID",
        "actual_high_memory",
        "predicted_high_memory",
        "predicted_probability",
        "actual_peak_memory",
        "alert_threshold",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --------------------------------------------------
    # FALSE POSITIVES AND FALSE NEGATIVES
    # --------------------------------------------------

    false_negatives = df[
        (df["actual_high_memory"] == 1)
        & (df["predicted_high_memory"] == 0)
    ].copy()

    false_positives = df[
        (df["actual_high_memory"] == 0)
        & (df["predicted_high_memory"] == 1)
    ].copy()

    true_positives = df[
        (df["actual_high_memory"] == 1)
        & (df["predicted_high_memory"] == 1)
    ].copy()

    print("\n1. ERROR SUMMARY")

    print(f"Total test samples: {len(df)}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"False positives: {len(false_positives)}")
    print(f"True positives: {len(true_positives)}")

    # --------------------------------------------------
    # FALSE NEGATIVE DETAILS
    # --------------------------------------------------

    print("\n2. FALSE NEGATIVE DETAILS")

    if false_negatives.empty:
        print("No false negatives.")
    else:
        print(
            false_negatives[
                [
                    "machineID",
                    "actual_peak_memory",
                    "predicted_probability",
                    "alert_threshold",
                ]
            ]
            .sort_values(
                "actual_peak_memory",
                ascending=False,
            )
            .to_string(index=False)
        )

    # --------------------------------------------------
    # MACHINE-WISE FALSE NEGATIVES
    # --------------------------------------------------

    print("\n3. MACHINE-WISE FALSE NEGATIVES")

    if not false_negatives.empty:

        fn_by_machine = (
            false_negatives
            .groupby("machineID")
            .agg(
                missed_cases=("machineID", "size"),
                max_actual_memory=("actual_peak_memory", "max"),
                mean_probability=("predicted_probability", "mean"),
            )
            .sort_values(
                "missed_cases",
                ascending=False,
            )
        )

        print(fn_by_machine.to_string())

        fn_by_machine.to_csv(
            OUTPUT_DIR / "memory_false_negatives_by_machine.csv"
        )

    # --------------------------------------------------
    # FALSE POSITIVE DETAILS
    # --------------------------------------------------

    print("\n4. FALSE POSITIVE SUMMARY")

    if not false_positives.empty:

        print(
            false_positives[
                [
                    "machineID",
                    "actual_peak_memory",
                    "predicted_probability",
                ]
            ]
            .sort_values(
                "predicted_probability",
                ascending=False,
            )
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------

    print("\n5. ALERT THRESHOLD COMPARISON")

    y_true = df["actual_high_memory"].to_numpy()

    probabilities = df[
        "predicted_probability"
    ].to_numpy()

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.40302382398153497,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "alerts": int(predictions.sum()),
            "false_positives": int(
                ((y_true == 0) & (predictions == 1)).sum()
            ),
            "false_negatives": int(
                ((y_true == 1) & (predictions == 0)).sum()
            ),
        })

    threshold_df = pd.DataFrame(rows)

    print(
        threshold_df.round(4).to_string(index=False)
    )

    threshold_df.to_csv(
        OUTPUT_DIR / "memory_classifier_threshold_analysis.csv",
        index=False,
    )

    # --------------------------------------------------
    # ACTUAL MEMORY DISTRIBUTION
    # --------------------------------------------------

    print("\n6. ACTUAL MEMORY PEAK DISTRIBUTION")

    print(
        df.groupby("actual_high_memory")[
            "actual_peak_memory"
        ].describe().round(2)
    )

    # --------------------------------------------------
    # SAVE ERROR RECORDS
    # --------------------------------------------------

    false_negatives.to_csv(
        OUTPUT_DIR / "memory_classifier_false_negatives.csv",
        index=False,
    )

    false_positives.to_csv(
        OUTPUT_DIR / "memory_classifier_false_positives.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nGenerated files:")
    print("1. memory_false_negatives_by_machine.csv")
    print("2. memory_classifier_threshold_analysis.csv")
    print("3. memory_classifier_false_negatives.csv")
    print("4. memory_classifier_false_positives.csv")


if __name__ == "__main__":
    main()
