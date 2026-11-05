
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DIR = (
    ROOT / "ml" / "models" / "lstm_sequence_classifier"
)

BALANCED_DIR = (
    ROOT / "ml" / "models" / "lstm_sequence_classifier_balanced"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "lstm_false_alert_comparison"
)

KEYS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_end_step",
]

ACTUAL_HIGH = "HIGH_DEMAND"

ORIGINAL_THRESHOLD = 0.11
BALANCED_THRESHOLD = 0.03


# =========================================================
# LOAD PREDICTIONS
# =========================================================

def load_predictions(directory, model_name):
    path = directory / "test_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = KEYS + [
        "target",
        "predicted_workload",
        "high_demand_probability",
        "risk_alert",
        "probability_HIGH_DEMAND",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} is missing columns: {missing}"
        )

    df["machineID"] = df["machineID"].astype(str)

    for col in [
        "history_start_step",
        "history_end_step",
        "target_end_step",
    ]:
        df[col] = df[col].astype(int)

    df["high_demand_probability"] = pd.to_numeric(
        df["high_demand_probability"],
        errors="raise",
    )

    if df.duplicated(KEYS).any():
        raise ValueError(
            f"Duplicate test-window identifiers in {model_name}"
        )

    print(
        f"{model_name}: loaded {len(df):,} predictions"
    )

    return df


# =========================================================
# ALIGN BOTH MODELS ON IDENTICAL WINDOWS
# =========================================================

def align_predictions(original, balanced):
    merged = original.merge(
        balanced,
        on=KEYS,
        how="inner",
        suffixes=("_original", "_balanced"),
        validate="one_to_one",
    )

    if len(merged) != len(original) or len(merged) != len(balanced):
        raise ValueError(
            "Prediction windows do not match exactly. "
            f"Original={len(original)}, "
            f"Balanced={len(balanced)}, "
            f"Matched={len(merged)}"
        )

    if not (
        merged["target_original"]
        == merged["target_balanced"]
    ).all():
        raise ValueError(
            "Actual labels differ between model prediction files."
        )

    merged["actual_high"] = (
        merged["target_original"] == ACTUAL_HIGH
    )

    merged["original_alert"] = (
        merged["high_demand_probability_original"]
        >= ORIGINAL_THRESHOLD
    )

    merged["balanced_alert"] = (
        merged["high_demand_probability_balanced"]
        >= BALANCED_THRESHOLD
    )

    merged["original_false_positive"] = (
        merged["original_alert"]
        & ~merged["actual_high"]
    )

    merged["balanced_false_positive"] = (
        merged["balanced_alert"]
        & ~merged["actual_high"]
    )

    merged["original_false_negative"] = (
        ~merged["original_alert"]
        & merged["actual_high"]
    )

    merged["balanced_false_negative"] = (
        ~merged["balanced_alert"]
        & merged["actual_high"]
    )

    return merged


# =========================================================
# MODEL METRICS
# =========================================================

def calculate_metrics(df, alert_column):
    actual = df["actual_high"].astype(bool)
    predicted = df[alert_column].astype(bool)

    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())

    precision = (
        tp / (tp + fp) if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn) if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    specificity = (
        tn / (tn + fp) if tn + fp else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


# =========================================================
# FALSE ALERT PROBABILITY ANALYSIS
# =========================================================

def probability_analysis(df, false_positive_column, probability_column):
    fp = df[df[false_positive_column]].copy()

    if fp.empty:
        return {
            "false_positive_windows": 0,
        }

    probabilities = fp[probability_column]

    return {
        "false_positive_windows": int(len(fp)),
        "unique_machines": int(fp["machineID"].nunique()),
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(probabilities.median()),
        "minimum_probability": float(probabilities.min()),
        "maximum_probability": float(probabilities.max()),
        "probability_at_least_90_percent": int(
            (probabilities >= 0.90).sum()
        ),
        "probability_at_least_95_percent": int(
            (probabilities >= 0.95).sum()
        ),
        "probability_at_least_99_percent": int(
            (probabilities >= 0.99).sum()
        ),
    }


# =========================================================
# FALSE ALERTS BY MACHINE
# =========================================================

def machine_analysis(df, false_positive_column, probability_column):
    fp = df[df[false_positive_column]].copy()

    if fp.empty:
        return pd.DataFrame(
            columns=[
                "machineID",
                "false_positive_windows",
                "mean_probability",
                "max_probability",
            ]
        )

    result = (
        fp.groupby("machineID")
        .agg(
            false_positive_windows=("machineID", "size"),
            mean_probability=(probability_column, "mean"),
            max_probability=(probability_column, "max"),
        )
        .reset_index()
        .sort_values(
            "false_positive_windows",
            ascending=False,
        )
    )

    return result


# =========================================================
# CONSECUTIVE FALSE ALERT SEQUENCES
# =========================================================

def find_false_alert_sequences(
    df,
    false_positive_column,
    probability_column,
    model_name,
):
    fp = df[df[false_positive_column]].copy()

    if fp.empty:
        return pd.DataFrame()

    fp = fp.sort_values(
        ["machineID", "history_start_step"]
    )

    sequences = []

    for machine_id, group in fp.groupby("machineID"):
        group = group.sort_values("history_start_step")

        current = []

        for row in group.itertuples(index=False):
            start = row.history_start_step

            if (
                current
                and start != current[-1]["history_start_step"] + 1
            ):
                sequences.append(
                    make_sequence_record(
                        machine_id,
                        current,
                        probability_column,
                        model_name,
                    )
                )

                current = []

            current.append({
                "history_start_step": start,
                "target_end_step": row.target_end_step,
                "probability": getattr(
                    row,
                    probability_column,
                ),
            })

        if current:
            sequences.append(
                make_sequence_record(
                    machine_id,
                    current,
                    probability_column,
                    model_name,
                )
            )

    return pd.DataFrame(sequences)


def make_sequence_record(
    machine_id,
    records,
    probability_column,
    model_name,
):
    probabilities = [
        record["probability"]
        for record in records
    ]

    return {
        "model": model_name,
        "machineID": machine_id,
        "sequence_length": len(records),
        "first_history_start_step": (
            records[0]["history_start_step"]
        ),
        "last_history_start_step": (
            records[-1]["history_start_step"]
        ),
        "first_target_end_step": (
            records[0]["target_end_step"]
        ),
        "last_target_end_step": (
            records[-1]["target_end_step"]
        ),
        "mean_probability": float(np.mean(probabilities)),
        "max_probability": float(np.max(probabilities)),
    }


# =========================================================
# ALERT AGREEMENT
# =========================================================

def alert_agreement(df):
    result = (
        df.groupby(
            ["original_alert", "balanced_alert"],
            dropna=False,
        )
        .agg(
            windows=("machineID", "size"),
            actual_high_windows=("actual_high", "sum"),
        )
        .reset_index()
    )

    result["agreement_group"] = result.apply(
        lambda row: (
            "BOTH_ALERT"
            if row["original_alert"] and row["balanced_alert"]
            else "ORIGINAL_ONLY"
            if row["original_alert"]
            else "BALANCED_ONLY"
            if row["balanced_alert"]
            else "NEITHER_ALERT"
        ),
        axis=1,
    )

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original = load_predictions(
        ORIGINAL_DIR,
        "Original LSTM",
    )

    balanced = load_predictions(
        BALANCED_DIR,
        "Moderate LSTM",
    )

    df = align_predictions(
        original,
        balanced,
    )

    print(
        f"\nMatched test windows: {len(df):,}"
    )

    # Model risk metrics.
    original_metrics = calculate_metrics(
        df,
        "original_alert",
    )

    balanced_metrics = calculate_metrics(
        df,
        "balanced_alert",
    )

    metrics = pd.DataFrame([
        {
            "model": "Original LSTM",
            **original_metrics,
        },
        {
            "model": "Moderate LSTM",
            **balanced_metrics,
        },
    ])

    metrics.to_csv(
        OUTPUT_DIR / "risk_metrics_comparison.csv",
        index=False,
    )

    # Probability analysis.
    original_probability = probability_analysis(
        df,
        "original_false_positive",
        "high_demand_probability_original",
    )

    balanced_probability = probability_analysis(
        df,
        "balanced_false_positive",
        "high_demand_probability_balanced",
    )

    with open(
        OUTPUT_DIR / "false_positive_probability_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "original_lstm": original_probability,
                "moderate_lstm": balanced_probability,
            },
            file,
            indent=4,
        )

    # False positives by machine.
    original_machines = machine_analysis(
        df,
        "original_false_positive",
        "high_demand_probability_original",
    )

    balanced_machines = machine_analysis(
        df,
        "balanced_false_positive",
        "high_demand_probability_balanced",
    )

    original_machines.to_csv(
        OUTPUT_DIR / "original_false_positives_by_machine.csv",
        index=False,
    )

    balanced_machines.to_csv(
        OUTPUT_DIR / "balanced_false_positives_by_machine.csv",
        index=False,
    )

    # Repeated false alert sequences.
    original_sequences = find_false_alert_sequences(
        df,
        "original_false_positive",
        "high_demand_probability_original",
        "Original LSTM",
    )

    balanced_sequences = find_false_alert_sequences(
        df,
        "balanced_false_positive",
        "high_demand_probability_balanced",
        "Moderate LSTM",
    )

    sequences = pd.concat(
        [original_sequences, balanced_sequences],
        ignore_index=True,
    )

    sequences.to_csv(
        OUTPUT_DIR / "false_alert_sequences.csv",
        index=False,
    )

    # Alert agreement.
    agreement = alert_agreement(df)

    agreement.to_csv(
        OUTPUT_DIR / "alert_agreement.csv",
        index=False,
    )

    # Window-level comparison.
    comparison_columns = (
        KEYS
        + [
            "target_original",
            "high_demand_probability_original",
            "original_alert",
            "original_false_positive",
            "original_false_negative",
            "target_balanced",
            "high_demand_probability_balanced",
            "balanced_alert",
            "balanced_false_positive",
            "balanced_false_negative",
        ]
    )

    df[comparison_columns].to_csv(
        OUTPUT_DIR / "window_level_comparison.csv",
        index=False,
    )

    # Summary.
    summary = {
        "matched_test_windows": len(df),
        "original_lstm": {
            "threshold": ORIGINAL_THRESHOLD,
            "metrics": original_metrics,
            "false_positive_analysis": original_probability,
        },
        "moderate_lstm": {
            "threshold": BALANCED_THRESHOLD,
            "metrics": balanced_metrics,
            "false_positive_analysis": balanced_probability,
        },
        "alert_agreement": agreement.to_dict(
            orient="records"
        ),
        "sequence_analysis": {
            "original_false_alert_sequences": len(
                original_sequences
            ),
            "balanced_false_alert_sequences": len(
                balanced_sequences
            ),
            "original_longest_sequence": (
                int(original_sequences["sequence_length"].max())
                if not original_sequences.empty else 0
            ),
            "balanced_longest_sequence": (
                int(balanced_sequences["sequence_length"].max())
                if not balanced_sequences.empty else 0
            ),
        },
    }

    with open(
        OUTPUT_DIR / "comparison_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # Console output.
    print("\n" + "=" * 65)
    print("LSTM FALSE ALERT COMPARISON")
    print("=" * 65)

    print("\nRISK METRICS")
    print(metrics.to_string(index=False))

    print("\nFALSE POSITIVE PROBABILITY ANALYSIS")
    print(
        json.dumps(
            {
                "Original LSTM": original_probability,
                "Moderate LSTM": balanced_probability,
            },
            indent=4,
        )
    )

    print("\nALERT AGREEMENT")
    print(agreement.to_string(index=False))

    print("\nSEQUENCE ANALYSIS")
    print(
        json.dumps(
            summary["sequence_analysis"],
            indent=4,
        )
    )

    print("\nMOST AFFECTED MACHINES - ORIGINAL LSTM")
    print(original_machines.head(10).to_string(index=False))

    print("\nMOST AFFECTED MACHINES - MODERATE LSTM")
    print(balanced_machines.head(10).to_string(index=False))

    print("\n" + "=" * 65)
    print("COMPARISON COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
