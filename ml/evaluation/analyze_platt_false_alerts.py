
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

PLATT_DIR = (
    ROOT / "ml" / "models" / "lstm_high_risk_platt"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "lstm_platt_false_alert_analysis"
)

KEYS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_end_step",
]

HIGH_LABEL = "HIGH_DEMAND"

ORIGINAL_THRESHOLD = 0.11
PLATT_THRESHOLD = 0.01


# =========================================================
# LOAD FILES
# =========================================================

def load_predictions():

    original_path = (
        ORIGINAL_DIR / "test_predictions.csv"
    )

    platt_path = (
        PLATT_DIR / "test_predictions_platt.csv"
    )

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    if not platt_path.exists():
        raise FileNotFoundError(platt_path)

    original = pd.read_csv(original_path)
    platt = pd.read_csv(platt_path)

    original["machineID"] = (
        original["machineID"].astype(str)
    )

    platt["machineID"] = (
        platt["machineID"].astype(str)
    )

    for df in [original, platt]:

        for col in [
            "history_start_step",
            "history_end_step",
            "target_end_step",
        ]:
            df[col] = df[col].astype(int)

    required_original = (
        KEYS
        + [
            "target",
            "high_demand_probability",
        ]
    )

    required_platt = (
        KEYS
        + [
            "target",
            "original_high_probability",
            "platt_high_probability",
        ]
    )

    missing_original = (
        set(required_original) - set(original.columns)
    )

    missing_platt = (
        set(required_platt) - set(platt.columns)
    )

    if missing_original:
        raise ValueError(
            f"Original file missing columns: {missing_original}"
        )

    if missing_platt:
        raise ValueError(
            f"Platt file missing columns: {missing_platt}"
        )

    if original.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate original test-window identifiers."
        )

    if platt.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate Platt test-window identifiers."
        )

    merged = original.merge(
        platt,
        on=KEYS,
        how="inner",
        suffixes=("_original", "_platt"),
        validate="one_to_one",
    )

    if len(merged) != len(original) or len(merged) != len(platt):
        raise ValueError(
            "Original and Platt test windows do not match exactly."
        )

    if not (
        merged["target_original"]
        == merged["target_platt"]
    ).all():
        raise ValueError(
            "Actual labels differ between the prediction files."
        )

    merged["actual_high"] = (
        merged["target_original"] == HIGH_LABEL
    )

    return merged


# =========================================================
# PREPARE ALERT FLAGS
# =========================================================

def prepare_alerts(df):

    df["original_probability"] = (
        df["high_demand_probability"]
    )

    df["platt_probability"] = (
        df["platt_high_probability"]
    )

    df["original_alert"] = (
        df["original_probability"] >= ORIGINAL_THRESHOLD
    )

    df["platt_alert"] = (
        df["platt_probability"] >= PLATT_THRESHOLD
    )

    df["original_false_positive"] = (
        df["original_alert"] & ~df["actual_high"]
    )

    df["platt_false_positive"] = (
        df["platt_alert"] & ~df["actual_high"]
    )

    df["original_true_positive"] = (
        df["original_alert"] & df["actual_high"]
    )

    df["platt_true_positive"] = (
        df["platt_alert"] & df["actual_high"]
    )

    df["original_false_negative"] = (
        ~df["original_alert"] & df["actual_high"]
    )

    df["platt_false_negative"] = (
        ~df["platt_alert"] & df["actual_high"]
    )

    return df


# =========================================================
# BASIC RISK METRICS
# =========================================================

def risk_metrics(df, alert_column):

    actual = df["actual_high"].astype(bool)
    predicted = df[alert_column].astype(bool)

    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0.0
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
# PROBABILITY DISTRIBUTION
# =========================================================

def probability_distribution(
    df,
    mask_column,
    probability_column,
    group_name,
):

    selected = df[df[mask_column]].copy()

    if selected.empty:
        return {
            "group": group_name,
            "count": 0,
        }

    values = selected[probability_column]

    return {
        "group": group_name,
        "count": int(len(selected)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "std": float(values.std()),
        "at_least_50_percent": int(
            (values >= 0.50).sum()
        ),
        "at_least_90_percent": int(
            (values >= 0.90).sum()
        ),
        "at_least_95_percent": int(
            (values >= 0.95).sum()
        ),
        "at_least_99_percent": int(
            (values >= 0.99).sum()
        ),
    }


# =========================================================
# MACHINE-LEVEL ANALYSIS
# =========================================================

def machine_analysis(
    df,
    mask_column,
    probability_column,
):

    selected = df[df[mask_column]].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "machineID",
                "false_positive_windows",
                "mean_probability",
                "median_probability",
                "max_probability",
            ]
        )

    result = (
        selected.groupby("machineID")
        .agg(
            false_positive_windows=("machineID", "size"),
            mean_probability=(probability_column, "mean"),
            median_probability=(probability_column, "median"),
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
# CONSECUTIVE FALSE-ALERT SEQUENCES
# =========================================================

def analyze_sequences(
    df,
    false_positive_column,
    probability_column,
    model_name,
):

    selected = df[
        df[false_positive_column]
    ].copy()

    if selected.empty:
        return pd.DataFrame()

    selected = selected.sort_values(
        ["machineID", "history_start_step"]
    )

    records = []

    for machine_id, group in selected.groupby("machineID"):

        group = group.sort_values(
            "history_start_step"
        )

        current = []

        for row in group.itertuples(index=False):

            start = row.history_start_step

            if (
                current
                and start != current[-1]["start"] + 1
            ):

                records.append(
                    build_sequence_record(
                        model_name,
                        machine_id,
                        current,
                    )
                )

                current = []

            current.append({
                "start": start,
                "target_end": row.target_end_step,
                "probability": getattr(
                    row,
                    probability_column,
                ),
            })

        if current:

            records.append(
                build_sequence_record(
                    model_name,
                    machine_id,
                    current,
                )
            )

    return pd.DataFrame(records)


def build_sequence_record(
    model_name,
    machine_id,
    sequence,
):

    probabilities = [
        row["probability"]
        for row in sequence
    ]

    return {
        "model": model_name,
        "machineID": machine_id,
        "sequence_length": len(sequence),
        "first_history_start_step": sequence[0]["start"],
        "last_history_start_step": sequence[-1]["start"],
        "first_target_end_step": sequence[0]["target_end"],
        "last_target_end_step": sequence[-1]["target_end"],
        "mean_probability": float(
            np.mean(probabilities)
        ),
        "max_probability": float(
            np.max(probabilities)
        ),
    }


# =========================================================
# ALERT AGREEMENT
# =========================================================

def alert_agreement(df):

    result = (
        df.groupby(
            ["original_alert", "platt_alert"]
        )
        .agg(
            windows=("machineID", "size"),
            actual_high_windows=("actual_high", "sum"),
        )
        .reset_index()
    )

    def identify(row):

        if row["original_alert"] and row["platt_alert"]:
            return "BOTH_ALERT"

        if row["original_alert"]:
            return "ORIGINAL_ONLY"

        if row["platt_alert"]:
            return "PLATT_ONLY"

        return "NEITHER_ALERT"

    result["agreement_group"] = result.apply(
        identify,
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

    print("Loading original and Platt predictions...")

    df = load_predictions()

    df = prepare_alerts(df)

    print(
        f"Matched test windows: {len(df):,}"
    )

    # Risk metrics.
    original_metrics = risk_metrics(
        df,
        "original_alert",
    )

    platt_metrics = risk_metrics(
        df,
        "platt_alert",
    )

    metrics = pd.DataFrame([
        {
            "model": "Original LSTM",
            **original_metrics,
        },
        {
            "model": "Platt Calibrated LSTM",
            **platt_metrics,
        },
    ])

    metrics.to_csv(
        OUTPUT_DIR / "risk_metrics_comparison.csv",
        index=False,
    )

    # Probability distributions.
    distributions = []

    distribution_specs = [
        (
            "original_false_positive",
            "original_probability",
            "Original False Positives",
        ),
        (
            "platt_false_positive",
            "platt_probability",
            "Platt False Positives",
        ),
        (
            "original_true_positive",
            "original_probability",
            "Original True Positives",
        ),
        (
            "platt_true_positive",
            "platt_probability",
            "Platt True Positives",
        ),
        (
            "original_false_negative",
            "original_probability",
            "Original False Negatives",
        ),
        (
            "platt_false_negative",
            "platt_probability",
            "Platt False Negatives",
        ),
    ]

    for mask, probability, name in distribution_specs:

        distributions.append(
            probability_distribution(
                df,
                mask,
                probability,
                name,
            )
        )

    with open(
        OUTPUT_DIR / "probability_distributions.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            distributions,
            file,
            indent=4,
        )

    # Machine-level analysis.
    original_machines = machine_analysis(
        df,
        "original_false_positive",
        "original_probability",
    )

    platt_machines = machine_analysis(
        df,
        "platt_false_positive",
        "platt_probability",
    )

    original_machines.to_csv(
        OUTPUT_DIR / "original_false_positives_by_machine.csv",
        index=False,
    )

    platt_machines.to_csv(
        OUTPUT_DIR / "platt_false_positives_by_machine.csv",
        index=False,
    )

    # Repeated sequences.
    original_sequences = analyze_sequences(
        df,
        "original_false_positive",
        "original_probability",
        "Original LSTM",
    )

    platt_sequences = analyze_sequences(
        df,
        "platt_false_positive",
        "platt_probability",
        "Platt LSTM",
    )

    sequences = pd.concat(
        [
            original_sequences,
            platt_sequences,
        ],
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

    # Detailed window-level data.
    df.to_csv(
        OUTPUT_DIR / "window_level_analysis.csv",
        index=False,
    )

    # Summary.
    summary = {
        "matched_test_windows": len(df),
        "original_threshold": ORIGINAL_THRESHOLD,
        "platt_threshold": PLATT_THRESHOLD,
        "original_metrics": original_metrics,
        "platt_metrics": platt_metrics,
        "probability_distributions": distributions,
        "sequence_analysis": {
            "original_sequences": len(original_sequences),
            "platt_sequences": len(platt_sequences),
            "original_longest_sequence": (
                int(original_sequences["sequence_length"].max())
                if not original_sequences.empty else 0
            ),
            "platt_longest_sequence": (
                int(platt_sequences["sequence_length"].max())
                if not platt_sequences.empty else 0
            ),
        },
        "alert_agreement": agreement.to_dict(
            orient="records"
        ),
    }

    with open(
        OUTPUT_DIR / "platt_false_alert_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # Console output.
    print("\n" + "=" * 70)
    print("PLATT FALSE-ALERT ANALYSIS")
    print("=" * 70)

    print("\nRISK METRICS")
    print(metrics.to_string(index=False))

    print("\nPROBABILITY DISTRIBUTIONS")

    print(
        pd.DataFrame(distributions).to_string(
            index=False
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

    print("\nTOP ORIGINAL FALSE-ALERT MACHINES")

    print(
        original_machines.head(10).to_string(
            index=False
        )
    )

    print("\nTOP PLATT FALSE-ALERT MACHINES")

    print(
        platt_machines.head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
