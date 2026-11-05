
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

LSTM_FILE = (
    ROOT / "ml/models/lstm_sequence_classifier/test_predictions.csv"
)

EXTRATREES_FILE = (
    ROOT
    / "ml/models/optimization_classifier/risk_evaluation/"
    / "risk_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "ml/models/lstm_extratrees_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_end_step",
]

CLASS_ORDER = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]


def normalize_keys(df):
    for column in KEYS:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        ).astype(int)

    return df


def binary_metrics(actual, predicted):
    return {
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "specificity": float(
            recall_score(
                actual,
                predicted,
                pos_label=0,
                zero_division=0,
            )
        ),
        "true_positives": int(
            ((actual == 1) & (predicted == 1)).sum()
        ),
        "false_positives": int(
            ((actual == 0) & (predicted == 1)).sum()
        ),
        "false_negatives": int(
            ((actual == 1) & (predicted == 0)).sum()
        ),
        "true_negatives": int(
            ((actual == 0) & (predicted == 0)).sum()
        ),
        "alert_count": int(predicted.sum()),
    }


def main():
    logger.info("Loading LSTM and Extra Trees predictions...")

    lstm = pd.read_csv(LSTM_FILE)
    extra = pd.read_csv(EXTRATREES_FILE)

    lstm = normalize_keys(lstm)
    extra = normalize_keys(extra)

    lstm = lstm.rename(
        columns={
            "target": "actual_lstm",
            "predicted_workload": "lstm_prediction",
            "high_demand_probability": "lstm_high_probability",
            "risk_threshold": "lstm_threshold",
            "risk_alert": "lstm_risk_alert",
        }
    )

    extra = extra.rename(
        columns={
            "actual_workload": "actual_extra",
            "predicted_workload": "extra_prediction",
            "high_demand_probability": "extra_high_probability",
            "risk_threshold": "extra_threshold",
            "risk_alert": "extra_risk_alert",
        }
    )

    logger.info("Matching identical machine/time windows...")

    df = lstm.merge(
        extra,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(df) == 0:
        raise ValueError("No matching prediction rows found.")

    if len(df) != len(lstm) or len(df) != len(extra):
        raise ValueError(
            "Prediction files do not have identical coverage. "
            f"LSTM={len(lstm)}, Extra Trees={len(extra)}, "
            f"Matched={len(df)}"
        )

    if not (df["actual_lstm"] == df["actual_extra"]).all():
        raise ValueError(
            "Actual workload labels do not match between models."
        )

    df["actual_workload"] = df["actual_lstm"]

    df["lstm_high_alert"] = (
        df["lstm_risk_alert"] == "HIGH_RISK"
    ).astype(int)

    df["extra_high_alert"] = (
        df["extra_risk_alert"] == "HIGH_RISK"
    ).astype(int)

    df["actual_high"] = (
        df["actual_workload"] == "HIGH_DEMAND"
    ).astype(int)

    logger.info("Matched rows: %s", len(df))

    # --------------------------------------------------
    # 1. Classification comparison
    # --------------------------------------------------

    classification_rows = []

    for model_name, prediction_column in [
        ("Extra Trees", "extra_prediction"),
        ("LSTM", "lstm_prediction"),
    ]:
        actual = df["actual_workload"]
        predicted = df[prediction_column]

        classification_rows.append({
            "model": model_name,
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "macro_f1": f1_score(
                actual,
                predicted,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            ),
            "high_precision": precision_score(
                actual == "HIGH_DEMAND",
                predicted == "HIGH_DEMAND",
                zero_division=0,
            ),
            "high_recall": recall_score(
                actual == "HIGH_DEMAND",
                predicted == "HIGH_DEMAND",
                zero_division=0,
            ),
            "high_f1": f1_score(
                actual == "HIGH_DEMAND",
                predicted == "HIGH_DEMAND",
                zero_division=0,
            ),
        })

        report = classification_report(
            actual,
            predicted,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR
            / f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
        )

        pd.DataFrame(
            confusion_matrix(
                actual,
                predicted,
                labels=CLASS_ORDER,
            ),
            index=CLASS_ORDER,
            columns=CLASS_ORDER,
        ).to_csv(
            OUTPUT_DIR
            / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
        )

    classification_df = pd.DataFrame(classification_rows)
    classification_df.to_csv(
        OUTPUT_DIR / "classification_comparison.csv",
        index=False,
    )

    # --------------------------------------------------
    # 2. Independent risk alert comparison
    # --------------------------------------------------

    actual = df["actual_high"].to_numpy()

    lstm_alerts = df["lstm_high_alert"].to_numpy()
    extra_alerts = df["extra_high_alert"].to_numpy()

    risk_comparison = {
        "Extra Trees": binary_metrics(actual, extra_alerts),
        "LSTM": binary_metrics(actual, lstm_alerts),
    }

    with open(
        OUTPUT_DIR / "risk_comparison_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            risk_comparison,
            file,
            indent=4,
        )

    risk_rows = []

    for model_name, metrics in risk_comparison.items():
        risk_rows.append({
            "model": model_name,
            **metrics,
        })

    pd.DataFrame(risk_rows).to_csv(
        OUTPUT_DIR / "risk_comparison.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. Alert agreement
    # --------------------------------------------------

    conditions = [
        (df["lstm_high_alert"] == 1)
        & (df["extra_high_alert"] == 1),

        (df["lstm_high_alert"] == 1)
        & (df["extra_high_alert"] == 0),

        (df["lstm_high_alert"] == 0)
        & (df["extra_high_alert"] == 1),

        (df["lstm_high_alert"] == 0)
        & (df["extra_high_alert"] == 0),
    ]

    groups = [
        "BOTH_ALERT",
        "LSTM_ONLY_ALERT",
        "EXTRA_TREES_ONLY_ALERT",
        "NEITHER_ALERT",
    ]

    df["alert_agreement"] = np.select(
        conditions,
        groups,
        default="UNKNOWN",
    )

    agreement = (
        df.groupby("alert_agreement")
        .agg(
            cases=("actual_high", "size"),
            actual_high=("actual_high", "sum"),
        )
        .reset_index()
    )

    agreement["non_high"] = (
        agreement["cases"] - agreement["actual_high"]
    )

    agreement.to_csv(
        OUTPUT_DIR / "alert_agreement.csv",
        index=False,
    )

    # --------------------------------------------------
    # 4. Workload classification agreement
    # --------------------------------------------------

    df["classification_agreement"] = (
        df["lstm_prediction"] == df["extra_prediction"]
    )

    workload_agreement = (
        df.groupby(
            ["actual_workload", "classification_agreement"]
        )
        .size()
        .reset_index(name="cases")
    )

    workload_agreement.to_csv(
        OUTPUT_DIR / "workload_agreement.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Important case files
    # --------------------------------------------------

    export_columns = KEYS + [
        "actual_workload",
        "lstm_prediction",
        "extra_prediction",
        "lstm_high_probability",
        "extra_high_probability",
        "lstm_threshold",
        "extra_threshold",
        "lstm_risk_alert",
        "extra_risk_alert",
        "alert_agreement",
    ]

    case_groups = {
        "lstm_only_alerts.csv": "LSTM_ONLY_ALERT",
        "extra_trees_only_alerts.csv": "EXTRA_TREES_ONLY_ALERT",
        "both_alerts.csv": "BOTH_ALERT",
        "both_missed_high.csv": "NEITHER_ALERT",
    }

    for filename, group in case_groups.items():
        cases = df[df["alert_agreement"] == group].copy()

        if group == "NEITHER_ALERT":
            cases = cases[cases["actual_high"] == 1]

        cases[export_columns].to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    changed = df[
        ~df["classification_agreement"]
    ]

    changed[export_columns].to_csv(
        OUTPUT_DIR / "different_workload_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. Final summary
    # --------------------------------------------------

    summary = {
        "matched_test_samples": int(len(df)),
        "actual_high_demand_cases": int(actual.sum()),
        "lstm_threshold": float(df["lstm_threshold"].iloc[0]),
        "extra_trees_threshold": float(
            df["extra_threshold"].iloc[0]
        ),
        "lstm_alerts": int(lstm_alerts.sum()),
        "extra_trees_alerts": int(extra_alerts.sum()),
        "same_workload_predictions": int(
            df["classification_agreement"].sum()
        ),
        "different_workload_predictions": int(
            (~df["classification_agreement"]).sum()
        ),
        "both_alert": int(
            (df["alert_agreement"] == "BOTH_ALERT").sum()
        ),
        "lstm_only_alert": int(
            (df["alert_agreement"] == "LSTM_ONLY_ALERT").sum()
        ),
        "extra_trees_only_alert": int(
            (df["alert_agreement"] == "EXTRA_TREES_ONLY_ALERT").sum()
        ),
        "neither_alert": int(
            (df["alert_agreement"] == "NEITHER_ALERT").sum()
        ),
    }

    with open(
        OUTPUT_DIR / "comparison_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    logger.info("COMPARISON COMPLETED")
    logger.info(
        "Matched samples: %s",
        summary["matched_test_samples"],
    )
    logger.info(
        "LSTM alerts: %s | Extra Trees alerts: %s",
        summary["lstm_alerts"],
        summary["extra_trees_alerts"],
    )
    logger.info(
        "LSTM-only alerts: %s | Extra Trees-only alerts: %s",
        summary["lstm_only_alert"],
        summary["extra_trees_only_alert"],
    )
    logger.info(
        "Different workload predictions: %s",
        summary["different_workload_predictions"],
    )
    logger.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
