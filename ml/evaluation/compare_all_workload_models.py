
from pathlib import Path

import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "ml" / "models"

OUTPUT_DIR = (
    MODEL_DIR / "all_workload_models_comparison"
)

KEYS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_end_step",
]

HIGH_LABEL = "HIGH_DEMAND"

# Thresholds are taken from the saved experiment reports.
THRESHOLDS = {
    "Extra Trees": 0.30,
    "Original LSTM": 0.11,
    "Balanced LSTM": 0.03,
    "Trend-aware LSTM": 0.60,
    "Temperature LSTM": 0.24,
    "Platt LSTM": 0.01,
}


# =========================================================
# FILE PATHS
# =========================================================

PATHS = {
    "Extra Trees": (
        MODEL_DIR
        / "optimization_classifier"
        / "risk_evaluation"
        / "risk_predictions.csv"
    ),

    "Original LSTM": (
        MODEL_DIR
        / "lstm_sequence_classifier"
        / "test_predictions.csv"
    ),

    "Balanced LSTM": (
        MODEL_DIR
        / "lstm_sequence_classifier_balanced"
        / "test_predictions.csv"
    ),

    "Trend-aware LSTM": (
        MODEL_DIR
        / "lstm_trend_aware"
        / "test_predictions.csv"
    ),

    "Temperature LSTM": (
        MODEL_DIR
        / "lstm_temperature_calibration"
        / "test_predictions_calibrated.csv"
    ),

    "Platt LSTM": (
        MODEL_DIR
        / "lstm_high_risk_platt"
        / "test_predictions_platt.csv"
    ),
}


# =========================================================
# HELPERS
# =========================================================

def normalize_keys(df):

    df = df.copy()

    df["machineID"] = (
        df["machineID"].astype(str)
    )

    for column in KEYS[1:]:

        df[column] = (
            df[column].astype(int)
        )

    return df


def calculate_metrics(actual, alerts, probabilities):

    actual = np.asarray(
        actual,
        dtype=int,
    )

    alerts = np.asarray(
        alerts,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    tn, fp, fn, tp = confusion_matrix(
        actual,
        alerts,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        actual,
        alerts,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        alerts,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        alerts,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0.0
    )

    actual_high_rate = float(
        actual.mean()
    )

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),

        "alerts": int(alerts.sum()),

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),

        "false_positive_rate": float(
            fp / (fp + tn)
            if fp + tn else 0.0
        ),

        "actual_high_windows": int(
            actual.sum()
        ),

        "actual_high_rate": actual_high_rate,

        "mean_high_probability": float(
            probabilities.mean()
        ),

        "median_high_probability": float(
            np.median(probabilities)
        ),

        "probability_std": float(
            probabilities.std()
        ),

        "probability_at_least_50": int(
            (probabilities >= 0.50).sum()
        ),

        "probability_at_least_90": int(
            (probabilities >= 0.90).sum()
        ),
    }


def load_csv(model_name):

    path = PATHS[model_name]

    if not path.exists():

        raise FileNotFoundError(
            f"Missing prediction file for {model_name}: {path}"
        )

    df = pd.read_csv(path)

    return normalize_keys(df)


# =========================================================
# LOAD COMMON REFERENCE
# =========================================================

def load_reference():

    original = load_csv(
        "Original LSTM"
    )

    required = (
        KEYS
        + [
            "target",
            "high_demand_probability",
            "risk_threshold",
        ]
    )

    missing = (
        set(required) - set(original.columns)
    )

    if missing:

        raise ValueError(
            f"Original LSTM missing columns: {missing}"
        )

    if original.duplicated(KEYS).any():

        raise ValueError(
            "Duplicate Original LSTM window identifiers."
        )

    reference = original[
        KEYS + ["target"]
    ].copy()

    reference["actual_high"] = (
        reference["target"] == HIGH_LABEL
    ).astype(int)

    return reference


# =========================================================
# MODEL-SPECIFIC PREDICTIONS
# =========================================================

def prepare_model_predictions(
    model_name,
    reference,
):

    df = load_csv(model_name)

    # Map each experiment to a common probability column.
    if model_name in [
        "Extra Trees",
        "Original LSTM",
        "Balanced LSTM",
        "Trend-aware LSTM",
    ]:

        probability_column = (
            "high_demand_probability"
        )

    elif model_name == "Temperature LSTM":

        probability_column = (
            "calibrated_high_probability"
        )

    elif model_name == "Platt LSTM":

        probability_column = (
            "platt_high_probability"
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if probability_column not in df.columns:

        raise ValueError(
            f"{model_name} missing {probability_column}"
        )

    if model_name == "Extra Trees":

        if "actual_workload" not in df.columns:

            raise ValueError(
                "Extra Trees missing actual_workload."
            )

        actual_column = "actual_workload"

    else:

        if "target" not in df.columns:

            raise ValueError(
                f"{model_name} missing target."
            )

        actual_column = "target"

    if df.duplicated(KEYS).any():

        raise ValueError(
            f"Duplicate windows in {model_name}."
        )

    merged = reference.merge(
        df,
        on=KEYS,
        how="left",
        suffixes=("_reference", "_model"),
        validate="one_to_one",
        indicator=True,
    )

    if not (
        merged["_merge"] == "both"
    ).all():

        missing_count = int(
            (merged["_merge"] != "both").sum()
        )

        raise ValueError(
            f"{model_name} is missing "
            f"{missing_count} reference windows."
        )

    if len(df) != len(reference):

        raise ValueError(
            f"{model_name} has {len(df)} windows; "
            f"reference has {len(reference)}."
        )

    # -----------------------------------------------------
    # CORRECTED LABEL VALIDATION
    # -----------------------------------------------------

    if actual_column == "actual_workload":

        # Extra Trees does not contain a target column.
        # Therefore, reference target keeps its original name.
        actual_labels = merged[
            "actual_workload"
        ]

        reference_labels = merged[
            "target"
        ]

    else:

        # Both reference and LSTM contain target.
        # Pandas creates target_reference and target_model.
        actual_labels = merged[
            "target_model"
        ]

        reference_labels = merged[
            "target_reference"
        ]

    if not (
        actual_labels.astype(str).to_numpy()
        == reference_labels.astype(str).to_numpy()
    ).all():

        raise ValueError(
            f"Actual labels do not match for {model_name}."
        )

    # -----------------------------------------------------
    # PROBABILITY VALIDATION
    # -----------------------------------------------------

    probabilities = pd.to_numeric(
        merged[probability_column],
        errors="coerce",
    ).to_numpy()

    if not np.isfinite(probabilities).all():

        raise ValueError(
            f"Invalid probabilities in {model_name}."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):

        raise ValueError(
            f"Probabilities outside [0,1] in {model_name}."
        )

    threshold = THRESHOLDS[model_name]

    alerts = (
        probabilities >= threshold
    ).astype(int)

    result = reference[KEYS].copy()

    result["model"] = model_name

    result["actual_high"] = (
        reference["actual_high"].to_numpy()
    )

    result["high_probability"] = probabilities

    result["threshold"] = threshold

    result["risk_alert"] = alerts

    result["true_positive"] = (
        (result["actual_high"] == 1)
        & (result["risk_alert"] == 1)
    )

    result["false_positive"] = (
        (result["actual_high"] == 0)
        & (result["risk_alert"] == 1)
    )

    result["false_negative"] = (
        (result["actual_high"] == 1)
        & (result["risk_alert"] == 0)
    )

    return result


# =========================================================
# CLASSIFICATION METRICS
# =========================================================

def load_classification_predictions(
    model_name,
    reference,
):

    if model_name == "Extra Trees":

        # The temporal risk evaluation file does not contain
        # the full 3-class classification prediction.
        return None

    df = load_csv(model_name)

    if model_name in [
        "Original LSTM",
        "Balanced LSTM",
        "Trend-aware LSTM",
    ]:

        prediction_column = (
            "predicted_workload"
        )

    elif model_name == "Temperature LSTM":

        prediction_column = (
            "calibrated_predicted_workload"
        )

    elif model_name == "Platt LSTM":

        # Platt calibration changes only the HIGH-risk
        # probability, not the original 3-class predictions.
        # Use the original LSTM prediction file.
        original = load_csv(
            "Original LSTM"
        )

        df = original

        prediction_column = (
            "predicted_workload"
        )

    else:

        return None

    if prediction_column not in df.columns:

        raise ValueError(
            f"{model_name} missing {prediction_column}"
        )

    merged = reference.merge(
        df[KEYS + [prediction_column]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if merged[prediction_column].isna().any():

        raise ValueError(
            f"Missing classification predictions for {model_name}."
        )

    return merged[prediction_column].to_numpy()


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("CONSOLIDATED WORKLOAD MODEL COMPARISON")
    print("=" * 75)

    reference = load_reference()

    print(
        f"Reference test windows: {len(reference):,}"
    )

    model_results = []

    detailed_predictions = []

    classification_results = []

    common_alerts = {}

    for model_name in PATHS:

        print(
            f"\nEvaluating: {model_name}"
        )

        predictions = prepare_model_predictions(
            model_name,
            reference,
        )

        detailed_predictions.append(
            predictions
        )

        common_alerts[model_name] = (
            predictions["risk_alert"].to_numpy()
        )

        metrics = calculate_metrics(
            predictions["actual_high"],
            predictions["risk_alert"],
            predictions["high_probability"],
        )

        model_results.append({

            "model": model_name,

            "threshold": THRESHOLDS[model_name],

            **metrics,

        })

        # Classification metrics are separate from binary risk metrics.
        classification_predictions = (
            load_classification_predictions(
                model_name,
                reference,
            )
        )

        if classification_predictions is not None:

            actual_labels = (
                reference["target"].to_numpy()
            )

            classification_results.append({

                "model": model_name,

                "accuracy": float(
                    accuracy_score(
                        actual_labels,
                        classification_predictions,
                    )
                ),

                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        actual_labels,
                        classification_predictions,
                    )
                ),

                "macro_f1": float(
                    f1_score(
                        actual_labels,
                        classification_predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),

            })

    # =====================================================
    # SAVE RISK COMPARISON
    # =====================================================

    risk_df = pd.DataFrame(
        model_results
    )

    risk_df.to_csv(
        OUTPUT_DIR / "risk_alert_comparison.csv",
        index=False,
    )

    # =====================================================
    # SAVE CLASSIFICATION COMPARISON
    # =====================================================

    classification_df = pd.DataFrame(
        classification_results
    )

    classification_df.to_csv(
        OUTPUT_DIR / "classification_comparison.csv",
        index=False,
    )

    # =====================================================
    # SAVE DETAILED PREDICTIONS
    # =====================================================

    detailed_df = pd.concat(
        detailed_predictions,
        ignore_index=True,
    )

    detailed_df.to_csv(
        OUTPUT_DIR / "all_model_window_predictions.csv",
        index=False,
    )

    # =====================================================
    # ALERT AGREEMENT
    # =====================================================

    agreement = pd.DataFrame({

        "machineID": reference["machineID"],

        "history_start_step": reference["history_start_step"],

        "target_end_step": reference["target_end_step"],

        "actual_high": reference["actual_high"],

    })

    for model_name, alerts in common_alerts.items():

        agreement[
            model_name.replace(" ", "_").replace("-", "_")
        ] = alerts

    agreement.to_csv(
        OUTPUT_DIR / "all_model_alert_agreement.csv",
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = {

        "test_windows": len(reference),

        "actual_high_windows": int(
            reference["actual_high"].sum()
        ),

        "actual_high_prevalence": float(
            reference["actual_high"].mean()
        ),

        "risk_thresholds": THRESHOLDS,

        "risk_metrics": model_results,

        "classification_metrics": classification_results,

        "important_notes": [

            "All risk comparisons use matched test windows.",

            "All thresholds are sourced from prior validation experiments.",

            "Extra Trees is evaluated using its temporal risk prediction file.",

            "Extra Trees multiclass metrics are not included because "
            "the selected temporal risk file does not contain its "
            "three-class predictions.",

            "Platt calibration does not change the original LSTM "
            "multiclass predictions.",

            "Temperature calibration uses calibrated HIGH probabilities.",

            "Risk alert metrics and multiclass classification metrics "
            "are separate evaluations.",

            "These are historical dataset experiments, not live "
            "cloud provider monitoring results.",

        ],

    }

    with open(
        OUTPUT_DIR / "all_models_comparison.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # =====================================================
    # CONSOLE OUTPUT
    # =====================================================

    print("\n" + "=" * 75)
    print("RISK ALERT COMPARISON")
    print("=" * 75)

    print(
        risk_df[
            [
                "model",
                "threshold",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "specificity",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("MULTICLASS CLASSIFICATION COMPARISON")
    print("=" * 75)

    print(
        classification_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("COMPARISON COMPLETED")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
