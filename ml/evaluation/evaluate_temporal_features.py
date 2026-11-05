
"""
CloudMorph - Temporal Feature Engineering Experiment

Compares:
1. Baseline Extra Trees: existing 55 features.
2. Enhanced Extra Trees: existing 55 + 8 temporal features.

Temporal features are calculated exclusively from the 12 historical
observations. Future observations are used only for labels.

Chronological split:
- First 40%: training
- Next 30%: validation
- Final 30%: testing

Risk thresholds are selected on validation data only.
Production model and existing datasets are never modified.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT / "ml" / "data"
    / "optimization_training_dataset.csv"
)

RAW_DATA_PATH = (
    ROOT / "datasets" / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models"
    / "optimization_classifier"
    / "temporal_feature_evaluation"
)

METADATA_COLUMNS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
    "target",
]

CLASS_LABELS = [
    "LOW_DEMAND",
    "NORMAL_DEMAND",
    "HIGH_DEMAND",
]

HISTORY_SIZE = 12
HIGH_THRESHOLD = 85.0

RANDOM_STATE = 42
N_ESTIMATORS = 300

TRAIN_FRACTION = 0.40
TEST_FRACTION = 0.70

MIN_VALIDATION_PRECISION = 0.60

TEMPORAL_FEATURES = [
    "memory_last_delta",
    "memory_recent_delta",
    "memory_positive_steps",
    "memory_max_rise",
    "memory_acceleration",
    "cpu_last_delta",
    "cpu_recent_delta",
    "cpu_positive_steps",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphTemporalFeatures")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    logger.info("Loading existing optimization dataset...")

    df = pd.read_csv(DATASET_PATH)

    missing = [
        col for col in METADATA_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset metadata: {missing}"
        )

    base_features = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    if len(base_features) != 55:
        raise ValueError(
            f"Expected 55 existing features, got {len(base_features)}."
        )

    X = df[base_features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError("Missing values in base features.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Infinite values in base features.")

    y = df["target"].astype(str)

    if set(y.unique()) != set(CLASS_LABELS):
        raise ValueError("Unexpected target classes.")

    logger.info("Samples: %s", len(df))
    logger.info("Base features: %s", len(base_features))

    return df, X, y, base_features


# ---------------------------------------------------------
# TEMPORAL FEATURE ENGINEERING
# ---------------------------------------------------------

def calculate_temporal_features(history):

    cpu = history["util_CPU"].to_numpy(dtype=float)

    memory = history["util_memory"].to_numpy(dtype=float)

    cpu_diff = np.diff(cpu)
    memory_diff = np.diff(memory)

    # Recent 3 observations versus previous 3 observations.
    cpu_recent_delta = (
        np.mean(cpu[-3:]) - np.mean(cpu[-6:-3])
    )

    memory_recent_delta = (
        np.mean(memory[-3:]) - np.mean(memory[-6:-3])
    )

    # Change in the average step-to-step rate.
    memory_acceleration = (
        np.mean(memory_diff[-3:])
        - np.mean(memory_diff[-6:-3])
    )

    return {
        "memory_last_delta": float(memory_diff[-1]),

        "memory_recent_delta": float(memory_recent_delta),

        "memory_positive_steps": int(
            np.sum(memory_diff > 0)
        ),

        "memory_max_rise": float(
            np.max(memory_diff)
        ),

        "memory_acceleration": float(
            memory_acceleration
        ),

        "cpu_last_delta": float(cpu_diff[-1]),

        "cpu_recent_delta": float(cpu_recent_delta),

        "cpu_positive_steps": int(
            np.sum(cpu_diff > 0)
        ),
    }


def build_temporal_features(df):

    logger.info("Loading raw workload observations...")

    raw = pd.read_csv(
        RAW_DATA_PATH,
        usecols=[
            "machineID",
            "step",
            "util_CPU",
            "util_memory",
        ],
    )

    raw = raw.sort_values(
        ["machineID", "step"]
    )

    machine_data = {
        machine_id: group.set_index("step")
        for machine_id, group in raw.groupby("machineID")
    }

    records = []

    logger.info(
        "Calculating temporal features for %s windows...",
        len(df),
    )

    for row in df.itertuples(index=False):

        machine_id = getattr(row, "machineID")
        start = int(getattr(row, "history_start_step"))

        if machine_id not in machine_data:
            raise ValueError(
                f"Missing machine {machine_id}"
            )

        machine = machine_data[machine_id]

        steps = list(
            range(start, start + HISTORY_SIZE)
        )

        if not set(steps).issubset(set(machine.index)):
            raise ValueError(
                f"Incomplete history for machine {machine_id}, "
                f"step {start}"
            )

        history = machine.loc[steps]

        features = calculate_temporal_features(
            history
        )

        records.append(features)

    temporal_df = pd.DataFrame(
        records,
        index=df.index,
    )

    if temporal_df.isna().any().any():
        raise ValueError("Missing temporal feature values.")

    if not np.isfinite(temporal_df.to_numpy()).all():
        raise ValueError("Infinite temporal feature values.")

    logger.info(
        "Generated %s temporal features.",
        len(temporal_df.columns),
    )

    return temporal_df


# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

def create_splits(df, X_base, X_enhanced, y):

    steps = np.sort(
        df["history_start_step"].unique()
    )

    validation_start = steps[
        int(np.ceil(len(steps) * TRAIN_FRACTION))
    ]

    test_start = steps[
        int(np.ceil(len(steps) * TEST_FRACTION))
    ]

    train_mask = (
        (df["history_start_step"] < validation_start)
        &
        (df["target_end_step"] < validation_start)
    )

    validation_mask = (
        (df["history_start_step"] >= validation_start)
        &
        (df["target_end_step"] < test_start)
    )

    test_mask = (
        df["history_start_step"] >= test_start
    )

    masks = {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }

    for name, mask in masks.items():

        if mask.sum() == 0:
            raise ValueError(f"Empty {name} split.")

        missing = (
            set(CLASS_LABELS)
            - set(y.loc[mask].unique())
        )

        if missing:
            raise ValueError(
                f"{name} split missing classes: {missing}"
            )

    purged = len(df) - sum(
        int(mask.sum()) for mask in masks.values()
    )

    logger.info(
        "Training: %s | Validation: %s | Test: %s | Purged: %s",
        train_mask.sum(),
        validation_mask.sum(),
        test_mask.sum(),
        purged,
    )

    return {
        "train": {
            "X_base": X_base.loc[train_mask],
            "X_enhanced": X_enhanced.loc[train_mask],
            "y": y.loc[train_mask],
        },
        "validation": {
            "X_base": X_base.loc[validation_mask],
            "X_enhanced": X_enhanced.loc[validation_mask],
            "y": y.loc[validation_mask],
        },
        "test": {
            "X_base": X_base.loc[test_mask],
            "X_enhanced": X_enhanced.loc[test_mask],
            "y": y.loc[test_mask],
            "metadata": df.loc[test_mask],
        },
        "validation_start": int(validation_start),
        "test_start": int(test_start),
        "purged": int(purged),
    }


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def create_model():

    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------
# THRESHOLD SELECTION
# ---------------------------------------------------------

def select_validation_threshold(
    y_validation,
    high_probabilities,
):

    actual_high = (
        y_validation.to_numpy() == "HIGH_DEMAND"
    )

    candidates = np.unique(
        high_probabilities
    )

    candidates = np.append(
        candidates,
        1.000001,
    )

    best = None

    for threshold in candidates:

        alerts = (
            high_probabilities >= threshold
        )

        precision = precision_score(
            actual_high,
            alerts,
            zero_division=0,
        )

        recall = recall_score(
            actual_high,
            alerts,
            zero_division=0,
        )

        f1 = f1_score(
            actual_high,
            alerts,
            zero_division=0,
        )

        if precision < MIN_VALIDATION_PRECISION:
            continue

        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "alerts": int(alerts.sum()),
        }

        if best is None or candidate["f1"] > best["f1"]:

            best = candidate

        elif (
            candidate["f1"] == best["f1"]
            and candidate["precision"] > best["precision"]
        ):

            best = candidate

    if best is None:
        raise ValueError(
            "No validation threshold satisfies the precision requirement."
        )

    return best


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    name,
    model,
    X_validation,
    y_validation,
    X_test,
    y_test,
):

    logger.info("Evaluating %s...", name)

    validation_probabilities = model.predict_proba(
        X_validation
    )

    high_index = list(model.classes_).index(
        "HIGH_DEMAND"
    )

    validation_high_prob = (
        validation_probabilities[:, high_index]
    )

    threshold_result = select_validation_threshold(
        y_validation,
        validation_high_prob,
    )

    threshold = threshold_result["threshold"]

    logger.info(
        "%s validation threshold: %.6f",
        name,
        threshold,
    )

    # Final test predictions. No test-based threshold tuning.

    predictions = model.predict(X_test)

    test_probabilities = model.predict_proba(
        X_test
    )

    high_probabilities = (
        test_probabilities[:, high_index]
    )

    risk_alerts = (
        high_probabilities >= threshold
    )

    actual_high = (
        y_test.to_numpy() == "HIGH_DEMAND"
    )

    classification_metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),

        "high_demand_precision": float(
            precision_score(
                y_test,
                predictions,
                labels=["HIGH_DEMAND"],
                average="micro",
                zero_division=0,
            )
        ),

        "high_demand_recall": float(
            recall_score(
                y_test,
                predictions,
                labels=["HIGH_DEMAND"],
                average="micro",
                zero_division=0,
            )
        ),

        "high_demand_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=["HIGH_DEMAND"],
                average="micro",
                zero_division=0,
            )
        ),
    }

    risk_metrics = {
        "precision": float(
            precision_score(
                actual_high,
                risk_alerts,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                actual_high,
                risk_alerts,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                actual_high,
                risk_alerts,
                zero_division=0,
            )
        ),

        "true_positives": int(
            np.sum(actual_high & risk_alerts)
        ),

        "false_positives": int(
            np.sum(~actual_high & risk_alerts)
        ),

        "false_negatives": int(
            np.sum(actual_high & ~risk_alerts)
        ),

        "true_negatives": int(
            np.sum(~actual_high & ~risk_alerts)
        ),

        "total_alerts": int(
            risk_alerts.sum()
        ),
    }

    logger.info(
        "%s classification: %s",
        name,
        classification_metrics,
    )

    logger.info(
        "%s risk: %s",
        name,
        risk_metrics,
    )

    return {
        "name": name,
        "threshold_selection": threshold_result,
        "classification": classification_metrics,
        "risk": risk_metrics,
        "predictions": predictions,
        "high_probabilities": high_probabilities,
        "risk_alerts": risk_alerts,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=CLASS_LABELS,
        ).tolist(),
    }


# ---------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------

def save_results(
    baseline,
    enhanced,
    splits,
    base_features,
    enhanced_features,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = [
        baseline,
        enhanced,
    ]

    comparison_rows = []

    for result in results:

        comparison_rows.append({
            "model": result["name"],
            "features": (
                55 if result["name"] == "BASELINE"
                else len(enhanced_features)
            ),
            "risk_threshold": (
                result["threshold_selection"]["threshold"]
            ),
            **{
                f"classification_{key}": value
                for key, value in result["classification"].items()
            },
            **{
                f"risk_{key}": value
                for key, value in result["risk"].items()
            },
        })

        prediction_df = splits["test"]["metadata"][
            [
                "machineID",
                "history_start_step",
                "target_start_step",
                "target_end_step",
                "target",
            ]
        ].copy()

        prediction_df["predicted_workload"] = (
            result["predictions"]
        )

        prediction_df["high_demand_probability"] = (
            result["high_probabilities"]
        )

        prediction_df["risk_threshold"] = (
            result["threshold_selection"]["threshold"]
        )

        prediction_df["risk_alert"] = np.where(
            result["risk_alerts"],
            "HIGH_RISK",
            "NO_HIGH_RISK",
        )

        prediction_df.to_csv(
            OUTPUT_DIR
            / f"{result['name'].lower()}_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            result["confusion_matrix"],
            index=CLASS_LABELS,
            columns=CLASS_LABELS,
        ).to_csv(
            OUTPUT_DIR
            / f"{result['name'].lower()}_confusion_matrix.csv"
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    report = {
        "project": "CloudMorph",
        "experiment": "Temporal feature engineering",
        "baseline_feature_count": len(base_features),
        "enhanced_feature_count": len(enhanced_features),
        "added_temporal_features": TEMPORAL_FEATURES,
        "split": {
            "training_samples": int(
                len(splits["train"]["y"])
            ),
            "validation_samples": int(
                len(splits["validation"]["y"])
            ),
            "test_samples": int(
                len(splits["test"]["y"])
            ),
            "purged_samples": splits["purged"],
            "validation_start_step": splits["validation_start"],
            "test_start_step": splits["test_start"],
        },
        "threshold_selection": {
            "method": (
                "Maximum validation HIGH_DEMAND F1 "
                "subject to precision >= 0.60"
            ),
            "baseline": baseline["threshold_selection"],
            "enhanced": enhanced["threshold_selection"],
        },
        "baseline": {
            "classification": baseline["classification"],
            "risk": baseline["risk"],
            "confusion_matrix": baseline["confusion_matrix"],
        },
        "enhanced": {
            "classification": enhanced["classification"],
            "risk": enhanced["risk"],
            "confusion_matrix": enhanced["confusion_matrix"],
        },
        "important_notes": [
            "Both models use identical chronological splits.",
            "Both models use identical Extra Trees configuration.",
            "Temporal features use historical observations only.",
            "Future observations are used only for labels.",
            "Each model's risk threshold is selected using validation data only.",
            "Final test data is not used for threshold selection.",
            "The production model and dataset were not modified.",
            "The enhanced model is experimental and not deployed.",
        ],
    }

    with open(
        OUTPUT_DIR / "temporal_feature_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info("Saved model comparison.")
    logger.info("Saved experimental report.")
    logger.info("Saved both models' predictions.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH TEMPORAL FEATURE EXPERIMENT")
    logger.info("=" * 65)

    df, X_base, y, base_features = load_data()

    temporal_features = build_temporal_features(
        df
    )

    X_enhanced = pd.concat(
        [
            X_base,
            temporal_features,
        ],
        axis=1,
    )

    if len(X_enhanced.columns) != 63:
        raise ValueError(
            "Expected 63 enhanced features."
        )

    splits = create_splits(
        df,
        X_base,
        X_enhanced,
        y,
    )

    # Baseline model.

    baseline_model = create_model()

    logger.info("Training baseline model...")

    baseline_model.fit(
        splits["train"]["X_base"],
        splits["train"]["y"],
    )

    baseline = evaluate_model(
        "BASELINE",
        baseline_model,
        splits["validation"]["X_base"],
        splits["validation"]["y"],
        splits["test"]["X_base"],
        splits["test"]["y"],
    )

    # Enhanced model.

    enhanced_model = create_model()

    logger.info("Training enhanced temporal model...")

    enhanced_model.fit(
        splits["train"]["X_enhanced"],
        splits["train"]["y"],
    )

    enhanced = evaluate_model(
        "ENHANCED",
        enhanced_model,
        splits["validation"]["X_enhanced"],
        splits["validation"]["y"],
        splits["test"]["X_enhanced"],
        splits["test"]["y"],
    )

    save_results(
        baseline,
        enhanced,
        splits,
        base_features,
        list(X_enhanced.columns),
    )

    logger.info("=" * 65)
    logger.info("TEMPORAL FEATURE EXPERIMENT COMPLETED")
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
