
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "ml" / "data" / "optimization_training_dataset.csv"

RAW_PATH = ROOT / "datasets" / "processed" / "server_usage_clean.csv"

MODEL_PATH = (
    ROOT / "ml" / "models" / "memory_high_demand_classifier"
    / "memory_high_demand_classifier.joblib"
)

PREDICTION_PATH = (
    ROOT / "ml" / "models" / "memory_high_demand_classifier"
    / "test_predictions.csv"
)

OUTPUT_DIR = ROOT / "ml" / "evaluation"


def main():

    print("\n" + "=" * 70)
    print("CLOUDMORPH - MISSED MEMORY EVENT INVESTIGATION")
    print("=" * 70)

    features = pd.read_csv(FEATURE_PATH)
    raw = pd.read_csv(RAW_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    feature_columns = package["feature_columns"]

    features["machineID"] = features["machineID"].astype(str)
    raw["machineID"] = raw["machineID"].astype(str)

    # Reconstruct the exact original machine-aware split.

    X = features[feature_columns].apply(
        pd.to_numeric, errors="raise"
    )

    groups = features["machineID"]

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_validation_idx, test_idx = next(
        outer.split(X, groups=groups)
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=43,
    )

    train_relative, validation_relative = next(
        inner.split(
            X.iloc[train_validation_idx],
            groups=groups.iloc[train_validation_idx],
        )
    )

    # Confirm prediction order matches the saved test split.

    X_test = X.iloc[test_idx].reset_index(drop=True)

    test_features = features.iloc[test_idx].reset_index(
        drop=True
    )

    if len(test_features) != len(predictions):
        raise ValueError("Test prediction alignment mismatch.")

    reproduced_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    if not np.allclose(
        reproduced_probabilities,
        predictions["predicted_probability"].to_numpy(),
        atol=1e-7,
    ):
        raise ValueError(
            "Reconstructed predictions do not match saved predictions."
        )

    print("\nSUCCESS: Exact test rows and prediction order verified.")

    combined = test_features.copy()

    combined["actual_high_memory"] = (
        predictions["actual_high_memory"].to_numpy()
    )

    combined["predicted_high_memory"] = (
        predictions["predicted_high_memory"].to_numpy()
    )

    combined["predicted_probability"] = (
        predictions["predicted_probability"].to_numpy()
    )

    combined["actual_peak_memory"] = (
        predictions["actual_peak_memory"].to_numpy()
    )

    # --------------------------------------------------
    # 1. MACHINE-WISE HIGH MEMORY CASES
    # --------------------------------------------------

    print("\n1. MACHINE-WISE TEST RESULTS")

    machine_summary = (
        combined.groupby("machineID")
        .agg(
            total_windows=("machineID", "size"),
            actual_high_windows=("actual_high_memory", "sum"),
            detected_windows=("predicted_high_memory", "sum"),
            mean_probability=("predicted_probability", "mean"),
        )
    )

    machine_summary["missed_windows"] = (
        combined.assign(
            missed=(
                (combined["actual_high_memory"] == 1)
                & (combined["predicted_high_memory"] == 0)
            )
        )
        .groupby("machineID")["missed"]
        .sum()
    )

    machine_summary = machine_summary.sort_values(
        "missed_windows", ascending=False
    )

    print(
        machine_summary[
            machine_summary["actual_high_windows"] > 0
        ].to_string()
    )

    # --------------------------------------------------
    # 2. OVERLAPPING WINDOW INVESTIGATION
    # --------------------------------------------------

    print("\n2. FALSE NEGATIVE WINDOW OVERLAP")

    false_negatives = combined[
        (combined["actual_high_memory"] == 1)
        & (combined["predicted_high_memory"] == 0)
    ].copy()

    window_columns = [
        "machineID",
        "target_start_step",
        "target_end_step",
        "actual_peak_memory",
        "predicted_probability",
    ]

    print(
        false_negatives[window_columns]
        .sort_values(["machineID", "target_start_step"])
        .to_string(index=False)
    )

    print("\nUnique machine + target-window combinations:")
    print(
        false_negatives[
            ["machineID", "target_start_step", "target_end_step"]
        ].drop_duplicates().shape[0]
    )

    # --------------------------------------------------
    # 3. FEATURE IMPORTANCE
    # --------------------------------------------------

    print("\n3. TOP 15 MODEL FEATURE IMPORTANCES")

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance", ascending=False
    )

    print(importance.head(15).to_string(index=False))

    # --------------------------------------------------
    # 4. MISSED CASES VS DETECTED HIGH CASES
    # --------------------------------------------------

    print("\n4. IMPORTANT FEATURE COMPARISON")

    top_features = importance.head(15)["feature"].tolist()

    false_positives = combined[
        (combined["actual_high_memory"] == 0)
        & (combined["predicted_high_memory"] == 1)
    ]

    true_positives = combined[
        (combined["actual_high_memory"] == 1)
        & (combined["predicted_high_memory"] == 1)
    ]

    groups_to_compare = {
        "False negatives": false_negatives,
        "True positives": true_positives,
        "False positives": false_positives,
    }

    for label, group in groups_to_compare.items():

        print(f"\n{label}: {len(group)} samples")

        if group.empty:
            continue

        print(
            group[top_features]
            .median()
            .round(3)
            .to_string()
        )

    # --------------------------------------------------
    # 5. SAVE OUTPUTS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    machine_summary.to_csv(
        OUTPUT_DIR / "memory_missed_event_machine_summary.csv"
    )

    false_negatives.to_csv(
        OUTPUT_DIR / "memory_missed_event_details.csv",
        index=False,
    )

    importance.to_csv(
        OUTPUT_DIR / "memory_classifier_feature_importance.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
