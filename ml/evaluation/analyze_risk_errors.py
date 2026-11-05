
"""
CloudMorph - Independent Risk Error Analysis

Investigates:
1. Missed HIGH_DEMAND cases (false negatives)
2. False HIGH_RISK alerts (false positives)
3. Correctly detected HIGH_DEMAND cases
4. Correctly ignored non-high-demand cases

Diagnostic only:
- Does not modify the production model.
- Does not change the selected risk threshold.
- Does not use test data for model or threshold tuning.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "ml" / "data" / "optimization_training_dataset.csv"

MODEL_PATH = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "optimization_classifier.joblib"
)

THRESHOLD_REPORT = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "threshold_evaluation" / "threshold_evaluation_report.json"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "error_analysis"
)

RANDOM_STATE = 42
N_ESTIMATORS = 300

TRAIN_FRACTION = 0.40
TEST_FRACTION = 0.70

HIGH_LABEL = "HIGH_DEMAND"

METADATA_COLUMNS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_start_step",
    "target_end_step",
    "target",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphErrorAnalysis")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    missing = [
        col for col in METADATA_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError("Missing or non-numeric feature values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Infinite feature values found.")

    y = df["target"].astype(str)

    logger.info("Dataset samples: %s", len(df))
    logger.info("Features: %s", len(feature_columns))

    return df, X, y, feature_columns


# ---------------------------------------------------------
# LOAD FROZEN THRESHOLD
# ---------------------------------------------------------

def load_threshold():

    with open(
        THRESHOLD_REPORT,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    threshold = float(
        report["threshold_selection"]["selected_threshold"]
    )

    logger.info("Using frozen threshold: %.4f", threshold)

    return threshold


# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

def create_splits(df, X, y):

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

    logger.info(
        "Train: %s | Validation: %s | Test: %s",
        train_mask.sum(),
        validation_mask.sum(),
        test_mask.sum(),
    )

    return (
        df.loc[train_mask].copy(),
        df.loc[validation_mask].copy(),
        df.loc[test_mask].copy(),
        X.loc[train_mask],
        y.loc[train_mask],
        X.loc[test_mask],
        y.loc[test_mask],
    )


# ---------------------------------------------------------
# TRAIN EXPERIMENTAL MODEL
# ---------------------------------------------------------

def train_model(X_train, y_train):

    logger.info("Training chronological Extra Trees model...")

    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------
# BUILD ERROR GROUPS
# ---------------------------------------------------------

def build_error_groups(
    test_df,
    y_test,
    predictions,
    high_probabilities,
    threshold,
):

    result = test_df[
        METADATA_COLUMNS
    ].copy()

    result["actual_workload"] = y_test.to_numpy()

    result["predicted_workload"] = predictions

    result["high_demand_probability"] = high_probabilities

    result["risk_threshold"] = threshold

    result["risk_alert"] = (
        high_probabilities >= threshold
    )

    actual_high = (
        result["actual_workload"] == HIGH_LABEL
    )

    alert = result["risk_alert"]

    conditions = [
        actual_high & alert,
        actual_high & ~alert,
        ~actual_high & alert,
        ~actual_high & ~alert,
    ]

    categories = [
        "DETECTED_HIGH",
        "MISSED_HIGH",
        "FALSE_ALERT",
        "CORRECTLY_IGNORED",
    ]

    result["error_group"] = np.select(
        conditions,
        categories,
        default="UNKNOWN",
    )

    return result


# ---------------------------------------------------------
# GROUP COUNTS
# ---------------------------------------------------------

def create_group_summary(result):

    summary = (
        result["error_group"]
        .value_counts()
        .reindex(
            [
                "DETECTED_HIGH",
                "MISSED_HIGH",
                "FALSE_ALERT",
                "CORRECTLY_IGNORED",
            ],
            fill_value=0,
        )
        .rename_axis("error_group")
        .reset_index(name="cases")
    )

    summary["percentage_of_test_set"] = (
        summary["cases"] / len(result) * 100
    )

    return summary


# ---------------------------------------------------------
# FEATURE COMPARISON
# ---------------------------------------------------------

def create_feature_comparison(
    X_test,
    result,
    feature_columns,
):

    logger.info("Comparing features across error groups...")

    combined = X_test.copy()

    combined["error_group"] = (
        result["error_group"].to_numpy()
    )

    groups = [
        "DETECTED_HIGH",
        "MISSED_HIGH",
        "FALSE_ALERT",
        "CORRECTLY_IGNORED",
    ]

    records = []

    for feature in feature_columns:

        record = {
            "feature": feature,
        }

        for group in groups:

            values = combined.loc[
                combined["error_group"] == group,
                feature,
            ]

            record[f"{group}_mean"] = (
                float(values.mean())
                if len(values) else None
            )

            record[f"{group}_median"] = (
                float(values.median())
                if len(values) else None
            )

            record[f"{group}_count"] = int(len(values))

        detected_mean = record["DETECTED_HIGH_mean"]
        missed_mean = record["MISSED_HIGH_mean"]

        if detected_mean is not None and missed_mean is not None:

            record["missed_minus_detected_mean"] = (
                missed_mean - detected_mean
            )

        else:
            record["missed_minus_detected_mean"] = None

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

def create_feature_importance(model, feature_columns):

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ---------------------------------------------------------

def create_probability_summary(result):

    summary = (
        result.groupby("error_group")[
            "high_demand_probability"
        ]
        .agg(
            count="count",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
            standard_deviation="std",
        )
        .reset_index()
    )

    return summary


def create_probability_bands(result):

    bins = [
        -0.000001,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.000001,
    ]

    labels = [
        "0.00-0.10",
        "0.10-0.20",
        "0.20-0.30",
        "0.30-0.40",
        "0.40-0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]

    result = result.copy()

    result["probability_band"] = pd.cut(
        result["high_demand_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    bands = (
        result.groupby(
            ["error_group", "probability_band"],
            observed=False,
        )
        .size()
        .reset_index(name="cases")
    )

    return bands


# ---------------------------------------------------------
# MACHINE AND TIME ANALYSIS
# ---------------------------------------------------------

def create_machine_analysis(result):

    machine_summary = (
        result.groupby("machineID")
        .agg(
            total_cases=("error_group", "size"),
            missed_high=(
                "error_group",
                lambda x: (x == "MISSED_HIGH").sum(),
            ),
            false_alerts=(
                "error_group",
                lambda x: (x == "FALSE_ALERT").sum(),
            ),
            detected_high=(
                "error_group",
                lambda x: (x == "DETECTED_HIGH").sum(),
            ),
        )
        .reset_index()
    )

    machine_summary = machine_summary.sort_values(
        ["missed_high", "false_alerts"],
        ascending=False,
    )

    return machine_summary


def create_time_analysis(result):

    time_summary = (
        result.groupby("history_start_step")
        .agg(
            total_cases=("error_group", "size"),
            missed_high=(
                "error_group",
                lambda x: (x == "MISSED_HIGH").sum(),
            ),
            false_alerts=(
                "error_group",
                lambda x: (x == "FALSE_ALERT").sum(),
            ),
            detected_high=(
                "error_group",
                lambda x: (x == "DETECTED_HIGH").sum(),
            ),
        )
        .reset_index()
    )

    return time_summary


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

def save_outputs(
    result,
    group_summary,
    feature_comparison,
    feature_importance,
    probability_summary,
    probability_bands,
    machine_analysis,
    time_analysis,
    threshold,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "risk_error_cases.csv": result,
        "error_group_summary.csv": group_summary,
        "error_feature_comparison.csv": feature_comparison,
        "error_feature_importance.csv": feature_importance,
        "error_probability_summary.csv": probability_summary,
        "error_probability_bands.csv": probability_bands,
        "error_machine_analysis.csv": machine_analysis,
        "error_time_analysis.csv": time_analysis,
    }

    for filename, dataframe in outputs.items():

        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

        logger.info("Saved: %s", filename)

    report = {
        "project": "CloudMorph",
        "analysis": "Independent high-demand risk error analysis",
        "risk_threshold": threshold,
        "test_samples": int(len(result)),
        "error_group_counts": {
            row["error_group"]: int(row["cases"])
            for _, row in group_summary.iterrows()
        },
        "important_notes": [
            "This is a diagnostic experiment only.",
            "The production model was not modified.",
            "The risk threshold was not changed.",
            "The same chronological split and training configuration were used.",
            "Feature differences are descriptive, not proof of causation.",
            "Machine-level error counts do not establish machine-specific failure.",
            "No cloud resources were modified.",
        ],
    }

    with open(
        OUTPUT_DIR / "error_analysis_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info("All error-analysis outputs saved.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH RISK ERROR ANALYSIS")
    logger.info("=" * 65)

    df, X, y, feature_columns = load_data()

    # Verify production model schema without changing it.

    if MODEL_PATH.exists():

        package = joblib.load(MODEL_PATH)

        if package.get("feature_columns") != feature_columns:
            raise ValueError(
                "Production model feature schema mismatch."
            )

        logger.info("Production model schema verified.")

    threshold = load_threshold()

    (
        train_df,
        validation_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = create_splits(
        df,
        X,
        y,
    )

    model = train_model(
        X_train,
        y_train,
    )

    logger.info("Generating test predictions...")

    predictions = model.predict(X_test)

    high_index = list(model.classes_).index(
        HIGH_LABEL
    )

    high_probabilities = model.predict_proba(
        X_test
    )[:, high_index]

    result = build_error_groups(
        test_df=test_df,
        y_test=y_test,
        predictions=predictions,
        high_probabilities=high_probabilities,
        threshold=threshold,
    )

    group_summary = create_group_summary(result)

    feature_comparison = create_feature_comparison(
        X_test,
        result,
        feature_columns,
    )

    feature_importance = create_feature_importance(
        model,
        feature_columns,
    )

    probability_summary = create_probability_summary(
        result
    )

    probability_bands = create_probability_bands(
        result
    )

    machine_analysis = create_machine_analysis(
        result
    )

    time_analysis = create_time_analysis(
        result
    )

    logger.info("=" * 65)
    logger.info("ERROR GROUP SUMMARY")
    logger.info("=" * 65)

    logger.info(
        "\n%s",
        group_summary.to_string(index=False),
    )

    logger.info("=" * 65)
    logger.info("TOP 10 MODEL FEATURES")
    logger.info("=" * 65)

    logger.info(
        "\n%s",
        feature_importance.head(10).to_string(index=False),
    )

    logger.info("=" * 65)
    logger.info("PROBABILITY SUMMARY")
    logger.info("=" * 65)

    logger.info(
        "\n%s",
        probability_summary.to_string(index=False),
    )

    save_outputs(
        result=result,
        group_summary=group_summary,
        feature_comparison=feature_comparison,
        feature_importance=feature_importance,
        probability_summary=probability_summary,
        probability_bands=probability_bands,
        machine_analysis=machine_analysis,
        time_analysis=time_analysis,
        threshold=threshold,
    )

    logger.info("=" * 65)
    logger.info("RISK ERROR ANALYSIS COMPLETED")
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
