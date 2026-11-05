
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "machine_adaptive_forecast_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"

METHODS = [
    "Persistence",
    "Moving Average",
    "Linear Trend",
]

GROUP_COLUMNS = ["horizon", "target"]

MACHINE_GROUP_COLUMNS = [
    "machineID",
    "horizon",
    "target",
]


# ============================================================
# LOAD AND VALIDATE PREDICTIONS
# ============================================================

def load_predictions():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "machineID",
        "split",
        "horizon",
        "target",
        "machine_group",
        "method",
        "actual",
        "prediction",
        "absolute_error",
        "squared_error",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    df = df[df["method"].isin(METHODS)].copy()

    expected_splits = {"validation", "test"}

    if set(df["split"].unique()) != expected_splits:
        raise ValueError(
            f"Expected validation and test splits, got {df['split'].unique()}"
        )

    if df.empty:
        raise ValueError("No prediction rows found.")

    # Recalculate errors to ensure consistency.
    df["absolute_error"] = (
        df["actual"] - df["prediction"]
    ).abs()

    df["squared_error"] = (
        df["actual"] - df["prediction"]
    ) ** 2

    return df


# ============================================================
# SELECT GLOBAL METHODS USING VALIDATION ONLY
# ============================================================

def select_global_methods(validation):

    scores = (
        validation
        .groupby(GROUP_COLUMNS + ["method"])
        .agg(
            validation_MAE=("absolute_error", "mean"),
            validation_RMSE=(
                "squared_error",
                lambda values: float(np.sqrt(values.mean()))
            ),
        )
        .reset_index()
    )

    scores = scores.sort_values(
        GROUP_COLUMNS + ["validation_MAE", "method"]
    )

    selected = (
        scores
        .groupby(GROUP_COLUMNS, as_index=False)
        .first()
        .rename(columns={
            "method": "selected_method",
        })
    )

    return scores, selected


# ============================================================
# SELECT BEST METHOD PER MACHINE USING VALIDATION ONLY
# ============================================================

def select_machine_methods(validation):

    scores = (
        validation
        .groupby(MACHINE_GROUP_COLUMNS + ["method"])
        .agg(
            validation_MAE=("absolute_error", "mean"),
            validation_RMSE=(
                "squared_error",
                lambda values: float(np.sqrt(values.mean()))
            ),
            validation_samples=("absolute_error", "size"),
        )
        .reset_index()
    )

    scores = scores.sort_values(
        MACHINE_GROUP_COLUMNS + ["validation_MAE", "method"]
    )

    selected = (
        scores
        .groupby(MACHINE_GROUP_COLUMNS, as_index=False)
        .first()
        .rename(columns={
            "method": "selected_method",
        })
    )

    return scores, selected


# ============================================================
# APPLY SELECTIONS TO TEST DATA
# ============================================================

def evaluate_test_strategies(
    test,
    global_selection,
    machine_selection
):

    # Strategy 1: Global validation-selected method.

    global_test = test.merge(
        global_selection[
            GROUP_COLUMNS + ["selected_method"]
        ],
        on=GROUP_COLUMNS,
        how="inner",
    )

    global_test = global_test[
        global_test["method"] == global_test["selected_method"]
    ].copy()

    global_test["strategy"] = "Global validation-selected"

    # Strategy 2: Per-machine validation-selected method.

    machine_test = test.merge(
        machine_selection[
            MACHINE_GROUP_COLUMNS + ["selected_method"]
        ],
        on=MACHINE_GROUP_COLUMNS,
        how="inner",
    )

    machine_test = machine_test[
        machine_test["method"] == machine_test["selected_method"]
    ].copy()

    machine_test["strategy"] = "Per-machine validation-selected"

    # Strategy 3: Persistence for every machine.

    persistence_test = test[
        test["method"] == "Persistence"
    ].copy()

    persistence_test["strategy"] = "Global Persistence"

    combined = pd.concat(
        [
            persistence_test,
            global_test,
            machine_test,
        ],
        ignore_index=True,
    )

    return combined


# ============================================================
# SUMMARIZE TEST PERFORMANCE
# ============================================================

def summarize_test(combined):

    summary = (
        combined
        .groupby(
            ["horizon", "target", "strategy"],
            dropna=False,
        )
        .agg(
            machines=("machineID", "nunique"),
            samples=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            MSE=("squared_error", "mean"),
        )
        .reset_index()
    )

    summary["RMSE"] = np.sqrt(summary["MSE"])

    summary = summary.drop(
        columns=["MSE"]
    )

    return summary.sort_values(
        ["horizon", "target", "MAE"]
    )


# ============================================================
# METHOD SELECTION FREQUENCY
# ============================================================

def summarize_selection_frequency(machine_selection):

    frequency = (
        machine_selection
        .groupby(
            ["horizon", "target", "selected_method"]
        )
        .agg(
            machines=("machineID", "nunique")
        )
        .reset_index()
    )

    frequency["percentage"] = (
        frequency["machines"]
        / frequency.groupby(
            ["horizon", "target"]
        )["machines"].transform("sum")
        * 100
    )

    return frequency.sort_values(
        ["horizon", "target", "machines"],
        ascending=[True, True, False],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading existing machine-level predictions...")

    df = load_predictions()

    print(f"Prediction rows: {len(df):,}")
    print(f"Unique machines: {df['machineID'].nunique()}")

    validation = df[
        df["split"] == "validation"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    print("\nSelecting global methods using validation data only...")

    global_scores, global_selection = (
        select_global_methods(validation)
    )

    print("Selecting per-machine methods using validation data only...")

    machine_scores, machine_selection = (
        select_machine_methods(validation)
    )

    print("\nEvaluating selected strategies on unseen test data...")

    combined = evaluate_test_strategies(
        test,
        global_selection,
        machine_selection,
    )

    summary = summarize_test(combined)

    frequency = summarize_selection_frequency(
        machine_selection
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    global_scores_path = (
        OUTPUT_DIR / "adaptive_global_validation_scores.csv"
    )

    machine_scores_path = (
        OUTPUT_DIR / "adaptive_machine_validation_scores.csv"
    )

    machine_selection_path = (
        OUTPUT_DIR / "adaptive_machine_method_selection.csv"
    )

    frequency_path = (
        OUTPUT_DIR / "adaptive_method_selection_frequency.csv"
    )

    test_predictions_path = (
        OUTPUT_DIR / "adaptive_test_strategy_predictions.csv"
    )

    summary_path = (
        OUTPUT_DIR / "adaptive_forecasting_test_comparison.csv"
    )

    json_path = (
        OUTPUT_DIR / "adaptive_forecasting_test_comparison.json"
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    global_scores.to_csv(
        global_scores_path,
        index=False
    )

    machine_scores.to_csv(
        machine_scores_path,
        index=False
    )

    machine_selection.to_csv(
        machine_selection_path,
        index=False
    )

    frequency.to_csv(
        frequency_path,
        index=False
    )

    combined.to_csv(
        test_predictions_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    report = {
        "experiment": "Validation-selected adaptive forecasting",

        "selection_data": "Validation split only",

        "final_evaluation_data": "Unseen test split only",

        "selection_unit": [
            "machineID",
            "horizon",
            "target",
        ],

        "methods": METHODS,

        "strategies": [
            "Global Persistence",
            "Global validation-selected",
            "Per-machine validation-selected",
        ],

        "summary": summary.to_dict(
            orient="records"
        ),

        "selection_frequency": frequency.to_dict(
            orient="records"
        ),

        "outputs": {
            "global_validation_scores": str(
                global_scores_path.relative_to(PROJECT_ROOT)
            ),

            "machine_validation_scores": str(
                machine_scores_path.relative_to(PROJECT_ROOT)
            ),

            "machine_method_selection": str(
                machine_selection_path.relative_to(PROJECT_ROOT)
            ),

            "method_selection_frequency": str(
                frequency_path.relative_to(PROJECT_ROOT)
            ),

            "test_predictions": str(
                test_predictions_path.relative_to(PROJECT_ROOT)
            ),

            "summary_csv": str(
                summary_path.relative_to(PROJECT_ROOT)
            ),
        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\nGLOBAL METHOD SELECTION FROM VALIDATION:")

    print(
        global_selection[
            [
                "horizon",
                "target",
                "selected_method",
                "validation_MAE"
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nPER-MACHINE METHOD SELECTION FREQUENCY:")

    print(
        frequency.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nUNSEEN TEST PERFORMANCE:")

    print(
        summary[
            [
                "horizon",
                "target",
                "strategy",
                "machines",
                "samples",
                "MAE",
                "RMSE",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved files:")

    print(global_scores_path)
    print(machine_scores_path)
    print(machine_selection_path)
    print(frequency_path)
    print(test_predictions_path)
    print(summary_path)
    print(json_path)


if __name__ == "__main__":
    main()
