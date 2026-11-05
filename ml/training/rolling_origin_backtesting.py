
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "server_usage_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"

# Alibaba data contains 5-minute observations.
HORIZONS = {
    "15min": 3,
    "30min": 6,
}

# Evaluate 72 consecutive forecast origins.
# Every origin has enough future observations for 30-minute forecasts.
FIRST_ORIGIN = 60
LAST_ORIGIN = 131

# Four chronological evaluation periods.
FOLDS = {
    "Period_1": (60, 77),
    "Period_2": (78, 95),
    "Period_3": (96, 113),
    "Period_4": (114, 131),
}

METHODS = [
    "Persistence",
    "Weighted Moving Average",
]

TARGETS = {
    "CPU": "util_CPU",
    "Memory": "util_memory",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "machineID",
        "step",
        "util_CPU",
        "util_memory",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    df["step"] = pd.to_numeric(
        df["step"],
        errors="coerce",
    )

    for column in TARGETS.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "machineID",
            "step",
            "util_CPU",
            "util_memory",
        ]
    )

    df = df.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    if df.duplicated(
        subset=["machineID", "step"]
    ).any():
        raise ValueError(
            "Duplicate machineID and step combinations found."
        )

    print(f"Dataset records: {len(df):,}")
    print(f"Machines: {df['machineID'].nunique()}")
    print(
        f"Steps per machine (first machine): "
        f"{df.groupby('machineID').size().iloc[0]}"
    )

    return df


# ============================================================
# GENERATE ROLLING-ORIGIN PREDICTIONS
# ============================================================

def generate_predictions(df):

    print("\nPreparing historical and future values...")

    grouped = df.groupby(
        "machineID",
        sort=False,
    )

    work = df[
        [
            "machineID",
            "step",
            "util_CPU",
            "util_memory",
        ]
    ].copy()

    # Current observation at forecast origin.
    work["cpu_current"] = work["util_CPU"]
    work["memory_current"] = work["util_memory"]

    # Two previous observations.
    work["cpu_previous_1"] = grouped["util_CPU"].shift(1)
    work["cpu_previous_2"] = grouped["util_CPU"].shift(2)

    work["memory_previous_1"] = grouped["util_memory"].shift(1)
    work["memory_previous_2"] = grouped["util_memory"].shift(2)

    # Future actual values for both forecast horizons.
    for horizon_name, steps_ahead in HORIZONS.items():

        work[f"cpu_actual_{horizon_name}"] = (
            grouped["util_CPU"].shift(-steps_ahead)
        )

        work[f"memory_actual_{horizon_name}"] = (
            grouped["util_memory"].shift(-steps_ahead)
        )

    # Keep only the chosen forecast origins.
    work = work[
        work["step"].between(
            FIRST_ORIGIN,
            LAST_ORIGIN,
        )
    ].copy()

    # Require complete historical observations.
    historical_columns = [
        "cpu_current",
        "cpu_previous_1",
        "cpu_previous_2",
        "memory_current",
        "memory_previous_1",
        "memory_previous_2",
    ]

    work = work.dropna(
        subset=historical_columns
    )

    # Assign chronological folds.
    def assign_fold(step):

        for fold_name, (start, end) in FOLDS.items():

            if start <= step <= end:
                return fold_name

        return None

    work["fold"] = work["step"].apply(assign_fold)

    work = work.dropna(
        subset=["fold"]
    )

    prediction_frames = []

    print(
        f"Forecast origins available: "
        f"{work['step'].nunique()}"
    )

    print(
        f"Machines available: "
        f"{work['machineID'].nunique()}"
    )

    for horizon_name, steps_ahead in HORIZONS.items():

        for target_name, target_column in TARGETS.items():

            if target_name == "CPU":

                current = work["cpu_current"]
                previous_1 = work["cpu_previous_1"]
                previous_2 = work["cpu_previous_2"]

            else:

                current = work["memory_current"]
                previous_1 = work["memory_previous_1"]
                previous_2 = work["memory_previous_2"]

            actual = work[
                f"{target_name.lower()}_actual_{horizon_name}"
            ]

            valid = actual.notna()

            base = pd.DataFrame({

                "machineID": work.loc[valid, "machineID"],

                "forecast_origin_step": work.loc[valid, "step"],

                "fold": work.loc[valid, "fold"],

                "horizon": horizon_name,

                "horizon_steps": steps_ahead,

                "target": target_name,

                "actual": actual.loc[valid],

                "persistence_prediction": current.loc[valid],

                # Weights: oldest = 1, previous = 2, latest = 3.
                "weighted_prediction": (
                    previous_2.loc[valid] * 1
                    + previous_1.loc[valid] * 2
                    + current.loc[valid] * 3
                ) / 6.0,

            })

            persistence = base[
                [
                    "machineID",
                    "forecast_origin_step",
                    "fold",
                    "horizon",
                    "horizon_steps",
                    "target",
                    "actual",
                ]
            ].copy()

            persistence["method"] = "Persistence"

            persistence["prediction"] = (
                base["persistence_prediction"].values
            )

            weighted = base[
                [
                    "machineID",
                    "forecast_origin_step",
                    "fold",
                    "horizon",
                    "horizon_steps",
                    "target",
                    "actual",
                ]
            ].copy()

            weighted["method"] = (
                "Weighted Moving Average"
            )

            weighted["prediction"] = (
                base["weighted_prediction"].values
            )

            prediction_frames.extend([
                persistence,
                weighted,
            ])

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions["absolute_error"] = (
        predictions["actual"]
        - predictions["prediction"]
    ).abs()

    predictions["squared_error"] = (
        predictions["actual"]
        - predictions["prediction"]
    ) ** 2

    return predictions


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(predictions, group_columns):

    results = (
        predictions.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            samples=("absolute_error", "size"),

            MAE=("absolute_error", "mean"),

            MSE=("squared_error", "mean"),
        )
        .reset_index()
    )

    results["RMSE"] = np.sqrt(
        results["MSE"]
    )

    return results.drop(
        columns=["MSE"]
    )


# ============================================================
# COMPARE METHODS
# ============================================================

def create_comparison(metrics):

    comparison = metrics.pivot(
        index=[
            "fold",
            "horizon",
            "target",
        ],
        columns="method",
        values=["MAE", "RMSE"],
    )

    comparison.columns = [
        f"{metric}_{method.replace(' ', '_')}"
        for metric, method in comparison.columns
    ]

    comparison = comparison.reset_index()

    comparison["MAE_difference"] = (
        comparison[
            "MAE_Weighted_Moving_Average"
        ]
        - comparison["MAE_Persistence"]
    )

    comparison["RMSE_difference"] = (
        comparison[
            "RMSE_Weighted_Moving_Average"
        ]
        - comparison["RMSE_Persistence"]
    )

    comparison["MAE_winner"] = np.where(
        comparison["MAE_difference"] < 0,
        "Weighted Moving Average",
        np.where(
            comparison["MAE_difference"] > 0,
            "Persistence",
            "Tie",
        ),
    )

    comparison["RMSE_winner"] = np.where(
        comparison["RMSE_difference"] < 0,
        "Weighted Moving Average",
        np.where(
            comparison["RMSE_difference"] > 0,
            "Persistence",
            "Tie",
        ),
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CLOUDMORPH ROLLING-ORIGIN BACKTESTING")
    print("=" * 65)

    df = load_data()

    predictions = generate_predictions(df)

    print(
        f"\nTotal forecast predictions: "
        f"{len(predictions):,}"
    )

    # Metrics for each chronological period.
    fold_metrics = calculate_metrics(
        predictions,
        [
            "fold",
            "horizon",
            "target",
            "method",
        ],
    )

    fold_comparison = create_comparison(
        fold_metrics
    )

    # Overall metrics across all forecast origins.
    overall_metrics = calculate_metrics(
        predictions,
        [
            "horizon",
            "target",
            "method",
        ],
    )

    overall_comparison = create_comparison(
        predictions.assign(
            fold="Overall"
        )
        .pipe(
            lambda x: calculate_metrics(
                x,
                [
                    "fold",
                    "horizon",
                    "target",
                    "method",
                ],
            )
        )
    )

    # Count which method wins each period.
    win_summary = (
        fold_comparison.groupby(
            ["horizon", "target", "MAE_winner"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Save outputs.
    predictions_path = (
        OUTPUT_DIR
        / "rolling_origin_backtest_predictions.csv"
    )

    fold_metrics_path = (
        OUTPUT_DIR
        / "rolling_origin_fold_metrics.csv"
    )

    fold_comparison_path = (
        OUTPUT_DIR
        / "rolling_origin_fold_comparison.csv"
    )

    overall_metrics_path = (
        OUTPUT_DIR
        / "rolling_origin_overall_metrics.csv"
    )

    overall_comparison_path = (
        OUTPUT_DIR
        / "rolling_origin_overall_comparison.csv"
    )

    win_summary_path = (
        OUTPUT_DIR
        / "rolling_origin_win_summary.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    fold_comparison.to_csv(
        fold_comparison_path,
        index=False,
    )

    overall_metrics.to_csv(
        overall_metrics_path,
        index=False,
    )

    overall_comparison.to_csv(
        overall_comparison_path,
        index=False,
    )

    win_summary.to_csv(
        win_summary_path,
        index=False,
    )

    # JSON report.
    report = {

        "experiment": "Rolling-Origin Backtesting",

        "dataset": "Alibaba Cluster Trace 2017",

        "observation_interval_minutes": 5,

        "forecast_origins": {
            "first": FIRST_ORIGIN,
            "last": LAST_ORIGIN,
            "total": LAST_ORIGIN - FIRST_ORIGIN + 1,
        },

        "folds": {
            name: {
                "start_origin": bounds[0],
                "end_origin": bounds[1],
            }
            for name, bounds in FOLDS.items()
        },

        "horizons": HORIZONS,

        "methods": METHODS,

        "overall_metrics": overall_metrics.to_dict(
            orient="records"
        ),

        "fold_comparison": fold_comparison.to_dict(
            orient="records"
        ),

        "mae_win_summary": win_summary.to_dict(
            orient="records"
        ),

        "note": (
            "Forecast origins overlap in time. Results across "
            "origins are not statistically independent. "
            "This is a temporal robustness comparison, not "
            "a new untouched confirmatory test."
        ),

    }

    report_path = (
        OUTPUT_DIR
        / "rolling_origin_backtest_report.json"
    )

    with open(
        report_path,
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

    print("\n" + "=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)

    print(
        overall_metrics[
            [
                "horizon",
                "target",
                "method",
                "samples",
                "MAE",
                "RMSE",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 65)
    print("CHRONOLOGICAL PERIOD COMPARISON")
    print("=" * 65)

    print(
        fold_comparison[
            [
                "fold",
                "horizon",
                "target",
                "MAE_Persistence",
                "MAE_Weighted_Moving_Average",
                "MAE_difference",
                "MAE_winner",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 65)
    print("MAE WIN COUNTS")
    print("=" * 65)

    print(
        win_summary.to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(predictions_path)
    print(fold_metrics_path)
    print(fold_comparison_path)
    print(overall_metrics_path)
    print(overall_comparison_path)
    print(win_summary_path)
    print(report_path)

    print("\nBacktesting completed successfully.")


if __name__ == "__main__":
    main()
    