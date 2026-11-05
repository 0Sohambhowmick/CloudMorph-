
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

WINDOW_SIZE = 12

HORIZONS = {
    "15min": 3,
    "30min": 6,
}

TARGETS = {
    "CPU": "cpu",
    "Memory": "memory",
}

SPLITS = {
    "validation": range(97, 116),
    "test": range(119, 138),
}

GAMMA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "machineID",
        "step",
        "util_CPU",
        "util_memory",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    machines = {}

    for machine_id, group in df.groupby(
        "machineID",
        sort=True,
    ):

        group = group.sort_values("step")

        if len(group) != 144:
            continue

        if not np.array_equal(
            group["step"].to_numpy(),
            np.arange(144),
        ):
            continue

        machines[int(machine_id)] = {
            "cpu": group["util_CPU"].to_numpy(dtype=np.float64),
            "memory": group["util_memory"].to_numpy(dtype=np.float64),
        }

    if not machines:
        raise ValueError(
            "No machines with exactly 144 consecutive steps were found."
        )

    return machines


# ============================================================
# FORECASTING METHODS
# ============================================================

def predict_persistence(history):
    return float(history[-1])


def predict_ema(history, alpha=0.5):

    value = float(history[0])

    for observation in history[1:]:

        value = (
            alpha * float(observation)
            + (1.0 - alpha) * value
        )

    return value


def predict_weighted_moving_average(history):

    recent = np.asarray(
        history[-3:],
        dtype=np.float64,
    )

    weights = np.array([1.0, 2.0, 3.0])

    return float(
        np.average(recent, weights=weights)
    )


def predict_rolling_median(history):

    return float(
        np.median(history[-3:])
    )


def predict_volatility_aware(history, horizon_steps, gamma):

    recent = np.asarray(
        history[-6:],
        dtype=np.float64,
    )

    x = np.arange(
        len(recent),
        dtype=np.float64,
    )

    slope, _ = np.polyfit(x, recent, 1)

    changes = np.diff(recent)

    volatility = float(
        np.std(changes)
    )

    trend_strength = abs(float(slope))

    damping = (
        trend_strength
        / (trend_strength + volatility + 1e-8)
    )

    adjustment = (
        gamma
        * float(slope)
        * horizon_steps
        * damping
    )

    prediction = float(history[-1]) + adjustment

    return float(
        np.clip(prediction, 0.0, 100.0)
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def build_predictions(machines):

    rows = []

    for machine_id, series in machines.items():

        for split_name, origins in SPLITS.items():

            for origin in origins:

                start = origin - WINDOW_SIZE + 1
                end = origin + 1

                for horizon_name, horizon_steps in HORIZONS.items():

                    target_step = origin + horizon_steps

                    if target_step >= 144:
                        continue

                    for target_name, series_key in TARGETS.items():

                        values = series[series_key]

                        history = values[start:end]

                        actual = float(
                            values[target_step]
                        )

                        common = {
                            "machineID": machine_id,
                            "split": split_name,
                            "horizon": horizon_name,
                            "target": target_name,
                            "forecast_origin_step": origin,
                            "target_step": target_step,
                            "actual": actual,
                        }

                        fixed_predictions = {

                            "Persistence":
                                predict_persistence(history),

                            "EMA":
                                predict_ema(history, alpha=0.5),

                            "Weighted Moving Average":
                                predict_weighted_moving_average(history),

                            "Rolling Median":
                                predict_rolling_median(history),
                        }

                        for method, prediction in fixed_predictions.items():

                            error = actual - prediction

                            rows.append({
                                **common,
                                "method": method,
                                "gamma": np.nan,
                                "prediction": prediction,
                                "absolute_error": abs(error),
                                "squared_error": error ** 2,
                            })

                        # Evaluate all gamma candidates.
                        for gamma in GAMMA_VALUES:

                            prediction = predict_volatility_aware(
                                history,
                                horizon_steps,
                                gamma,
                            )

                            error = actual - prediction

                            rows.append({
                                **common,
                                "method": "Volatility-aware Persistence",
                                "gamma": gamma,
                                "prediction": prediction,
                                "absolute_error": abs(error),
                                "squared_error": error ** 2,
                            })

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise ValueError(
            "No forecast predictions were generated."
        )

    return predictions


# ============================================================
# SELECT GAMMA USING VALIDATION ONLY
# ============================================================

def select_gamma(validation_predictions):

    volatility_rows = validation_predictions[
        validation_predictions["method"]
        == "Volatility-aware Persistence"
    ].copy()

    scores = (
        volatility_rows
        .groupby(["horizon", "target", "gamma"])
        .agg(
            validation_MAE=("absolute_error", "mean"),
            validation_MSE=("squared_error", "mean"),
        )
        .reset_index()
    )

    scores["validation_RMSE"] = np.sqrt(
        scores["validation_MSE"]
    )

    scores = scores.drop(
        columns=["validation_MSE"]
    )

    scores = scores.sort_values(
        ["horizon", "target", "validation_MAE", "gamma"]
    )

    selected = (
        scores
        .groupby(
            ["horizon", "target"],
            as_index=False,
        )
        .first()
        .rename(
            columns={"gamma": "selected_gamma"}
        )
    )

    return scores, selected


# ============================================================
# BUILD TEST COMPARISON
# ============================================================

def build_test_comparison(predictions, selected_gamma):

    test = predictions[
        predictions["split"] == "test"
    ].copy()

    # Keep fixed-method predictions.
    fixed = test[
        test["method"] != "Volatility-aware Persistence"
    ].copy()

    fixed["selected_gamma"] = np.nan

    # Keep only the gamma selected using validation.
    adaptive = test[
        test["method"] == "Volatility-aware Persistence"
    ].merge(
        selected_gamma[
            ["horizon", "target", "selected_gamma"]
        ],
        on=["horizon", "target"],
        how="inner",
    )

    adaptive = adaptive[
        np.isclose(
            adaptive["gamma"],
            adaptive["selected_gamma"],
        )
    ].copy()

    return pd.concat(
        [fixed, adaptive],
        ignore_index=True,
    )


# ============================================================
# SUMMARIES
# ============================================================

def summarize_validation(predictions):

    validation = predictions[
        predictions["split"] == "validation"
    ].copy()

    validation["method_label"] = np.where(
        validation["method"] == "Volatility-aware Persistence",

        validation["method"]
        + " (gamma="
        + validation["gamma"].astype(str)
        + ")",

        validation["method"],
    )

    summary = (
        validation
        .groupby(
            ["horizon", "target", "method_label"]
        )
        .agg(
            samples=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            MSE=("squared_error", "mean"),
        )
        .reset_index()
    )

    summary["RMSE"] = np.sqrt(
        summary["MSE"]
    )

    return summary.drop(
        columns=["MSE"]
    )


def summarize_test(test_predictions):

    summary = (
        test_predictions
        .groupby(
            ["horizon", "target", "method"]
        )
        .agg(
            machines=("machineID", "nunique"),
            samples=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            MSE=("squared_error", "mean"),
        )
        .reset_index()
    )

    summary["RMSE"] = np.sqrt(
        summary["MSE"]
    )

    summary = summary.drop(
        columns=["MSE"]
    )

    summary["rank_by_MAE"] = (
        summary
        .groupby(
            ["horizon", "target"]
        )["MAE"]
        .rank(method="min")
        .astype(int)
    )

    return summary.sort_values(
        ["horizon", "target", "MAE"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading cleaned Alibaba Cluster Trace dataset..."
    )

    machines = load_dataset()

    print(
        f"Machines included: {len(machines)}"
    )

    print("\nGenerating baseline predictions...")

    predictions = build_predictions(machines)

    validation_predictions = predictions[
        predictions["split"] == "validation"
    ].copy()

    print(
        "\nSelecting volatility-aware adjustment strength "
        "using validation data only..."
    )

    gamma_scores, selected_gamma = select_gamma(
        validation_predictions
    )

    print("\nSelected gamma values:")

    print(
        selected_gamma.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nBuilding unseen test comparison...")

    test_predictions = build_test_comparison(
        predictions,
        selected_gamma,
    )

    validation_summary = summarize_validation(
        predictions
    )

    test_summary = summarize_test(
        test_predictions
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    predictions_path = (
        OUTPUT_DIR / "improved_baseline_all_predictions.csv"
    )

    gamma_scores_path = (
        OUTPUT_DIR / "volatility_gamma_validation_scores.csv"
    )

    gamma_selection_path = (
        OUTPUT_DIR / "volatility_gamma_selected.csv"
    )

    validation_summary_path = (
        OUTPUT_DIR / "improved_baseline_validation_summary.csv"
    )

    test_predictions_path = (
        OUTPUT_DIR / "improved_baseline_test_predictions.csv"
    )

    test_summary_path = (
        OUTPUT_DIR / "improved_baseline_test_comparison.csv"
    )

    json_path = (
        OUTPUT_DIR / "improved_baseline_benchmark.json"
    )

    # Save CSV files.

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    gamma_scores.to_csv(
        gamma_scores_path,
        index=False,
    )

    selected_gamma.to_csv(
        gamma_selection_path,
        index=False,
    )

    validation_summary.to_csv(
        validation_summary_path,
        index=False,
    )

    test_predictions.to_csv(
        test_predictions_path,
        index=False,
    )

    test_summary.to_csv(
        test_summary_path,
        index=False,
    )

    # JSON report.

    report = {
        "experiment": "Improved forecasting baseline benchmark",

        "dataset": "Alibaba Cluster Trace 2017",

        "machines": int(len(machines)),

        "input_window_observations": WINDOW_SIZE,

        "observation_interval_minutes": 5,

        "forecast_horizons": HORIZONS,

        "validation_origins": [97, 115],

        "test_origins": [119, 137],

        "methods": [
            "Persistence",
            "EMA",
            "Weighted Moving Average",
            "Rolling Median",
            "Volatility-aware Persistence",
        ],

        "gamma_candidates": GAMMA_VALUES,

        "gamma_selection": "Validation MAE only",

        "test_summary": test_summary.to_dict(
            orient="records"
        ),

        "outputs": {

            "all_predictions": str(
                predictions_path.relative_to(PROJECT_ROOT)
            ),

            "gamma_validation_scores": str(
                gamma_scores_path.relative_to(PROJECT_ROOT)
            ),

            "selected_gamma": str(
                gamma_selection_path.relative_to(PROJECT_ROOT)
            ),

            "validation_summary": str(
                validation_summary_path.relative_to(PROJECT_ROOT)
            ),

            "test_predictions": str(
                test_predictions_path.relative_to(PROJECT_ROOT)
            ),

            "test_summary": str(
                test_summary_path.relative_to(PROJECT_ROOT)
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

    print("\nVALIDATION SUMMARY:")

    print(
        validation_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nUNSEEN TEST COMPARISON:")

    print(
        test_summary[
            [
                "horizon",
                "target",
                "method",
                "machines",
                "samples",
                "MAE",
                "RMSE",
                "rank_by_MAE",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved files:")

    print(predictions_path)
    print(gamma_scores_path)
    print(gamma_selection_path)
    print(validation_summary_path)
    print(test_predictions_path)
    print(test_summary_path)
    print(json_path)


if __name__ == "__main__":
    main()
