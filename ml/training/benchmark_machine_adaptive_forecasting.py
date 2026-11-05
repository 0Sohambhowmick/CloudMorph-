
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "server_usage_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"

STABLE_QUANTILE = 0.25
VOLATILE_QUANTILE = 0.75
WINDOW_SIZE = 12

HORIZONS = {
    "15min": 3,
    "30min": 6
}

TARGETS = {
    "CPU": "cpu",
    "Memory": "memory",
}

SPLITS = {
    "validation": range(97, 116),
    "test": range(119, 138),
}

METHODS = [
    "Persistence",
    "Moving Average",
    "Linear Trend"
]


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
        "util_memory"
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
        sort=True
    ):
        group = group.sort_values("step")

        if len(group) != 144:
            continue

        if not np.array_equal(
            group["step"].to_numpy(),
            np.arange(144)
        ):
            continue

        machines[int(machine_id)] = {
            "cpu": group["util_CPU"].to_numpy(
                dtype=np.float64
            ),
            "memory": group["util_memory"].to_numpy(
                dtype=np.float64
            ),
        }

    if not machines:
        raise ValueError(
            "No machines with exactly 144 consecutive steps were found."
        )

    return machines


def calculate_training_volatility(machines):

    rows = []

    for machine_id, series in machines.items():

        # Only training steps 0..99 are used.
        # Validation and test data are excluded.

        cpu_changes = np.abs(
            np.diff(series["cpu"][:100])
        )

        memory_changes = np.abs(
            np.diff(series["memory"][:100])
        )

        rows.append({
            "machineID": machine_id,

            "cpu_mean_abs_change": float(
                np.mean(cpu_changes)
            ),

            "memory_mean_abs_change": float(
                np.mean(memory_changes)
            ),
        })

    volatility = pd.DataFrame(rows)

    cpu_low = volatility[
        "cpu_mean_abs_change"
    ].quantile(STABLE_QUANTILE)

    cpu_high = volatility[
        "cpu_mean_abs_change"
    ].quantile(VOLATILE_QUANTILE)

    memory_low = volatility[
        "memory_mean_abs_change"
    ].quantile(STABLE_QUANTILE)

    memory_high = volatility[
        "memory_mean_abs_change"
    ].quantile(VOLATILE_QUANTILE)

    volatility["cpu_group"] = np.select(
        [
            volatility["cpu_mean_abs_change"] <= cpu_low,
            volatility["cpu_mean_abs_change"] >= cpu_high
        ],
        [
            "Stable",
            "Volatile"
        ],
        default="Middle"
    )

    volatility["memory_group"] = np.select(
        [
            volatility["memory_mean_abs_change"] <= memory_low,
            volatility["memory_mean_abs_change"] >= memory_high
        ],
        [
            "Stable",
            "Volatile"
        ],
        default="Middle"
    )

    thresholds = {
        "cpu_stable_max_mean_abs_change": float(cpu_low),

        "cpu_volatile_min_mean_abs_change": float(cpu_high),

        "memory_stable_max_mean_abs_change": float(memory_low),

        "memory_volatile_min_mean_abs_change": float(memory_high),

        "stable_quantile": STABLE_QUANTILE,

        "volatile_quantile": VOLATILE_QUANTILE,

        "volatility_calculation_steps": "0..99 (training period only)",
    }

    return volatility, thresholds


# ============================================================
# FORECASTING METHODS
# ============================================================

def predict_value(history, method, horizon_steps):

    history = np.asarray(
        history,
        dtype=np.float64
    )

    if method == "Persistence":

        return float(history[-1])

    if method == "Moving Average":

        return float(
            np.mean(history[-3:])
        )

    if method == "Linear Trend":

        recent = history[-6:]

        x = np.arange(
            len(recent),
            dtype=np.float64
        )

        slope, intercept = np.polyfit(
            x,
            recent,
            1
        )

        # Corrected:
        # Forecast 3 steps ahead for 15 minutes.
        # Forecast 6 steps ahead for 30 minutes.

        future_x = (
            len(recent) - 1 + horizon_steps
        )

        return float(
            slope * future_x + intercept
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(machines, volatility):

    volatility_lookup = (
        volatility
        .set_index("machineID")
        .to_dict("index")
    )

    rows = []

    for machine_id, series in machines.items():

        groups = volatility_lookup[machine_id]

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

                        group_name = (
                            groups["cpu_group"]
                            if target_name == "CPU"
                            else groups["memory_group"]
                        )

                        for method in METHODS:

                            # Pass the actual forecast horizon
                            # into every forecasting method.

                            prediction = predict_value(
                                history,
                                method,
                                horizon_steps
                            )

                            rows.append({

                                "machineID": machine_id,

                                "split": split_name,

                                "horizon": horizon_name,

                                "target": target_name,

                                "machine_group": group_name,

                                "method": method,

                                "forecast_origin_step": origin,

                                "target_step": target_step,

                                "actual": actual,

                                "prediction": prediction,

                                "absolute_error": abs(
                                    actual - prediction
                                ),

                                "squared_error": (
                                    actual - prediction
                                ) ** 2,

                            })

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise ValueError(
            "No forecast predictions were generated."
        )

    return predictions


# ============================================================
# SUMMARY
# ============================================================

def summarize(predictions):

    grouping = [
        "split",
        "horizon",
        "target",
        "machine_group",
        "method"
    ]

    summary = (
        predictions
        .groupby(grouping, dropna=False)
        .agg(
            samples=("absolute_error", "size"),

            machines=("machineID", "nunique"),

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
            [
                "split",
                "horizon",
                "target",
                "machine_group"
            ]
        )["MAE"]
        .rank(method="min")
        .astype(int)
    )

    return summary.sort_values(
        [
            "split",
            "horizon",
            "target",
            "machine_group",
            "MAE"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Loading cleaned Alibaba Cluster Trace dataset..."
    )

    machines = load_dataset()

    print(
        f"Machines included: {len(machines)}"
    )

    print(
        "Calculating volatility from training steps only..."
    )

    volatility, thresholds = (
        calculate_training_volatility(machines)
    )

    print(
        "Evaluating Persistence, Moving Average, and Linear Trend..."
    )

    predictions = evaluate(
        machines,
        volatility
    )

    summary = summarize(
        predictions
    )

    # Output files

    volatility_path = (
        OUTPUT_DIR /
        "machine_training_volatility_groups.csv"
    )

    predictions_path = (
        OUTPUT_DIR /
        "machine_adaptive_forecast_predictions.csv"
    )

    summary_path = (
        OUTPUT_DIR /
        "machine_adaptive_forecast_benchmark.csv"
    )

    json_path = (
        OUTPUT_DIR /
        "machine_adaptive_forecast_benchmark.json"
    )

    # Save CSV files

    volatility.to_csv(
        volatility_path,
        index=False
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # JSON report

    report = {

        "dataset": "Alibaba Cluster Trace 2017",

        "machines_evaluated": int(
            len(machines)
        ),

        "observations_per_machine": 144,

        "input_window_observations": WINDOW_SIZE,

        "observation_interval_minutes": 5,

        "forecast_horizons": HORIZONS,

        "split_origins": {
            "validation": [97, 115],
            "test": [119, 137]
        },

        "volatility_grouping": thresholds,

        "methods": METHODS,

        "metric_note": (
            "MAE and RMSE are in utilization percentage points."
        ),

        "summary": summary.to_dict(
            orient="records"
        ),

        "outputs": {

            "volatility_groups": str(
                volatility_path.relative_to(PROJECT_ROOT)
            ),

            "predictions": str(
                predictions_path.relative_to(PROJECT_ROOT)
            ),

            "summary_csv": str(
                summary_path.relative_to(PROJECT_ROOT)
            ),

        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # Display machine groups

    print("\nCPU and memory group counts:")

    print(
        "CPU:\n",
        volatility["cpu_group"].value_counts().to_string()
    )

    print(
        "Memory:\n",
        volatility["memory_group"].value_counts().to_string()
    )

    # Display benchmark results

    print("\nBenchmark summary:")

    print(
        summary[
            [
                "split",
                "horizon",
                "target",
                "machine_group",
                "method",
                "machines",
                "samples",
                "MAE",
                "RMSE",
                "rank_by_MAE"
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nSaved files:")

    print(volatility_path)

    print(predictions_path)

    print(summary_path)

    print(json_path)


if __name__ == "__main__":

    main()
