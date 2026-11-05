
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
    / "rolling_origin_backtest_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"

METHOD_BASELINE = "Persistence"
METHOD_CANDIDATE = "Weighted Moving Average"

HORIZONS = ["15min", "30min"]
TARGETS = ["CPU", "Memory"]

BOOTSTRAP_ITERATIONS = 2000
TIME_BLOCK_LENGTH = 6

RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "machineID",
        "forecast_origin_step",
        "horizon",
        "target",
        "method",
        "actual",
        "prediction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        df["method"].isin(
            [METHOD_BASELINE, METHOD_CANDIDATE]
        )
    ].copy()

    df["absolute_error"] = (
        df["actual"] - df["prediction"]
    ).abs()

    df["squared_error"] = (
        df["actual"] - df["prediction"]
    ) ** 2

    if df.empty:
        raise ValueError(
            "No forecasting predictions found."
        )

    return df


# ============================================================
# PREPARE ERROR MATRICES
# ============================================================

def prepare_task_matrices(
    df,
    horizon,
    target,
):

    task = df[
        (df["horizon"] == horizon)
        & (df["target"] == target)
    ].copy()

    if task.empty:
        raise ValueError(
            f"No data for {horizon}, {target}"
        )

    machines = sorted(
        task["machineID"].unique()
    )

    origins = sorted(
        task["forecast_origin_step"].unique()
    )

    expected_rows = (
        len(machines)
        * len(origins)
        * 2
    )

    if len(task) != expected_rows:
        raise ValueError(
            f"Unexpected row count for {horizon} {target}. "
            f"Expected {expected_rows}, found {len(task)}."
        )

    if task.duplicated(
        subset=[
            "machineID",
            "forecast_origin_step",
            "method",
        ]
    ).any():
        raise ValueError(
            "Duplicate machine/origin/method combinations found."
        )

    matrices = {}

    for method in [
        METHOD_BASELINE,
        METHOD_CANDIDATE,
    ]:

        method_data = task[
            task["method"] == method
        ]

        absolute_error = (
            method_data.pivot(
                index="machineID",
                columns="forecast_origin_step",
                values="absolute_error",
            )
            .reindex(
                index=machines,
                columns=origins,
            )
        )

        squared_error = (
            method_data.pivot(
                index="machineID",
                columns="forecast_origin_step",
                values="squared_error",
            )
            .reindex(
                index=machines,
                columns=origins,
            )
        )

        if absolute_error.isna().any().any():
            raise ValueError(
                f"Missing absolute errors for {method}."
            )

        if squared_error.isna().any().any():
            raise ValueError(
                f"Missing squared errors for {method}."
            )

        matrices[method] = {
            "absolute_error": absolute_error.to_numpy(
                dtype=np.float64
            ),
            "squared_error": squared_error.to_numpy(
                dtype=np.float64
            ),
        }

    return machines, origins, matrices


# ============================================================
# TWO-WAY RESAMPLING
# ============================================================

def sample_time_block_weights(
    rng,
    number_of_origins,
):

    block_length = min(
        TIME_BLOCK_LENGTH,
        number_of_origins,
    )

    number_of_blocks = int(
        np.ceil(
            number_of_origins / block_length
        )
    )

    sampled_positions = []

    for _ in range(number_of_blocks):

        start = rng.integers(
            0,
            number_of_origins,
        )

        block = (
            start
            + np.arange(block_length)
        ) % number_of_origins

        sampled_positions.extend(
            block.tolist()
        )

    sampled_positions = np.array(
        sampled_positions[:number_of_origins],
        dtype=int,
    )

    return np.bincount(
        sampled_positions,
        minlength=number_of_origins,
    ).astype(np.float64)


# ============================================================
# BOOTSTRAP ONE TASK
# ============================================================

def bootstrap_task(
    machines,
    origins,
    matrices,
    rng,
):

    number_of_machines = len(machines)
    number_of_origins = len(origins)

    if number_of_origins < 2:
        raise ValueError(
            "At least two forecast origins are required."
        )

    persistence_abs = matrices[
        METHOD_BASELINE
    ]["absolute_error"]

    weighted_abs = matrices[
        METHOD_CANDIDATE
    ]["absolute_error"]

    persistence_sq = matrices[
        METHOD_BASELINE
    ]["squared_error"]

    weighted_sq = matrices[
        METHOD_CANDIDATE
    ]["squared_error"]

    # Observed metrics.
    persistence_mae = float(
        persistence_abs.mean()
    )

    weighted_mae = float(
        weighted_abs.mean()
    )

    persistence_rmse = float(
        np.sqrt(persistence_sq.mean())
    )

    weighted_rmse = float(
        np.sqrt(weighted_sq.mean())
    )

    observed_mae_difference = (
        weighted_mae - persistence_mae
    )

    observed_rmse_difference = (
        weighted_rmse - persistence_rmse
    )

    bootstrap_mae_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    bootstrap_rmse_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    bootstrap_relative_improvements = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):

        # Resample machines with replacement.
        sampled_machines = rng.integers(
            0,
            number_of_machines,
            size=number_of_machines,
        )

        machine_weights = np.bincount(
            sampled_machines,
            minlength=number_of_machines,
        ).astype(np.float64)

        # Resample contiguous time blocks.
        time_weights = sample_time_block_weights(
            rng,
            number_of_origins,
        )

        denominator = (
            machine_weights.sum()
            * time_weights.sum()
        )

        # Weighted MAE for each method.
        persistence_sample_mae = (
            machine_weights
            @ persistence_abs
            @ time_weights
        ) / denominator

        weighted_sample_mae = (
            machine_weights
            @ weighted_abs
            @ time_weights
        ) / denominator

        # Weighted RMSE for each method.
        persistence_sample_rmse = np.sqrt(
            (
                machine_weights
                @ persistence_sq
                @ time_weights
            ) / denominator
        )

        weighted_sample_rmse = np.sqrt(
            (
                machine_weights
                @ weighted_sq
                @ time_weights
            ) / denominator
        )

        mae_difference = (
            weighted_sample_mae
            - persistence_sample_mae
        )

        rmse_difference = (
            weighted_sample_rmse
            - persistence_sample_rmse
        )

        bootstrap_mae_differences[iteration] = (
            mae_difference
        )

        bootstrap_rmse_differences[iteration] = (
            rmse_difference
        )

        bootstrap_relative_improvements[iteration] = (
            (
                persistence_sample_mae
                - weighted_sample_mae
            )
            / persistence_sample_mae
        ) * 100.0

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    mae_ci = np.percentile(
        bootstrap_mae_differences,
        [lower, upper],
    )

    rmse_ci = np.percentile(
        bootstrap_rmse_differences,
        [lower, upper],
    )

    improvement_ci = np.percentile(
        bootstrap_relative_improvements,
        [lower, upper],
    )

    if mae_ci[1] < 0:
        mae_conclusion = (
            "Weighted Moving Average has a lower MAE "
            "with a 95% confidence interval below zero."
        )

    elif mae_ci[0] > 0:
        mae_conclusion = (
            "Persistence has a lower MAE with a "
            "95% confidence interval above zero."
        )

    else:
        mae_conclusion = (
            "The MAE confidence interval includes zero. "
            "The difference is not clearly distinguishable "
            "from zero."
        )

    if rmse_ci[0] > 0:
        rmse_conclusion = (
            "Weighted Moving Average has a higher RMSE "
            "with a 95% confidence interval above zero."
        )

    elif rmse_ci[1] < 0:
        rmse_conclusion = (
            "Weighted Moving Average has a lower RMSE "
            "with a 95% confidence interval below zero."
        )

    else:
        rmse_conclusion = (
            "The RMSE confidence interval includes zero."
        )

    return {

        "machines": number_of_machines,

        "forecast_origins": number_of_origins,

        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,

        "time_block_length": TIME_BLOCK_LENGTH,

        "persistence_MAE": persistence_mae,

        "weighted_MAE": weighted_mae,

        "MAE_difference_weighted_minus_persistence":
            observed_mae_difference,

        "MAE_difference_95CI": {
            "lower": float(mae_ci[0]),
            "upper": float(mae_ci[1]),
        },

        "persistence_RMSE": persistence_rmse,

        "weighted_RMSE": weighted_rmse,

        "RMSE_difference_weighted_minus_persistence":
            observed_rmse_difference,

        "RMSE_difference_95CI": {
            "lower": float(rmse_ci[0]),
            "upper": float(rmse_ci[1]),
        },

        "relative_MAE_improvement_percent": float(
            (
                persistence_mae - weighted_mae
            )
            / persistence_mae
            * 100
        ),

        "relative_MAE_improvement_95CI": {
            "lower": float(improvement_ci[0]),
            "upper": float(improvement_ci[1]),
        },

        "bootstrap_probability_weighted_MAE_better": float(
            np.mean(
                bootstrap_mae_differences < 0
            )
        ),

        "MAE_interpretation": mae_conclusion,

        "RMSE_interpretation": rmse_conclusion,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CLOUDMORPH TWO-WAY BLOCK BOOTSTRAP")
    print("=" * 70)

    df = load_data()

    print(
        f"\nPrediction rows: {len(df):,}"
    )

    print(
        f"Machines: {df['machineID'].nunique()}"
    )

    print(
        f"Forecast origins: "
        f"{df['forecast_origin_step'].nunique()}"
    )

    print(
        f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}"
    )

    print(
        f"Time block length: {TIME_BLOCK_LENGTH} origins"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    all_results = []

    for horizon in HORIZONS:

        for target in TARGETS:

            print("\n" + "-" * 70)

            print(
                f"Evaluating {horizon} {target}..."
            )

            machines, origins, matrices = (
                prepare_task_matrices(
                    df,
                    horizon,
                    target,
                )
            )

            result = bootstrap_task(
                machines,
                origins,
                matrices,
                rng,
            )

            result["horizon"] = horizon
            result["target"] = target

            all_results.append(
                result
            )

            print(
                f"Persistence MAE: "
                f"{result['persistence_MAE']:.5f}"
            )

            print(
                f"Weighted MAE: "
                f"{result['weighted_MAE']:.5f}"
            )

            print(
                f"MAE difference: "
                f"{result['MAE_difference_weighted_minus_persistence']:.5f}"
            )

            print(
                "MAE 95% CI: "
                f"[{result['MAE_difference_95CI']['lower']:.5f}, "
                f"{result['MAE_difference_95CI']['upper']:.5f}]"
            )

            print(
                f"Persistence RMSE: "
                f"{result['persistence_RMSE']:.5f}"
            )

            print(
                f"Weighted RMSE: "
                f"{result['weighted_RMSE']:.5f}"
            )

            print(
                f"RMSE difference: "
                f"{result['RMSE_difference_weighted_minus_persistence']:.5f}"
            )

            print(
                "RMSE 95% CI: "
                f"[{result['RMSE_difference_95CI']['lower']:.5f}, "
                f"{result['RMSE_difference_95CI']['upper']:.5f}]"
            )

            print(
                f"Relative MAE improvement: "
                f"{result['relative_MAE_improvement_percent']:.2f}%"
            )

            print(
                f"Bootstrap probability WMA has lower MAE: "
                f"{result['bootstrap_probability_weighted_MAE_better'] * 100:.2f}%"
            )

            print(
                "\nMAE interpretation: "
                f"{result['MAE_interpretation']}"
            )

            print(
                "RMSE interpretation: "
                f"{result['RMSE_interpretation']}"
            )

    results_df = pd.DataFrame(
        all_results
    )

    csv_path = (
        OUTPUT_DIR
        / "rolling_origin_two_way_bootstrap_results.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "rolling_origin_two_way_bootstrap_report.json"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    report = {

        "experiment": "Two-way machine and time block bootstrap",

        "dataset": "Alibaba Cluster Trace 2017",

        "baseline": METHOD_BASELINE,

        "candidate": METHOD_CANDIDATE,

        "resampling": {
            "machine": "Machines sampled with replacement",
            "time": "Circular moving block bootstrap",
            "time_block_length": TIME_BLOCK_LENGTH,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "random_state": RANDOM_STATE,
            "confidence_level": CONFIDENCE_LEVEL,
        },

        "results": all_results,

        "important_note": (
            "This bootstrap accounts for machine and local time "
            "dependence under the selected block length. It does "
            "not remove model-selection bias from prior experiments, "
            "and results may be sensitive to block length."
        ),

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

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "horizon",
                "target",
                "relative_MAE_improvement_percent",
                "MAE_difference_95CI",
                "RMSE_difference_95CI",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(csv_path)
    print(json_path)

    print(
        "\nTwo-way block bootstrap completed successfully."
    )


if __name__ == "__main__":
    main()
