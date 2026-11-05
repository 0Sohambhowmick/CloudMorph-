
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

BASELINE = "Persistence"
CANDIDATE = "Weighted Moving Average"

BLOCK_LENGTHS = [3, 6, 12, 18]

BOOTSTRAP_ITERATIONS = 5000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95

TASKS = [
    ("15min", "CPU"),
    ("15min", "Memory"),
    ("30min", "CPU"),
    ("30min", "Memory"),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "machineID",
        "forecast_origin_step",
        "horizon",
        "target",
        "method",
        "actual",
        "prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        df["method"].isin(
            [BASELINE, CANDIDATE]
        )
    ].copy()

    df["absolute_error"] = (
        df["actual"] - df["prediction"]
    ).abs()

    df["squared_error"] = (
        df["actual"] - df["prediction"]
    ) ** 2

    return df


# ============================================================
# PREPARE ERROR MATRICES
# ============================================================

def prepare_matrices(df, horizon, target):

    task = df[
        (df["horizon"] == horizon)
        & (df["target"] == target)
    ].copy()

    if task.empty:
        raise ValueError(
            f"No data for {horizon} {target}"
        )

    machines = sorted(
        task["machineID"].unique()
    )

    origins = sorted(
        task["forecast_origin_step"].unique()
    )

    matrices = {}

    for method in [BASELINE, CANDIDATE]:

        method_data = task[
            task["method"] == method
        ]

        abs_matrix = (
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

        sq_matrix = (
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

        if abs_matrix.isna().any().any():
            raise ValueError(
                f"Missing absolute errors for {method}"
            )

        if sq_matrix.isna().any().any():
            raise ValueError(
                f"Missing squared errors for {method}"
            )

        matrices[method] = {
            "absolute": abs_matrix.to_numpy(
                dtype=np.float64
            ),
            "squared": sq_matrix.to_numpy(
                dtype=np.float64
            ),
        }

    return machines, origins, matrices


# ============================================================
# CIRCULAR TIME-BLOCK RESAMPLING
# ============================================================

def sample_time_weights(
    rng,
    number_of_origins,
    block_length,
):

    block_length = min(
        block_length,
        number_of_origins,
    )

    number_of_blocks = int(
        np.ceil(
            number_of_origins / block_length
        )
    )

    positions = []

    for _ in range(number_of_blocks):

        start = rng.integers(
            0,
            number_of_origins,
        )

        block = (
            start
            + np.arange(block_length)
        ) % number_of_origins

        positions.extend(
            block.tolist()
        )

    positions = np.array(
        positions[:number_of_origins],
        dtype=int,
    )

    return np.bincount(
        positions,
        minlength=number_of_origins,
    ).astype(np.float64)


# ============================================================
# RUN BOOTSTRAP FOR ONE BLOCK LENGTH
# ============================================================

def run_bootstrap(
    matrices,
    number_of_machines,
    number_of_origins,
    block_length,
    rng,
):

    persistence_abs = matrices[
        BASELINE
    ]["absolute"]

    weighted_abs = matrices[
        CANDIDATE
    ]["absolute"]

    persistence_sq = matrices[
        BASELINE
    ]["squared"]

    weighted_sq = matrices[
        CANDIDATE
    ]["squared"]

    bootstrap_mae = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    bootstrap_rmse = np.empty(
        BOOTSTRAP_ITERATIONS
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):

        machine_indices = rng.integers(
            0,
            number_of_machines,
            size=number_of_machines,
        )

        machine_weights = np.bincount(
            machine_indices,
            minlength=number_of_machines,
        ).astype(np.float64)

        time_weights = sample_time_weights(
            rng,
            number_of_origins,
            block_length,
        )

        denominator = (
            machine_weights.sum()
            * time_weights.sum()
        )

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

        bootstrap_mae[iteration] = (
            weighted_sample_mae
            - persistence_sample_mae
        )

        bootstrap_rmse[iteration] = (
            weighted_sample_rmse
            - persistence_sample_rmse
        )

    alpha = 1 - CONFIDENCE_LEVEL

    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    mae_ci = np.percentile(
        bootstrap_mae,
        [lower, upper],
    )

    rmse_ci = np.percentile(
        bootstrap_rmse,
        [lower, upper],
    )

    return {
        "MAE_CI_lower": float(mae_ci[0]),
        "MAE_CI_upper": float(mae_ci[1]),

        "RMSE_CI_lower": float(rmse_ci[0]),
        "RMSE_CI_upper": float(rmse_ci[1]),

        "MAE_probability_WMA_better": float(
            np.mean(bootstrap_mae < 0)
        ),

        "RMSE_probability_WMA_better": float(
            np.mean(bootstrap_rmse < 0)
        ),
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
    print("CLOUDMORPH BLOCK-LENGTH SENSITIVITY ANALYSIS")
    print("=" * 70)

    df = load_data()

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    all_results = []

    for horizon, target in TASKS:

        print("\n" + "-" * 70)

        print(
            f"Task: {horizon} {target}"
        )

        machines, origins, matrices = (
            prepare_matrices(
                df,
                horizon,
                target,
            )
        )

        number_of_machines = len(machines)
        number_of_origins = len(origins)

        persistence_mae = float(
            matrices[BASELINE]["absolute"].mean()
        )

        weighted_mae = float(
            matrices[CANDIDATE]["absolute"].mean()
        )

        persistence_rmse = float(
            np.sqrt(
                matrices[BASELINE]["squared"].mean()
            )
        )

        weighted_rmse = float(
            np.sqrt(
                matrices[CANDIDATE]["squared"].mean()
            )
        )

        print(
            f"Persistence MAE: {persistence_mae:.5f}"
        )

        print(
            f"Weighted MAE: {weighted_mae:.5f}"
        )

        print(
            f"Persistence RMSE: {persistence_rmse:.5f}"
        )

        print(
            f"Weighted RMSE: {weighted_rmse:.5f}"
        )

        for block_length in BLOCK_LENGTHS:

            print(
                f"\nTesting block length: "
                f"{block_length} origins "
                f"({block_length * 5} minutes)"
            )

            result = run_bootstrap(
                matrices,
                number_of_machines,
                number_of_origins,
                block_length,
                rng,
            )

            result.update({

                "horizon": horizon,

                "target": target,

                "block_length_origins": block_length,

                "block_length_minutes": (
                    block_length * 5
                ),

                "bootstrap_iterations": (
                    BOOTSTRAP_ITERATIONS
                ),

                "machines": number_of_machines,

                "forecast_origins": number_of_origins,

                "MAE_difference": (
                    weighted_mae - persistence_mae
                ),

                "RMSE_difference": (
                    weighted_rmse - persistence_rmse
                ),

            })

            all_results.append(
                result
            )

            print(
                "MAE CI: "
                f"[{result['MAE_CI_lower']:.5f}, "
                f"{result['MAE_CI_upper']:.5f}]"
            )

            print(
                "RMSE CI: "
                f"[{result['RMSE_CI_lower']:.5f}, "
                f"{result['RMSE_CI_upper']:.5f}]"
            )

            print(
                "Probability WMA has lower MAE: "
                f"{result['MAE_probability_WMA_better'] * 100:.2f}%"
            )

            print(
                "Probability WMA has lower RMSE: "
                f"{result['RMSE_probability_WMA_better'] * 100:.2f}%"
            )

    results_df = pd.DataFrame(
        all_results
    )

    csv_path = (
        OUTPUT_DIR
        / "block_length_sensitivity_results.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "block_length_sensitivity_report.json"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    report = {
        "experiment": "Block-length sensitivity analysis",

        "dataset": "Alibaba Cluster Trace 2017",

        "baseline": BASELINE,

        "candidate": CANDIDATE,

        "block_lengths_origins": BLOCK_LENGTHS,

        "block_lengths_minutes": [
            value * 5
            for value in BLOCK_LENGTHS
        ],

        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,

        "confidence_level": CONFIDENCE_LEVEL,

        "results": all_results,

        "note": (
            "Confidence intervals are conditional on the "
            "selected circular block bootstrap design. "
            "This sensitivity analysis does not eliminate "
            "model-selection bias or provide an independent "
            "confirmatory test."
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
    print("FINAL BLOCK-LENGTH COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "horizon",
                "target",
                "block_length_minutes",
                "MAE_CI_lower",
                "MAE_CI_upper",
                "RMSE_CI_lower",
                "RMSE_CI_upper",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\nSaved files:")

    print(csv_path)
    print(json_path)

    print(
        "\nBlock-length sensitivity analysis completed."
    )


if __name__ == "__main__":
    main()
