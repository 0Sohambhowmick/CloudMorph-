
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
    / "improved_baseline_test_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "models"

HORIZON = "15min"
TARGET = "Memory"

BASELINE_METHOD = "Persistence"
COMPARISON_METHOD = "Weighted Moving Average"

BOOTSTRAP_ITERATIONS = 10000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "machineID",
        "split",
        "horizon",
        "target",
        "method",
        "actual",
        "prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing)}"
        )

    df = df[
        (df["split"] == "test")
        & (df["horizon"] == HORIZON)
        & (df["target"] == TARGET)
        & (
            df["method"].isin(
                [BASELINE_METHOD, COMPARISON_METHOD]
            )
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No matching test predictions were found."
        )

    df["absolute_error"] = (
        df["actual"] - df["prediction"]
    ).abs()

    df["squared_error"] = (
        df["actual"] - df["prediction"]
    ) ** 2

    return df


# ============================================================
# BUILD PAIRED MACHINE METRICS
# ============================================================

def build_machine_metrics(df):

    machine_metrics = (
        df.groupby(
            ["machineID", "method"]
        )
        .agg(
            samples=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )

    mae_wide = machine_metrics.pivot(
        index="machineID",
        columns="method",
        values="mean_absolute_error",
    )

    mse_wide = machine_metrics.pivot(
        index="machineID",
        columns="method",
        values="mean_squared_error",
    )

    count_wide = machine_metrics.pivot(
        index="machineID",
        columns="method",
        values="samples",
    )

    required_methods = [
        BASELINE_METHOD,
        COMPARISON_METHOD,
    ]

    if not all(
        method in mae_wide.columns
        for method in required_methods
    ):
        raise ValueError(
            "One or both methods are missing from test data."
        )

    paired = pd.DataFrame(index=mae_wide.index)

    paired["persistence_mae"] = mae_wide[BASELINE_METHOD]
    paired["weighted_mae"] = mae_wide[COMPARISON_METHOD]

    paired["persistence_mse"] = mse_wide[BASELINE_METHOD]
    paired["weighted_mse"] = mse_wide[COMPARISON_METHOD]

    paired["persistence_samples"] = count_wide[BASELINE_METHOD]
    paired["weighted_samples"] = count_wide[COMPARISON_METHOD]

    paired = paired.dropna().reset_index()

    if len(paired) < 2:
        raise ValueError(
            "Not enough paired machines for bootstrap analysis."
        )

    if not (
        paired["persistence_samples"]
        == paired["weighted_samples"]
    ).all():
        raise ValueError(
            "Methods do not have matching sample counts per machine."
        )

    paired["mae_difference"] = (
        paired["weighted_mae"]
        - paired["persistence_mae"]
    )

    paired["mse_difference"] = (
        paired["weighted_mse"]
        - paired["persistence_mse"]
    )

    paired["weighted_wins"] = (
        paired["weighted_mae"]
        < paired["persistence_mae"]
    )

    paired["tie"] = np.isclose(
        paired["weighted_mae"],
        paired["persistence_mae"],
        atol=1e-12,
    )

    return paired


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def bootstrap_analysis(paired):

    rng = np.random.default_rng(RANDOM_STATE)

    n_machines = len(paired)

    persistence_mae = paired[
        "persistence_mae"
    ].to_numpy()

    weighted_mae = paired[
        "weighted_mae"
    ].to_numpy()

    persistence_mse = paired[
        "persistence_mse"
    ].to_numpy()

    weighted_mse = paired[
        "weighted_mse"
    ].to_numpy()

    observed_mae_difference = float(
        np.mean(weighted_mae - persistence_mae)
    )

    observed_persistence_mae = float(
        np.mean(persistence_mae)
    )

    observed_weighted_mae = float(
        np.mean(weighted_mae)
    )

    observed_persistence_rmse = float(
        np.sqrt(np.mean(persistence_mse))
    )

    observed_weighted_rmse = float(
        np.sqrt(np.mean(weighted_mse))
    )

    observed_rmse_difference = (
        observed_weighted_rmse
        - observed_persistence_rmse
    )

    bootstrap_mae_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    bootstrap_rmse_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64,
    )

    for iteration in range(BOOTSTRAP_ITERATIONS):

        # Resample complete machines, not individual rows.
        indices = rng.integers(
            0,
            n_machines,
            size=n_machines,
        )

        bootstrap_mae_differences[iteration] = (
            np.mean(weighted_mae[indices])
            - np.mean(persistence_mae[indices])
        )

        bootstrap_rmse_differences[iteration] = (
            np.sqrt(np.mean(weighted_mse[indices]))
            - np.sqrt(np.mean(persistence_mse[indices]))
        )

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    mae_ci = np.percentile(
        bootstrap_mae_differences,
        [lower_percentile, upper_percentile],
    )

    rmse_ci = np.percentile(
        bootstrap_rmse_differences,
        [lower_percentile, upper_percentile],
    )

    return {

        "machines": int(n_machines),

        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,

        "confidence_level": CONFIDENCE_LEVEL,

        "persistence_MAE": observed_persistence_mae,

        "weighted_MAE": observed_weighted_mae,

        "MAE_difference_weighted_minus_persistence":
            observed_mae_difference,

        "MAE_difference_95CI": {
            "lower": float(mae_ci[0]),
            "upper": float(mae_ci[1]),
        },

        "persistence_RMSE": observed_persistence_rmse,

        "weighted_RMSE": observed_weighted_rmse,

        "RMSE_difference_weighted_minus_persistence":
            observed_rmse_difference,

        "RMSE_difference_95CI": {
            "lower": float(rmse_ci[0]),
            "upper": float(rmse_ci[1]),
        },

        "weighted_machine_win_rate": float(
            paired["weighted_wins"].mean()
        ),

        "weighted_machine_win_count": int(
            paired["weighted_wins"].sum()
        ),

        "persistence_machine_win_count": int(
            (
                paired["weighted_mae"]
                > paired["persistence_mae"]
            ).sum()
        ),

        "machine_tie_count": int(
            paired["tie"].sum()
        ),

    }


# ============================================================
# INTERPRETATION
# ============================================================

def interpret_results(results):

    mae_lower = results[
        "MAE_difference_95CI"
    ]["lower"]

    mae_upper = results[
        "MAE_difference_95CI"
    ]["upper"]

    rmse_lower = results[
        "RMSE_difference_95CI"
    ]["lower"]

    rmse_upper = results[
        "RMSE_difference_95CI"
    ]["upper"]

    if mae_upper < 0:

        mae_conclusion = (
            "Weighted Moving Average has a lower MAE "
            "with a 95% paired bootstrap confidence interval "
            "below zero."
        )

    elif mae_lower > 0:

        mae_conclusion = (
            "Persistence has a lower MAE with a 95% paired "
            "bootstrap confidence interval above zero."
        )

    else:

        mae_conclusion = (
            "The 95% paired bootstrap confidence interval "
            "includes zero. The MAE difference is not clearly "
            "distinguishable from zero."
        )

    if rmse_lower > 0:

        rmse_conclusion = (
            "Weighted Moving Average has a higher RMSE "
            "with a confidence interval above zero."
        )

    elif rmse_upper < 0:

        rmse_conclusion = (
            "Weighted Moving Average has a lower RMSE "
            "with a confidence interval below zero."
        )

    else:

        rmse_conclusion = (
            "The RMSE difference confidence interval includes zero."
        )

    return {
        "MAE_interpretation": mae_conclusion,
        "RMSE_interpretation": rmse_conclusion,
        "production_recommendation": (
            "Keep Persistence as the production baseline until "
            "the evidence supports a reliable improvement."
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

    print(
        "Loading 15-minute Memory test predictions..."
    )

    df = load_data()

    print(
        f"Prediction rows: {len(df):,}"
    )

    print(
        f"Machines available: {df['machineID'].nunique()}"
    )

    print("\nBuilding paired machine-level metrics...")

    paired = build_machine_metrics(df)

    print(
        f"Paired machines: {len(paired)}"
    )

    print("\nRunning paired machine-level bootstrap...")

    results = bootstrap_analysis(paired)

    interpretation = interpret_results(results)

    report = {

        "experiment": "Paired machine-level bootstrap comparison",

        "dataset": "Alibaba Cluster Trace 2017",

        "horizon": HORIZON,

        "target": TARGET,

        "baseline_method": BASELINE_METHOD,

        "comparison_method": COMPARISON_METHOD,

        "resampling_unit": "Complete machine",

        "resampling_strategy": "Paired bootstrap with replacement",

        "random_state": RANDOM_STATE,

        "results": results,

        "interpretation": interpretation,
    }

    paired_path = (
        OUTPUT_DIR
        / "weighted_average_paired_machine_metrics.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "weighted_average_paired_bootstrap_report.json"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

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

    print("\nPAIRED BOOTSTRAP RESULTS:")

    print(
        f"Machines: {results['machines']}"
    )

    print(
        f"Bootstrap iterations: {results['bootstrap_iterations']}"
    )

    print(
        f"Persistence MAE: {results['persistence_MAE']:.6f}"
    )

    print(
        f"Weighted Moving Average MAE: "
        f"{results['weighted_MAE']:.6f}"
    )

    print(
        f"MAE difference (Weighted - Persistence): "
        f"{results['MAE_difference_weighted_minus_persistence']:.6f}"
    )

    print(
        "MAE difference 95% CI: "
        f"[{results['MAE_difference_95CI']['lower']:.6f}, "
        f"{results['MAE_difference_95CI']['upper']:.6f}]"
    )

    print(
        f"\nPersistence RMSE: "
        f"{results['persistence_RMSE']:.6f}"
    )

    print(
        f"Weighted Moving Average RMSE: "
        f"{results['weighted_RMSE']:.6f}"
    )

    print(
        f"RMSE difference (Weighted - Persistence): "
        f"{results['RMSE_difference_weighted_minus_persistence']:.6f}"
    )

    print(
        "RMSE difference 95% CI: "
        f"[{results['RMSE_difference_95CI']['lower']:.6f}, "
        f"{results['RMSE_difference_95CI']['upper']:.6f}]"
    )

    print(
        f"\nWeighted Moving Average machine win rate: "
        f"{results['weighted_machine_win_rate'] * 100:.2f}%"
    )

    print(
        f"Machines where Weighted Moving Average wins: "
        f"{results['weighted_machine_win_count']}"
    )

    print(
        f"Machines where Persistence wins: "
        f"{results['persistence_machine_win_count']}"
    )

    print(
        f"Machine ties: {results['machine_tie_count']}"
    )

    print("\nINTERPRETATION:")

    print(
        interpretation["MAE_interpretation"]
    )

    print(
        interpretation["RMSE_interpretation"]
    )

    print(
        interpretation["production_recommendation"]
    )

    print("\nSaved files:")

    print(paired_path)
    print(json_path)


if __name__ == "__main__":
    main()
