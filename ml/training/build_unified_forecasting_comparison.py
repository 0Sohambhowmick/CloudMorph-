
import csv
import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

BASELINE_PATH = MODEL_DIR / "improved_baseline_test_comparison.csv"
V3_PATH = MODEL_DIR / "training_metrics_v3.json"
V4_PATH = MODEL_DIR / "training_metrics_v4.json"
ROLLING_PATH = MODEL_DIR / "rolling_origin_overall_comparison.csv"

FIXED_CSV_PATH = MODEL_DIR / "unified_fixed_test_comparison.csv"
ROLLING_CSV_PATH = MODEL_DIR / "unified_rolling_origin_comparison.csv"
REPORT_PATH = MODEL_DIR / "unified_forecasting_comparison.json"


# ============================================================
# FILE LOADING
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# ============================================================
# FIXED TEST COMPARISON
# ============================================================

def build_fixed_comparison():

    baseline_rows = load_csv(BASELINE_PATH)

    v3 = load_json(V3_PATH)
    v4 = load_json(V4_PATH)

    rows = []

    # Include Persistence and WMA from the same fixed test.
    for item in baseline_rows:

        method = item["method"]

        if method not in [
            "Persistence",
            "Weighted Moving Average",
        ]:
            continue

        rows.append({
            "evaluation": "fixed_test",
            "horizon": item["horizon"],
            "target": item["target"],
            "method": method,
            "machines": int(item["machines"]),
            "samples": int(item["samples"]),
            "MAE": float(item["MAE"]),
            "RMSE": float(item["RMSE"]),
        })

    # Add LSTM V3 and V4 results.
    experiments = [
        ("LSTM V3", v3),
        ("LSTM V4", v4),
    ]

    for model_name, experiment in experiments:

        for horizon, targets in experiment["test_metrics"].items():

            for target_name, scores in targets.items():

                target = (
                    "CPU"
                    if target_name == "CPU utilization"
                    else "Memory"
                )

                lstm_scores = scores["LSTM"]

                rows.append({
                    "evaluation": "fixed_test",
                    "horizon": horizon,
                    "target": target,
                    "method": model_name,
                    "machines": 975,
                    "samples": experiment["test_sample_count"],
                    "MAE": float(lstm_scores["MAE"]),
                    "RMSE": float(lstm_scores["RMSE"]),
                })

    rows.sort(
        key=lambda x: (
            x["horizon"],
            x["target"],
            x["MAE"],
        )
    )

    return rows


# ============================================================
# ROLLING ORIGIN COMPARISON
# ============================================================

def build_rolling_comparison():

    source_rows = load_csv(ROLLING_PATH)

    rows = []

    for item in source_rows:

        rows.append({
            "evaluation": "rolling_origin",
            "fold": item["fold"],
            "horizon": item["horizon"],
            "target": item["target"],
            "Persistence_MAE": float(item["MAE_Persistence"]),
            "WMA_MAE": float(
                item["MAE_Weighted_Moving_Average"]
            ),
            "Persistence_RMSE": float(item["RMSE_Persistence"]),
            "WMA_RMSE": float(
                item["RMSE_Weighted_Moving_Average"]
            ),
        })

    return rows


# ============================================================
# WINNER IDENTIFICATION
# ============================================================

def identify_fixed_test_winners(rows):

    grouped = {}

    for row in rows:

        key = (row["horizon"], row["target"])

        grouped.setdefault(key, []).append(row)

    winners = []

    for (horizon, target), methods in grouped.items():

        best_mae = min(methods, key=lambda x: x["MAE"])
        best_rmse = min(methods, key=lambda x: x["RMSE"])

        winners.append({
            "horizon": horizon,
            "target": target,
            "best_MAE_method": best_mae["method"],
            "best_MAE": best_mae["MAE"],
            "best_RMSE_method": best_rmse["method"],
            "best_RMSE": best_rmse["RMSE"],
        })

    return winners


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLOUDMORPH - UNIFIED FORECASTING COMPARISON")
    print("=" * 70)

    fixed_rows = build_fixed_comparison()

    rolling_rows = build_rolling_comparison()

    winners = identify_fixed_test_winners(fixed_rows)

    report = {
        "experiment": "CloudMorph Unified Forecasting Comparison",

        "fixed_test_evaluation": {
            "description": (
                "Persistence, WMA, LSTM V3 and LSTM V4 "
                "evaluated on the existing fixed test split."
            ),
            "results": fixed_rows,
            "winners": winners,
        },

        "rolling_origin_evaluation": {
            "description": (
                "Persistence and WMA evaluated across "
                "the existing rolling forecast origins."
            ),
            "results": rolling_rows,
        },

        "interpretation": {
            "fixed_test": (
                "Compare methods using the same fixed test samples."
            ),
            "rolling_origin": (
                "Compare Persistence and WMA across multiple "
                "chronological forecast origins."
            ),
            "important_note": (
                "Fixed-test and rolling-origin metrics must not "
                "be combined into a single numerical ranking."
            ),
        },
    }

    save_csv(
        FIXED_CSV_PATH,
        fixed_rows,
    )

    save_csv(
        ROLLING_CSV_PATH,
        rolling_rows,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\nFIXED TEST COMPARISON")

    for winner in winners:

        print(
            f"\n{winner['horizon']} | {winner['target']}"
        )

        print(
            f"Best MAE: {winner['best_MAE_method']} "
            f"({winner['best_MAE']:.4f})"
        )

        print(
            f"Best RMSE: {winner['best_RMSE_method']} "
            f"({winner['best_RMSE']:.4f})"
        )

    print("\nROLLING ORIGIN COMPARISON")

    for row in rolling_rows:

        print(
            f"{row['horizon']} | {row['target']} | "
            f"Persistence MAE={row['Persistence_MAE']:.4f} | "
            f"WMA MAE={row['WMA_MAE']:.4f}"
        )

    print("\nREPORTS GENERATED:")

    print(FIXED_CSV_PATH)
    print(ROLLING_CSV_PATH)
    print(REPORT_PATH)

    print("\nUNIFIED FORECASTING COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
