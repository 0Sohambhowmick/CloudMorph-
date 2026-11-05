
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    ROOT / "datasets" / "processed" / "server_usage_clean.csv"
)

ORIGINAL_PREDICTIONS_PATH = (
    ROOT / "ml" / "models" / "lstm_sequence_classifier"
    / "test_predictions.csv"
)

PLATT_PREDICTIONS_PATH = (
    ROOT / "ml" / "models" / "lstm_high_risk_platt"
    / "test_predictions_platt.csv"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "lstm_platt_machine_patterns"
)

KEYS = [
    "machineID",
    "history_start_step",
    "history_end_step",
    "target_end_step",
]

HIGH_LABEL = "HIGH_DEMAND"

ORIGINAL_THRESHOLD = 0.11
PLATT_THRESHOLD = 0.01

HISTORY_LENGTH = 12
FUTURE_LENGTH = 6

CPU_COLUMN = "util_CPU"
MEMORY_COLUMN = "util_memory"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    required_files = [
        RAW_DATA_PATH,
        ORIGINAL_PREDICTIONS_PATH,
        PLATT_PREDICTIONS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_csv(RAW_DATA_PATH)

    original = pd.read_csv(
        ORIGINAL_PREDICTIONS_PATH
    )

    platt = pd.read_csv(
        PLATT_PREDICTIONS_PATH
    )

    required_raw = [
        "machineID",
        CPU_COLUMN,
        MEMORY_COLUMN,
    ]

    missing = set(required_raw) - set(raw.columns)

    if missing:
        raise ValueError(
            f"Raw dataset missing columns: {missing}"
        )

    for df in [raw, original, platt]:

        df["machineID"] = (
            df["machineID"].astype(str)
        )

    for df in [original, platt]:

        for column in [
            "history_start_step",
            "history_end_step",
            "target_end_step",
        ]:
            df[column] = df[column].astype(int)

    return raw, original, platt


# =========================================================
# VALIDATE PREDICTIONS
# =========================================================

def merge_predictions(original, platt):

    merged = original.merge(
        platt,
        on=KEYS,
        how="inner",
        suffixes=("_original", "_platt"),
        validate="one_to_one",
    )

    if len(merged) != len(original) or len(merged) != len(platt):
        raise ValueError(
            "Original and Platt prediction windows do not match."
        )

    if not (
        merged["target_original"]
        == merged["target_platt"]
    ).all():
        raise ValueError(
            "Original and Platt labels do not match."
        )

    merged["actual_high"] = (
        merged["target_original"] == HIGH_LABEL
    )

    merged["original_probability"] = (
        merged["high_demand_probability"]
    )

    merged["platt_probability"] = (
        merged["platt_high_probability"]
    )

    merged["original_alert"] = (
        merged["original_probability"]
        >= ORIGINAL_THRESHOLD
    )

    merged["platt_alert"] = (
        merged["platt_probability"]
        >= PLATT_THRESHOLD
    )

    merged["original_false_positive"] = (
        merged["original_alert"]
        & ~merged["actual_high"]
    )

    merged["platt_false_positive"] = (
        merged["platt_alert"]
        & ~merged["actual_high"]
    )

    return merged


# =========================================================
# PREPARE MACHINE SEQUENCES
# =========================================================

def prepare_machine_sequences(raw):

    raw = raw.copy()

    raw["machineID"] = (
        raw["machineID"].astype(str)
    )

    # The processed dataset is expected to be ordered
    # chronologically within each machine.
    raw = raw.reset_index(drop=True)

    machine_sequences = {}

    for machine_id, group in raw.groupby(
        "machineID",
        sort=False,
    ):

        group = group.reset_index(drop=True)

        machine_sequences[machine_id] = {
            CPU_COLUMN: group[CPU_COLUMN].to_numpy(
                dtype=float
            ),
            MEMORY_COLUMN: group[MEMORY_COLUMN].to_numpy(
                dtype=float
            ),
        }

    return machine_sequences


# =========================================================
# WINDOW STATISTICS
# =========================================================

def calculate_statistics(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return {}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "first": float(values[0]),
        "last": float(values[-1]),
        "change": float(values[-1] - values[0]),
    }


def extract_window_features(
    machine_sequences,
    prediction,
):

    machine_id = prediction["machineID"]

    if machine_id not in machine_sequences:
        return None

    machine = machine_sequences[machine_id]

    history_start = int(
        prediction["history_start_step"]
    )

    history_end = int(
        prediction["history_end_step"]
    )

    target_end = int(
        prediction["target_end_step"]
    )

    expected_history_end = (
        history_start + HISTORY_LENGTH - 1
    )

    expected_target_end = (
        history_end + FUTURE_LENGTH
    )

    if history_end != expected_history_end:
        raise ValueError(
            f"Unexpected history window for machine {machine_id}"
        )

    if target_end != expected_target_end:
        raise ValueError(
            f"Unexpected target window for machine {machine_id}"
        )

    cpu = machine[CPU_COLUMN]
    memory = machine[MEMORY_COLUMN]

    if target_end >= len(cpu):
        raise ValueError(
            f"Target window exceeds raw data for machine {machine_id}"
        )

    history_slice = slice(
        history_start,
        history_end + 1,
    )

    future_slice = slice(
        history_end + 1,
        target_end + 1,
    )

    history_cpu = cpu[history_slice]
    history_memory = memory[history_slice]

    future_cpu = cpu[future_slice]
    future_memory = memory[future_slice]

    if (
        len(history_cpu) != HISTORY_LENGTH
        or len(history_memory) != HISTORY_LENGTH
        or len(future_cpu) != FUTURE_LENGTH
        or len(future_memory) != FUTURE_LENGTH
    ):
        raise ValueError(
            f"Incorrect window length for machine {machine_id}"
        )

    cpu_hist = calculate_statistics(
        history_cpu
    )

    memory_hist = calculate_statistics(
        history_memory
    )

    cpu_recent = history_cpu[-6:]
    memory_recent = history_memory[-6:]

    cpu_previous = history_cpu[:6]
    memory_previous = history_memory[:6]

    cpu_recent_mean = float(
        np.mean(cpu_recent)
    )

    cpu_previous_mean = float(
        np.mean(cpu_previous)
    )

    memory_recent_mean = float(
        np.mean(memory_recent)
    )

    memory_previous_mean = float(
        np.mean(memory_previous)
    )

    return {
        "history_cpu_mean": cpu_hist["mean"],
        "history_cpu_std": cpu_hist["std"],
        "history_cpu_min": cpu_hist["minimum"],
        "history_cpu_max": cpu_hist["maximum"],
        "history_cpu_first": cpu_hist["first"],
        "history_cpu_last": cpu_hist["last"],
        "history_cpu_change": cpu_hist["change"],

        "history_memory_mean": memory_hist["mean"],
        "history_memory_std": memory_hist["std"],
        "history_memory_min": memory_hist["minimum"],
        "history_memory_max": memory_hist["maximum"],
        "history_memory_first": memory_hist["first"],
        "history_memory_last": memory_hist["last"],
        "history_memory_change": memory_hist["change"],

        "recent_cpu_mean": cpu_recent_mean,
        "previous_cpu_mean": cpu_previous_mean,
        "recent_cpu_change": (
            cpu_recent_mean - cpu_previous_mean
        ),

        "recent_memory_mean": memory_recent_mean,
        "previous_memory_mean": memory_previous_mean,
        "recent_memory_change": (
            memory_recent_mean - memory_previous_mean
        ),

        "future_cpu_mean": float(
            np.mean(future_cpu)
        ),
        "future_cpu_max": float(
            np.max(future_cpu)
        ),
        "future_cpu_min": float(
            np.min(future_cpu)
        ),

        "future_memory_mean": float(
            np.mean(future_memory)
        ),
        "future_memory_max": float(
            np.max(future_memory)
        ),
        "future_memory_min": float(
            np.min(future_memory)
        ),

        "future_cpu_high": bool(
            np.any(future_cpu >= 85)
        ),
        "future_memory_high": bool(
            np.any(future_memory >= 85)
        ),

        "future_cpu_low": bool(
            np.all(future_cpu <= 35)
        ),
        "future_memory_low": bool(
            np.all(future_memory <= 35)
        ),
    }


# =========================================================
# BUILD WINDOW-LEVEL ANALYSIS
# =========================================================

def build_window_analysis(
    predictions,
    machine_sequences,
):

    records = []

    for row in predictions.to_dict(
        orient="records"
    ):

        features = extract_window_features(
            machine_sequences,
            row,
        )

        if features is None:
            continue

        record = {
            "machineID": row["machineID"],
            "history_start_step": row["history_start_step"],
            "history_end_step": row["history_end_step"],
            "target_end_step": row["target_end_step"],

            "actual_label": row["target_original"],
            "actual_high": row["actual_high"],

            "original_probability": row[
                "original_probability"
            ],

            "platt_probability": row[
                "platt_probability"
            ],

            "original_alert": row[
                "original_alert"
            ],

            "platt_alert": row[
                "platt_alert"
            ],

            "original_false_positive": row[
                "original_false_positive"
            ],

            "platt_false_positive": row[
                "platt_false_positive"
            ],
        }

        record.update(features)

        records.append(record)

    result = pd.DataFrame(records)

    if len(result) != len(predictions):
        raise ValueError(
            "Some prediction windows could not be reconstructed."
        )

    return result


# =========================================================
# FALSE-POSITIVE GROUP COMPARISON
# =========================================================

def summarize_false_positives(
    windows,
    false_positive_column,
    model_name,
):

    selected = windows[
        windows[false_positive_column]
    ].copy()

    if selected.empty:
        return {
            "model": model_name,
            "false_positive_windows": 0,
        }

    metrics = [
        "history_cpu_mean",
        "history_memory_mean",
        "recent_cpu_change",
        "recent_memory_change",
        "future_cpu_mean",
        "future_cpu_max",
        "future_memory_mean",
        "future_memory_max",
    ]

    summary = {
        "model": model_name,
        "false_positive_windows": len(selected),
        "unique_machines": int(
            selected["machineID"].nunique()
        ),
    }

    for column in metrics:

        summary[f"{column}_mean"] = float(
            selected[column].mean()
        )

        summary[f"{column}_median"] = float(
            selected[column].median()
        )

    summary["memory_declining_windows"] = int(
        (
            selected["recent_memory_change"] < 0
        ).sum()
    )

    summary["cpu_declining_windows"] = int(
        (
            selected["recent_cpu_change"] < 0
        ).sum()
    )

    summary["future_memory_high_windows"] = int(
        selected["future_memory_high"].sum()
    )

    summary["future_cpu_high_windows"] = int(
        selected["future_cpu_high"].sum()
    )

    return summary


# =========================================================
# MACHINE-LEVEL SUMMARY
# =========================================================

def machine_summary(windows):

    rows = []

    for machine_id, group in windows.groupby(
        "machineID"
    ):

        original_fp = group[
            group["original_false_positive"]
        ]

        platt_fp = group[
            group["platt_false_positive"]
        ]

        if original_fp.empty and platt_fp.empty:
            continue

        combined = pd.concat(
            [
                original_fp,
                platt_fp,
            ]
        ).drop_duplicates(
            subset=KEYS
        )

        rows.append({
            "machineID": machine_id,

            "original_false_positive_windows": len(
                original_fp
            ),

            "platt_false_positive_windows": len(
                platt_fp
            ),

            "historical_cpu_mean": float(
                combined["history_cpu_mean"].mean()
            ),

            "historical_memory_mean": float(
                combined["history_memory_mean"].mean()
            ),

            "recent_cpu_change_mean": float(
                combined["recent_cpu_change"].mean()
            ),

            "recent_memory_change_mean": float(
                combined["recent_memory_change"].mean()
            ),

            "future_cpu_peak_mean": float(
                combined["future_cpu_max"].mean()
            ),

            "future_memory_peak_mean": float(
                combined["future_memory_max"].mean()
            ),

            "memory_declining_windows": int(
                (
                    combined["recent_memory_change"] < 0
                ).sum()
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "original_false_positive_windows",
        ascending=False,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading raw dataset and predictions...")

    raw, original, platt = load_data()

    print("Matching prediction windows...")

    predictions = merge_predictions(
        original,
        platt,
    )

    print(
        f"Matched prediction windows: {len(predictions):,}"
    )

    print("Preparing machine-level time sequences...")

    machine_sequences = prepare_machine_sequences(
        raw
    )

    print(
        f"Machines loaded: {len(machine_sequences):,}"
    )

    print("Reconstructing historical and future windows...")

    windows = build_window_analysis(
        predictions,
        machine_sequences,
    )

    print(
        f"Reconstructed windows: {len(windows):,}"
    )

    # Full window-level output.
    windows.to_csv(
        OUTPUT_DIR / "machine_window_analysis.csv",
        index=False,
    )

    # False-positive subsets.
    original_fp = windows[
        windows["original_false_positive"]
    ].copy()

    platt_fp = windows[
        windows["platt_false_positive"]
    ].copy()

    original_fp.to_csv(
        OUTPUT_DIR / "original_false_positive_windows.csv",
        index=False,
    )

    platt_fp.to_csv(
        OUTPUT_DIR / "platt_false_positive_windows.csv",
        index=False,
    )

    # False-positive summaries.
    summaries = [
        summarize_false_positives(
            windows,
            "original_false_positive",
            "Original LSTM",
        ),

        summarize_false_positives(
            windows,
            "platt_false_positive",
            "Platt LSTM",
        ),
    ]

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUTPUT_DIR / "false_positive_feature_summary.csv",
        index=False,
    )

    # Machine-level patterns.
    machines = machine_summary(
        windows
    )

    machines.to_csv(
        OUTPUT_DIR / "machine_level_patterns.csv",
        index=False,
    )

    # Compare false-positive windows with actual high windows.
    comparison_rows = []

    groups = [
        (
            "Original False Positives",
            windows["original_false_positive"],
        ),
        (
            "Platt False Positives",
            windows["platt_false_positive"],
        ),
        (
            "Actual High Demand",
            windows["actual_high"],
        ),
        (
            "Neither Model False Positive",
            ~(
                windows["original_false_positive"]
                | windows["platt_false_positive"]
            ),
        ),
    ]

    feature_columns = [
        "history_cpu_mean",
        "history_memory_mean",
        "recent_cpu_change",
        "recent_memory_change",
        "future_cpu_max",
        "future_memory_max",
    ]

    for group_name, mask in groups:

        selected = windows[mask]

        for feature in feature_columns:

            comparison_rows.append({
                "group": group_name,
                "feature": feature,
                "count": len(selected),
                "mean": (
                    float(selected[feature].mean())
                    if len(selected) else None
                ),
                "median": (
                    float(selected[feature].median())
                    if len(selected) else None
                ),
            })

    feature_comparison = pd.DataFrame(
        comparison_rows
    )

    feature_comparison.to_csv(
        OUTPUT_DIR / "feature_group_comparison.csv",
        index=False,
    )

    # Final summary.
    summary = {
        "matched_windows": len(windows),
        "machines": len(machine_sequences),

        "original_false_positive_windows": len(
            original_fp
        ),

        "platt_false_positive_windows": len(
            platt_fp
        ),

        "original_false_positive_machines": int(
            original_fp["machineID"].nunique()
        ),

        "platt_false_positive_machines": int(
            platt_fp["machineID"].nunique()
        ),

        "original_false_positive_summary": summaries[0],
        "platt_false_positive_summary": summaries[1],

        "interpretation_notes": [
            "Historical windows contain 12 observations.",
            "Future windows contain 6 observations.",
            "Each observation represents 5 minutes.",
            "Recent trend compares the last 6 historical rows "
            "against the first 6 historical rows.",
            "Future peaks are used only for retrospective analysis.",
            "They are not model input features.",
            "Overlapping windows are not independent incidents.",
        ],
    }

    with open(
        OUTPUT_DIR / "machine_pattern_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # Console output.
    print("\n" + "=" * 75)
    print("PLATT MACHINE-LEVEL PATTERN ANALYSIS")
    print("=" * 75)

    print("\nFALSE-POSITIVE FEATURE SUMMARY")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nTOP MACHINE-LEVEL PATTERNS")

    print(
        machines.head(15).to_string(
            index=False
        )
    )

    print("\nFEATURE GROUP COMPARISON")

    print(
        feature_comparison.to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
