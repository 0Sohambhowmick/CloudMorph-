
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "datasets/processed/server_usage_clean.csv"
PREDICTION_PATH = (
    ROOT / "ml/models/extratrees_selected_features/test_predictions.csv"
)
FEATURE_PATH = (
    ROOT / "ml/models/temporal_features_experiment/temporal_features_dataset.csv"
)
OUTPUT_DIR = ROOT / "ml/models/early_warning_diagnostic"

OUTPUT_PATH = OUTPUT_DIR / "early_warning_cases.csv"
SUMMARY_PATH = OUTPUT_DIR / "early_warning_summary.csv"

REQUIRED_RAW = [
    "machineID", "step", "timestamp", "util_CPU", "util_memory"
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    raw = pd.read_csv(RAW_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)
    features = pd.read_csv(FEATURE_PATH)

    if not set(REQUIRED_RAW).issubset(raw.columns):
        raise ValueError("Raw dataset is missing required columns.")

    # Join prediction rows to their historical engineered features.
    merged = predictions.merge(
    features[
        [
            "machineID",
            "history_start_step",
            "util_memory_max",
        ]
    ],
    on=["machineID", "history_start_step"],
    how="left",
    validate="one_to_one",
)

    if merged["util_memory_max"].isna().any():
        raise ValueError("Some predictions could not be matched to features.")

    # Focus on actual high-demand cases where historical memory max < 85%.
    cases = merged[
        (merged["actual_workload"] == "HIGH_DEMAND")
        & (merged["risk_alert"] == "NO_HIGH_RISK")
        & (merged["util_memory_max"] < 85)
    ].copy()

    print(f"Selected missed cases: {len(cases)}")

    if len(cases) == 0:
        raise ValueError("No cases matched the diagnostic criteria.")

    raw = raw.sort_values(["machineID", "step"]).copy()
    raw_groups = {
        int(machine_id): group.set_index("step")
        for machine_id, group in raw.groupby("machineID", sort=False)
    }

    records = []

    for _, case in cases.iterrows():
        machine_id = int(case["machineID"])
        history_end = int(case["history_end_step"])
        target_start = int(case["target_start_step"])
        target_end = int(case["target_end_step"])

        if machine_id not in raw_groups:
            raise ValueError(f"Machine {machine_id} not found in raw data.")

        machine = raw_groups[machine_id]

        history = machine.loc[
            (machine.index >= history_end - 11)
            & (machine.index <= history_end)
        ]
        future = machine.loc[
            (machine.index >= target_start)
            & (machine.index <= target_end)
        ]

        if len(history) != 12 or len(future) != 6:
            raise ValueError(
                f"Unexpected window size for machine {machine_id}, "
                f"history start {case['history_start_step']}: "
                f"history={len(history)}, future={len(future)}"
            )

        future_cpu = future["util_CPU"].to_numpy(dtype=float)
        future_memory = future["util_memory"].to_numpy(dtype=float)

        cpu_crossings = np.where(future_cpu >= 85)[0]
        memory_crossings = np.where(future_memory >= 85)[0]

        first_cpu = int(cpu_crossings[0]) if len(cpu_crossings) else None
        first_memory = (
            int(memory_crossings[0]) if len(memory_crossings) else None
        )

        crossing_candidates = [
            value for value in [first_cpu, first_memory]
            if value is not None
        ]

        first_crossing = (
            min(crossing_candidates) if crossing_candidates else None
        )

        if first_crossing is None:
            driver = "NO_85_CROSSING_FOUND"
            warning_minutes = None
        else:
            cpu_at_first = future_cpu[first_crossing] >= 85
            memory_at_first = future_memory[first_crossing] >= 85

            if cpu_at_first and memory_at_first:
                driver = "CPU_AND_MEMORY"
            elif cpu_at_first:
                driver = "CPU"
            else:
                driver = "MEMORY"

            # Each observation is 5 minutes after the previous one.
            warning_minutes = (first_crossing + 1) * 5

        recent_memory = history["util_memory"].to_numpy(dtype=float)
        recent_cpu = history["util_CPU"].to_numpy(dtype=float)

        records.append({
            "machineID": machine_id,
            "history_start_step": int(case["history_start_step"]),
            "history_end_step": history_end,
            "target_start_step": target_start,
            "target_end_step": target_end,
            "historical_memory_max": float(case["util_memory_max"]),
            "historical_memory_last": float(recent_memory[-1]),
            "historical_memory_mean": float(np.mean(recent_memory)),
            "historical_cpu_max": float(history["util_CPU"].max()),
            "historical_cpu_last": float(recent_cpu[-1]),
            "future_memory_max": float(np.max(future_memory)),
            "future_cpu_max": float(np.max(future_cpu)),
            "first_cpu_crossing_offset": (
                first_cpu + 1 if first_cpu is not None else None
            ),
            "first_memory_crossing_offset": (
                first_memory + 1 if first_memory is not None else None
            ),
            "first_crossing_offset": (
                first_crossing + 1 if first_crossing is not None else None
            ),
            "warning_minutes": warning_minutes,
            "spike_driver": driver,
            "memory_change_first_to_last": float(
                future_memory[-1] - recent_memory[-1]
            ),
            "cpu_change_first_to_last": float(
                future_cpu[-1] - recent_cpu[-1]
            ),
            "risk_probability": float(
                case["high_demand_probability"]
            ),
        })

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_PATH, index=False)

    summary = (
        result.groupby("spike_driver", dropna=False)
        .agg(
            cases=("machineID", "size"),
            average_historical_memory=("historical_memory_mean", "mean"),
            average_future_memory_max=("future_memory_max", "mean"),
            average_future_cpu_max=("future_cpu_max", "mean"),
            average_warning_minutes=("warning_minutes", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nSPIKE DRIVER SUMMARY")
    print(summary.to_string(index=False))

    print("\nWARNING TIME DISTRIBUTION")
    print(
        result["warning_minutes"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nDIAGNOSTIC OUTPUTS")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
