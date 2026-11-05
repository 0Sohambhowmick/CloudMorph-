
"""
CloudMorph - Workload Transition Analysis

Reconstructs actual historical-to-future CPU and memory
transitions for the previously evaluated risk cases.

Historical window: 12 observations (60 minutes)
Future window: 6 observations (30 minutes)
Sampling interval: 5 minutes

Diagnostic only. No model training or modification.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    ROOT / "datasets" / "processed"
    / "server_usage_clean.csv"
)

ERROR_CASES_PATH = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "error_analysis" / "risk_error_cases.csv"
)

OUTPUT_DIR = (
    ROOT / "ml" / "models" / "optimization_classifier"
    / "transition_analysis"
)

HISTORY_SIZE = 12
FUTURE_SIZE = 6
HIGH_THRESHOLD = 85.0

GROUPS = [
    "DETECTED_HIGH",
    "MISSED_HIGH",
    "FALSE_ALERT",
    "CORRECTLY_IGNORED",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("CloudMorphTransitions")


def load_data():

    logger.info("Loading raw workload data...")

    raw = pd.read_csv(RAW_DATA_PATH)

    required = {
        "machineID",
        "step",
        "util_CPU",
        "util_memory",
    }

    if not required.issubset(raw.columns):
        raise ValueError(
            f"Raw dataset missing columns: "
            f"{required - set(raw.columns)}"
        )

    raw = raw.sort_values(
        ["machineID", "step"]
    ).reset_index(drop=True)

    if raw.duplicated(
        ["machineID", "step"]
    ).any():
        raise ValueError("Duplicate machine/step observations.")

    logger.info("Raw observations: %s", len(raw))
    logger.info("Machines: %s", raw.machineID.nunique())

    logger.info("Loading previous risk error groups...")

    cases = pd.read_csv(ERROR_CASES_PATH)

    required_cases = {
        "machineID",
        "history_start_step",
        "actual_workload",
        "predicted_workload",
        "high_demand_probability",
        "risk_alert",
        "error_group",
    }

    if not required_cases.issubset(cases.columns):
        raise ValueError(
            f"Risk cases missing columns: "
            f"{required_cases - set(cases.columns)}"
        )

    return raw, cases


def reconstruct_transitions(raw, cases):

    logger.info("Reconstructing historical and future windows...")

    machine_groups = {
        machine_id: group.set_index("step").sort_index()
        for machine_id, group in raw.groupby("machineID")
    }

    records = []

    for _, case in cases.iterrows():

        machine_id = case["machineID"]
        start = int(case["history_start_step"])

        if machine_id not in machine_groups:
            raise ValueError(
                f"Machine {machine_id} not found."
            )

        machine = machine_groups[machine_id]

        history_steps = list(
            range(start, start + HISTORY_SIZE)
        )

        future_steps = list(
            range(
                start + HISTORY_SIZE,
                start + HISTORY_SIZE + FUTURE_SIZE,
            )
        )

        required_steps = history_steps + future_steps

        if not set(required_steps).issubset(
            set(machine.index)
        ):
            raise ValueError(
                f"Incomplete window for machine {machine_id}, "
                f"history step {start}"
            )

        history = machine.loc[history_steps]
        future = machine.loc[future_steps]

        history_cpu = history["util_CPU"].to_numpy(
            dtype=float
        )

        history_memory = history["util_memory"].to_numpy(
            dtype=float
        )

        future_cpu = future["util_CPU"].to_numpy(
            dtype=float
        )

        future_memory = future["util_memory"].to_numpy(
            dtype=float
        )

        cpu_crossings = np.where(
            future_cpu >= HIGH_THRESHOLD
        )[0]

        memory_crossings = np.where(
            future_memory >= HIGH_THRESHOLD
        )[0]

        cpu_crossing = (
            int(cpu_crossings[0])
            if len(cpu_crossings) else None
        )

        memory_crossing = (
            int(memory_crossings[0])
            if len(memory_crossings) else None
        )

        cpu_high = bool(len(cpu_crossings))
        memory_high = bool(len(memory_crossings))

        if cpu_high and memory_high:
            spike_source = "BOTH"

        elif cpu_high:
            spike_source = "CPU_ONLY"

        elif memory_high:
            spike_source = "MEMORY_ONLY"

        else:
            spike_source = "NO_HIGH_THRESHOLD_CROSSING"

        crossing_offsets = [
            value for value in [
                cpu_crossing,
                memory_crossing,
            ]
            if value is not None
        ]

        first_crossing = (
            min(crossing_offsets)
            if crossing_offsets else None
        )

        records.append({

            "machineID": machine_id,

            "history_start_step": start,

            "error_group": case["error_group"],

            "actual_workload": case["actual_workload"],

            "predicted_workload": case["predicted_workload"],

            "high_demand_probability": (
                case["high_demand_probability"]
            ),

            "risk_alert": case["risk_alert"],

            # Historical CPU

            "cpu_history_current": history_cpu[-1],

            "cpu_history_mean": history_cpu.mean(),

            "cpu_history_max": history_cpu.max(),

            "cpu_history_recent_mean": (
                history_cpu[-3:].mean()
            ),

            "cpu_history_trend": np.polyfit(
                np.arange(HISTORY_SIZE),
                history_cpu,
                1,
            )[0],

            # Historical memory

            "memory_history_current": history_memory[-1],

            "memory_history_mean": history_memory.mean(),

            "memory_history_max": history_memory.max(),

            "memory_history_recent_mean": (
                history_memory[-3:].mean()
            ),

            "memory_history_trend": np.polyfit(
                np.arange(HISTORY_SIZE),
                history_memory,
                1,
            )[0],

            # Future CPU

            "cpu_future_mean": future_cpu.mean(),

            "cpu_future_max": future_cpu.max(),

            "cpu_future_min": future_cpu.min(),

            "cpu_future_change_from_current": (
                future_cpu.max() - history_cpu[-1]
            ),

            "cpu_future_change_from_mean": (
                future_cpu.max() - history_cpu.mean()
            ),

            # Future memory

            "memory_future_mean": future_memory.mean(),

            "memory_future_max": future_memory.max(),

            "memory_future_min": future_memory.min(),

            "memory_future_change_from_current": (
                future_memory.max() - history_memory[-1]
            ),

            "memory_future_change_from_mean": (
                future_memory.max() - history_memory.mean()
            ),

            # Threshold crossings

            "cpu_crossed_85": cpu_high,

            "memory_crossed_85": memory_high,

            "spike_source": spike_source,

            "cpu_first_crossing_offset": cpu_crossing,

            "memory_first_crossing_offset": memory_crossing,

            "first_crossing_offset": first_crossing,

            "minutes_to_first_crossing": (
                (first_crossing + 1) * 5
                if first_crossing is not None else None
            ),

            # Verify derived target

            "reconstructed_high_label": (
                "HIGH_DEMAND"
                if cpu_high or memory_high
                else "NOT_HIGH_DEMAND"
            ),
        })

    result = pd.DataFrame(records)

    if len(result) != len(cases):
        raise RuntimeError("Transition reconstruction count mismatch.")

    # Validate reconstructed high-demand labels.

    actual_high = (
        result["actual_workload"] == "HIGH_DEMAND"
    )

    reconstructed_high = (
        result["reconstructed_high_label"] == "HIGH_DEMAND"
    )

    if not np.array_equal(
        actual_high.to_numpy(),
        reconstructed_high.to_numpy(),
    ):
        mismatches = result.loc[
            actual_high != reconstructed_high
        ]

        raise ValueError(
            "Reconstructed labels do not match stored labels:\n"
            f"{mismatches.head().to_string()}"
        )

    logger.info(
        "Successfully reconstructed %s windows.",
        len(result),
    )

    return result


def create_group_summary(result):

    summary = (
        result.groupby("error_group")
        .agg(
            cases=("machineID", "size"),

            mean_cpu_history=(
                "cpu_history_mean", "mean"
            ),

            mean_cpu_future_max=(
                "cpu_future_max", "mean"
            ),

            mean_memory_history=(
                "memory_history_mean", "mean"
            ),

            mean_memory_future_max=(
                "memory_future_max", "mean"
            ),

            mean_cpu_change=(
                "cpu_future_change_from_current", "mean"
            ),

            mean_memory_change=(
                "memory_future_change_from_current", "mean"
            ),

            cpu_crossings=(
                "cpu_crossed_85", "sum"
            ),

            memory_crossings=(
                "memory_crossed_85", "sum"
            ),
        )
        .reset_index()
    )

    return summary


def create_spike_summary(result):

    summary = (
        result.groupby(
            ["error_group", "spike_source"]
        )
        .size()
        .reset_index(name="cases")
    )

    return summary


def create_crossing_summary(result):

    high_cases = result[
        result["actual_workload"] == "HIGH_DEMAND"
    ].copy()

    summary = (
        high_cases.groupby(
            ["error_group", "spike_source"]
        )
        .agg(
            cases=("machineID", "size"),

            mean_first_crossing_offset=(
                "first_crossing_offset", "mean"
            ),

            mean_minutes_to_crossing=(
                "minutes_to_first_crossing", "mean"
            ),

            earliest_crossing_offset=(
                "first_crossing_offset", "min"
            ),

            latest_crossing_offset=(
                "first_crossing_offset", "max"
            ),
        )
        .reset_index()
    )

    return summary


def create_feature_comparison(result):

    features = [
        "cpu_history_current",
        "cpu_history_mean",
        "cpu_history_max",
        "cpu_history_recent_mean",
        "cpu_history_trend",
        "cpu_future_max",
        "cpu_future_change_from_current",
        "memory_history_current",
        "memory_history_mean",
        "memory_history_max",
        "memory_history_recent_mean",
        "memory_history_trend",
        "memory_future_max",
        "memory_future_change_from_current",
    ]

    rows = []

    for feature in features:

        row = {"feature": feature}

        for group in GROUPS:

            values = result.loc[
                result["error_group"] == group,
                feature,
            ]

            row[f"{group}_mean"] = (
                float(values.mean())
                if len(values) else None
            )

            row[f"{group}_median"] = (
                float(values.median())
                if len(values) else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


def create_time_summary(result):

    return (
        result.groupby(
            ["history_start_step", "error_group"]
        )
        .size()
        .reset_index(name="cases")
    )


def save_results(
    result,
    group_summary,
    spike_summary,
    crossing_summary,
    feature_comparison,
    time_summary,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "transition_cases.csv": result,
        "transition_group_summary.csv": group_summary,
        "transition_spike_summary.csv": spike_summary,
        "transition_crossing_summary.csv": crossing_summary,
        "transition_feature_comparison.csv": feature_comparison,
        "transition_time_summary.csv": time_summary,
    }

    for filename, dataframe in outputs.items():

        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

        logger.info("Saved %s", filename)


def main():

    logger.info("=" * 65)
    logger.info("CLOUDMORPH WORKLOAD TRANSITION ANALYSIS")
    logger.info("=" * 65)

    raw, cases = load_data()

    result = reconstruct_transitions(
        raw,
        cases,
    )

    group_summary = create_group_summary(result)

    spike_summary = create_spike_summary(result)

    crossing_summary = create_crossing_summary(result)

    feature_comparison = create_feature_comparison(result)

    time_summary = create_time_summary(result)

    logger.info("TRANSITION GROUP SUMMARY")

    logger.info(
        "\n%s",
        group_summary.to_string(index=False),
    )

    logger.info("SPIKE SOURCE SUMMARY")

    logger.info(
        "\n%s",
        spike_summary.to_string(index=False),
    )

    logger.info("HIGH-DEMAND CROSSING SUMMARY")

    logger.info(
        "\n%s",
        crossing_summary.to_string(index=False),
    )

    save_results(
        result,
        group_summary,
        spike_summary,
        crossing_summary,
        feature_comparison,
        time_summary,
    )

    logger.info("=" * 65)
    logger.info("WORKLOAD TRANSITION ANALYSIS COMPLETED")
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
