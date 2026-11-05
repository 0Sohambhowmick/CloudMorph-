
import math
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 12

FEATURE_NAMES = [
    "util_CPU",
    "util_memory",
    "util_disk",
    "load1",
    "load5",
    "load15",
]

UTILIZATION_FEATURES = {
    "cpu": 0,
    "memory": 1,
    "disk": 2,
}

FORECAST_HORIZONS = ["15min", "30min"]

TREND_TOLERANCE = 1.0

ANALYZER_NAME = "CloudMorph Workload Analyzer"
ANALYZER_VERSION = "1.0.0"


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_history(history):
    """
    Validate the 12 historical observations.

    Expected shape:
        (12, 6)

    Feature order:
        CPU, Memory, Disk, Load1, Load5, Load15
    """

    try:
        history = np.asarray(
            history,
            dtype=np.float64,
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Historical data must contain numeric values."
        ) from exc

    expected_shape = (
        WINDOW_SIZE,
        len(FEATURE_NAMES),
    )

    if history.shape != expected_shape:
        raise ValueError(
            f"Expected history shape {expected_shape}, "
            f"received {history.shape}."
        )

    if not np.isfinite(history).all():
        raise ValueError(
            "Historical data contains missing or non-finite values."
        )

    utilization = history[:, :3]

    if np.any(utilization < 0) or np.any(utilization > 100):
        raise ValueError(
            "CPU, memory and disk utilization must be "
            "between 0 and 100."
        )

    load_values = history[:, 3:]

    if np.any(load_values < 0):
        raise ValueError(
            "Load values cannot be negative."
        )

    return history


def validate_forecast(forecast_result):
    """
    Validate the output returned by forecast_utilization().
    """

    if not isinstance(forecast_result, dict):
        raise ValueError(
            "Forecast result must be a dictionary."
        )

    if forecast_result.get("status") != "success":
        raise ValueError(
            "A successful forecasting result is required."
        )

    forecasts = forecast_result.get("forecasts")

    if not isinstance(forecasts, dict):
        raise ValueError(
            "Forecast result does not contain valid forecasts."
        )

    validated = {}

    for horizon in FORECAST_HORIZONS:

        prediction = forecasts.get(horizon)

        if not isinstance(prediction, dict):
            raise ValueError(
                f"Missing forecast for {horizon}."
            )

        cpu = prediction.get("cpu_utilization")
        memory = prediction.get("memory_utilization")

        for field_name, value in [
            ("CPU forecast", cpu),
            ("Memory forecast", memory),
        ]:

            if isinstance(value, bool):
                raise ValueError(
                    f"{field_name} must be numeric."
                )

            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{field_name} must be numeric."
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    f"{field_name} must be finite."
                )

            if not 0 <= number <= 100:
                raise ValueError(
                    f"{field_name} must be between 0 and 100."
                )

            if field_name == "CPU forecast":
                cpu = number
            else:
                memory = number

        validated[horizon] = {
            "cpu_utilization": cpu,
            "memory_utilization": memory,
        }

    return validated


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def calculate_resource_statistics(values):
    """
    Calculate statistics for one resource utilization series.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    current = float(values[-1])

    average = float(np.mean(values))

    peak = float(np.max(values))

    minimum = float(np.min(values))

    standard_deviation = float(
        np.std(values)
    )

    return {
        "current": round(current, 2),
        "average": round(average, 2),
        "peak": round(peak, 2),
        "minimum": round(minimum, 2),
        "maximum": round(peak, 2),
        "standard_deviation": round(
            standard_deviation, 2
        ),
    }


# ============================================================
# WORKLOAD TREND ANALYSIS
# ============================================================

def analyze_trend(values):
    """
    Compare the latest three observations with
    the preceding three observations.

    Trend tolerance is expressed in percentage points.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    previous_average = float(
        np.mean(values[-6:-3])
    )

    recent_average = float(
        np.mean(values[-3:])
    )

    change = recent_average - previous_average

    if change > TREND_TOLERANCE:
        direction = "INCREASING"

    elif change < -TREND_TOLERANCE:
        direction = "DECREASING"

    else:
        direction = "STABLE"

    return {
        "direction": direction,
        "previous_average": round(
            previous_average, 2
        ),
        "recent_average": round(
            recent_average, 2
        ),
        "change_percentage_points": round(
            change, 2
        ),
    }


# ============================================================
# VARIABILITY ANALYSIS
# ============================================================

def analyze_variability(values):
    """
    Measure workload variability using standard deviation.

    This is a descriptive statistical indicator,
    not a prediction of future volatility.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    standard_deviation = float(
        np.std(values)
    )

    return {
        "standard_deviation": round(
            standard_deviation, 2
        ),
        "coefficient_of_variation": round(
            (
                standard_deviation /
                float(np.mean(values))
            ) * 100,
            2,
        ) if np.mean(values) > 0 else None,
    }


# ============================================================
# MAIN WORKLOAD ANALYZER
# ============================================================

def analyze_workload(history, forecast_result):
    """
    Analyze historical resource utilization and forecasts.

    Inputs:
        history:
            12 observations containing six features.

        forecast_result:
            Output from ml.inference.forecast_utilization().

    Returns:
        Current resource statistics,
        workload trends,
        variability,
        and forecast information.

    This module does not generate final
    optimization decisions.
    """

    history = validate_history(history)

    forecasts = validate_forecast(
        forecast_result
    )

    cpu_values = history[:, UTILIZATION_FEATURES["cpu"]]

    memory_values = history[:, UTILIZATION_FEATURES["memory"]]

    disk_values = history[:, UTILIZATION_FEATURES["disk"]]

    load1_values = history[:, 3]

    load5_values = history[:, 4]

    load15_values = history[:, 5]

    # --------------------------------------------------------
    # RESOURCE STATISTICS
    # --------------------------------------------------------

    cpu_statistics = calculate_resource_statistics(
        cpu_values
    )

    memory_statistics = calculate_resource_statistics(
        memory_values
    )

    disk_statistics = calculate_resource_statistics(
        disk_values
    )

    # --------------------------------------------------------
    # WORKLOAD TRENDS
    # --------------------------------------------------------

    cpu_trend = analyze_trend(
        cpu_values
    )

    memory_trend = analyze_trend(
        memory_values
    )

    # --------------------------------------------------------
    # WORKLOAD VARIABILITY
    # --------------------------------------------------------

    cpu_variability = analyze_variability(
        cpu_values
    )

    memory_variability = analyze_variability(
        memory_values
    )

    # --------------------------------------------------------
    # FORECAST COMPARISON
    # --------------------------------------------------------

    latest_cpu = cpu_statistics["current"]

    latest_memory = memory_statistics["current"]

    forecast_analysis = {}

    for horizon, prediction in forecasts.items():

        forecast_analysis[horizon] = {
            "cpu_utilization": round(
                prediction["cpu_utilization"], 2
            ),
            "memory_utilization": round(
                prediction["memory_utilization"], 2
            ),
            "cpu_change_from_current": round(
                prediction["cpu_utilization"] - latest_cpu,
                2,
            ),
            "memory_change_from_current": round(
                prediction["memory_utilization"] - latest_memory,
                2,
            ),
        }

    # --------------------------------------------------------
    # FINAL ANALYSIS RESULT
    # --------------------------------------------------------

    return {
        "status": "success",

        "analyzer": ANALYZER_NAME,

        "analyzer_version": ANALYZER_VERSION,

        "analysis_type": "historical_and_forecast_workload_analysis",

        "input_metadata": {
            "observations": WINDOW_SIZE,
            "features": FEATURE_NAMES,
            "forecast_method": forecast_result.get(
                "forecast_method",
                "UNKNOWN",
            ),
        },

        "resource_statistics": {
            "cpu": cpu_statistics,
            "memory": memory_statistics,
            "disk": disk_statistics,
        },

        "workload_trends": {
            "cpu": cpu_trend,
            "memory": memory_trend,
        },

        "workload_variability": {
            "cpu": cpu_variability,
            "memory": memory_variability,
        },

        "load_statistics": {
            "load1_average": round(
                float(np.mean(load1_values)), 4
            ),
            "load5_average": round(
                float(np.mean(load5_values)), 4
            ),
            "load15_average": round(
                float(np.mean(load15_values)), 4
            ),
        },

        "forecasts": forecast_analysis,

        "analysis_metadata": {
            "trend_window": 3,
            "trend_comparison_window": 3,
            "trend_tolerance": TREND_TOLERANCE,
            "automatic_resource_changes": False,
        },
    }
