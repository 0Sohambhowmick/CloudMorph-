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

TARGET_NAMES = [
    "cpu_utilization",
    "memory_utilization",
]

FORECAST_HORIZONS = {
    "15min": 3,
    "30min": 6,
}

MODEL_NAME = "Persistence"

MODEL_STATUS = "baseline"


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_history(history):

    try:
        history = np.asarray(
            history,
            dtype=np.float32,
        )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "History must contain numeric values."
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
            "History contains missing or non-finite values."
        )

    # CPU, memory and disk utilization must be percentages.
    utilization = history[:, :3]

    if np.any(utilization < 0) or np.any(utilization > 100):

        raise ValueError(
            "CPU, memory and disk utilization must be "
            "between 0 and 100."
        )

    # Load values cannot be negative.
    load_values = history[:, 3:]

    if np.any(load_values < 0):

        raise ValueError(
            "Load values cannot be negative."
        )

    return history


# ============================================================
# FORECASTING
# ============================================================

def forecast_utilization(history):

    """
    Generate CPU and memory utilization forecasts.

    Input:
        history: 12 observations with 6 features each.

    Feature order:
        util_CPU,
        util_memory,
        util_disk,
        load1,
        load5,
        load15

    Output:
        15-minute and 30-minute forecasts.

    Method:
        Persistence baseline.
    """

    history = validate_history(history)

    # Latest available observation.
    latest_observation = history[-1]

    cpu_prediction = float(
        latest_observation[0]
    )

    memory_prediction = float(
        latest_observation[1]
    )

    predictions = {}

    for horizon in FORECAST_HORIZONS:

        predictions[horizon] = {
            "cpu_utilization": cpu_prediction,
            "memory_utilization": memory_prediction,
        }

    return {
        "status": "success",

        "forecast_method": MODEL_NAME,

        "model_status": MODEL_STATUS,

        "input_window": WINDOW_SIZE,

        "input_features": FEATURE_NAMES,

        "forecast_unit": "percentage",

        "forecasts": predictions,
    }