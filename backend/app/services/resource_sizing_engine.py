
import math


# ============================================================
# CONFIGURATION
# ============================================================

ENGINE_NAME = "CloudMorph Rule-Based Resource Sizing Engine"
ENGINE_VERSION = "1.0.0"

HIGH_UTILIZATION_THRESHOLD = 85.0
LOW_UTILIZATION_THRESHOLD = 35.0

MAX_ACCEPTABLE_VARIABILITY = 30.0

SCALE_UP_TARGET_UTILIZATION = 70.0
DOWNSIZE_TARGET_UTILIZATION = 60.0

MIN_SCALE_UP_PERCENTAGE = 10
MAX_SCALE_UP_PERCENTAGE = 100

MAX_DOWNSIZE_PERCENTAGE = 30

ROUNDING_STEP = 5


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_number(value, field_name):
    """Validate a finite numeric value."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")

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

    return number


def validate_utilization(value, field_name):
    """Validate utilization between 0 and 100."""

    number = validate_number(value, field_name)

    if not 0 <= number <= 100:
        raise ValueError(
            f"{field_name} must be between 0 and 100."
        )

    return number


def round_up_to_step(value):
    """Round upward to the nearest configured percentage step."""

    return int(
        math.ceil(value / ROUNDING_STEP) * ROUNDING_STEP
    )


def round_down_to_step(value):
    """Round downward to the nearest configured percentage step."""

    return int(
        math.floor(value / ROUNDING_STEP) * ROUNDING_STEP
    )


# ============================================================
# INPUT EXTRACTION
# ============================================================

def extract_resource_data(workload_analysis, resource):
    """
    Extract CPU or memory information from workload_analyzer output.
    """

    if not isinstance(workload_analysis, dict):
        raise ValueError(
            "Workload analysis must be a dictionary."
        )

    if workload_analysis.get("status") != "success":
        raise ValueError(
            "A successful workload analysis is required."
        )

    statistics = workload_analysis.get(
        "resource_statistics", {}
    )

    trends = workload_analysis.get(
        "workload_trends", {}
    )

    variability = workload_analysis.get(
        "workload_variability", {}
    )

    forecasts = workload_analysis.get(
        "forecasts", {}
    )

    resource_stats = statistics.get(resource)
    resource_trend = trends.get(resource)
    resource_variability = variability.get(resource)

    if not isinstance(resource_stats, dict):
        raise ValueError(
            f"Missing statistics for {resource}."
        )

    if not isinstance(resource_trend, dict):
        raise ValueError(
            f"Missing trend information for {resource}."
        )

    if not isinstance(resource_variability, dict):
        raise ValueError(
            f"Missing variability information for {resource}."
        )

    current = validate_utilization(
        resource_stats.get("current"),
        f"{resource}.current",
    )

    average = validate_utilization(
        resource_stats.get("average"),
        f"{resource}.average",
    )

    peak = validate_utilization(
        resource_stats.get("peak"),
        f"{resource}.peak",
    )

    trend_direction = resource_trend.get("direction")

    if trend_direction not in [
        "INCREASING",
        "DECREASING",
        "STABLE",
    ]:
        raise ValueError(
            f"Invalid trend direction for {resource}."
        )

    coefficient = resource_variability.get(
        "coefficient_of_variation"
    )

    if coefficient is not None:
        coefficient = validate_number(
            coefficient,
            f"{resource}.coefficient_of_variation",
        )

        if coefficient < 0:
            raise ValueError(
                "Coefficient of variation cannot be negative."
            )

    if not isinstance(forecasts, dict):
        raise ValueError(
            "Forecast information is missing."
        )

    forecast_values = []

    for horizon in ["15min", "30min"]:

        prediction = forecasts.get(horizon)

        if not isinstance(prediction, dict):
            raise ValueError(
                f"Missing {horizon} forecast."
            )

        forecast_value = validate_utilization(
            prediction.get(
                f"{resource}_utilization"
            ),
            f"{resource}.{horizon}_forecast",
        )

        forecast_values.append({
            "horizon": horizon,
            "utilization": forecast_value,
        })

    return {
        "current": current,
        "average": average,
        "peak": peak,
        "trend": trend_direction,
        "coefficient_of_variation": coefficient,
        "forecasts": forecast_values,
    }


# ============================================================
# CAPACITY ADJUSTMENT CALCULATION
# ============================================================

def calculate_scale_up_percentage(peak_utilization):
    """
    Estimate additional capacity required to bring
    projected utilization towards the target utilization.

    This is a mathematical capacity estimate, not
    a provider-specific instance recommendation.
    """

    required_capacity_factor = (
        peak_utilization / SCALE_UP_TARGET_UTILIZATION
    )

    increase_percentage = (
        required_capacity_factor - 1
    ) * 100

    increase_percentage = max(
        MIN_SCALE_UP_PERCENTAGE,
        increase_percentage,
    )

    increase_percentage = min(
        MAX_SCALE_UP_PERCENTAGE,
        increase_percentage,
    )

    return round_up_to_step(
        increase_percentage
    )


def calculate_downsize_percentage(peak_utilization):
    """
    Estimate a conservative capacity reduction.

    Downsize estimates are capped at 30 percent.
    """

    if peak_utilization <= 0:
        return 0

    reduction_percentage = (
        1 - (
            peak_utilization /
            DOWNSIZE_TARGET_UTILIZATION
        )
    ) * 100

    reduction_percentage = max(
        0,
        reduction_percentage,
    )

    reduction_percentage = min(
        MAX_DOWNSIZE_PERCENTAGE,
        reduction_percentage,
    )

    return round_down_to_step(
        reduction_percentage
    )


# ============================================================
# INDIVIDUAL RESOURCE ANALYSIS
# ============================================================

def analyze_resource_sizing(workload_analysis, resource):
    """
    Generate a rule-based sizing recommendation
    for CPU or memory.
    """

    data = extract_resource_data(
        workload_analysis,
        resource,
    )

    current = data["current"]
    average = data["average"]
    historical_peak = data["peak"]

    trend = data["trend"]

    variability = data[
        "coefficient_of_variation"
    ]

    forecast_values = [
        item["utilization"]
        for item in data["forecasts"]
    ]

    projected_peak = max(
        [current, historical_peak] + forecast_values
    )

    reasons = []

    # --------------------------------------------------------
    # RULE 1: HIGH RESOURCE PRESSURE
    # --------------------------------------------------------

    if projected_peak >= HIGH_UTILIZATION_THRESHOLD:

        action = "SCALE_UP"

        adjustment = calculate_scale_up_percentage(
            projected_peak
        )

        reasons.append(
            f"{resource.upper()} projected utilization "
            f"reaches {projected_peak:.2f}%, "
            "crossing the high-pressure threshold."
        )

        reasons.append(
            "Additional capacity is recommended "
            "to restore utilization headroom."
        )

        if trend == "INCREASING":
            reasons.append(
                "Recent workload trend is increasing."
            )

    # --------------------------------------------------------
    # RULE 2: CONSERVATIVE DOWNSIZING
    # --------------------------------------------------------

    elif (
        historical_peak <= LOW_UTILIZATION_THRESHOLD
        and current <= LOW_UTILIZATION_THRESHOLD
        and average <= LOW_UTILIZATION_THRESHOLD
        and all(
            value <= LOW_UTILIZATION_THRESHOLD
            for value in forecast_values
        )
        and trend != "INCREASING"
        and variability is not None
        and variability <= MAX_ACCEPTABLE_VARIABILITY
    ):

        action = "RESIZE_DOWN"

        adjustment = calculate_downsize_percentage(
            historical_peak
        )

        if adjustment == 0:
            action = "KEEP"
            reasons.append(
                "Utilization is low, but the estimated "
                "capacity reduction is negligible."
            )
        else:
            reasons.append(
                "Current, average, peak and forecast "
                "utilization remain within the low-pressure range."
            )

            reasons.append(
                "Workload trend and variability satisfy "
                "the conservative downsizing conditions."
            )

    # --------------------------------------------------------
    # RULE 3: MAINTAIN CURRENT CAPACITY
    # --------------------------------------------------------

    else:

        action = "KEEP"
        adjustment = 0

        reasons.append(
            "Resource utilization does not satisfy "
            "the conditions for safe scaling or downsizing."
        )

        if trend == "INCREASING":
            reasons.append(
                "Increasing workload trend prevents downsizing."
            )

        if (
            variability is None
            or variability > MAX_ACCEPTABLE_VARIABILITY
        ):
            reasons.append(
                "Workload variability is unavailable or "
                "exceeds the safe downsizing threshold."
            )

    return {
        "resource": resource.upper(),
        "action": action,
        "estimated_capacity_adjustment_percentage": adjustment,
        "utilization": {
            "current": round(current, 2),
            "average": round(average, 2),
            "historical_peak": round(historical_peak, 2),
            "projected_peak": round(projected_peak, 2),
        },
        "trend": trend,
        "coefficient_of_variation": variability,
        "forecasts": data["forecasts"],
        "reasons": reasons,
    }


# ============================================================
# COMBINED RESOURCE SIZING ENGINE
# ============================================================

def optimize_resource_capacity(workload_analysis):
    """
    Analyze CPU and memory independently and
    generate a combined capacity recommendation.

    This engine is rule-based and does not modify
    actual cloud resources.
    """

    cpu_result = analyze_resource_sizing(
        workload_analysis,
        "cpu",
    )

    memory_result = analyze_resource_sizing(
        workload_analysis,
        "memory",
    )

    actions = [
        cpu_result["action"],
        memory_result["action"],
    ]

    if "SCALE_UP" in actions:
        overall_action = "SCALE_UP"

    elif "RESIZE_DOWN" in actions:
        overall_action = "RESIZE_DOWN"

    else:
        overall_action = "KEEP"

    reasons = (
        cpu_result["reasons"]
        + memory_result["reasons"]
    )

    return {
        "status": "success",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "decision_type": "RULE_BASED_RESOURCE_SIZING",
        "overall_action": overall_action,
        "resource_recommendations": {
            "cpu": cpu_result,
            "memory": memory_result,
        },
        "reasons": reasons,
        "safety": {
            "automatic_resource_changes": False,
            "provider_instance_matching": False,
            "provider_pricing_used": False,
            "recommendations_require_review": True,
        },
        "thresholds": {
            "high_utilization": HIGH_UTILIZATION_THRESHOLD,
            "low_utilization": LOW_UTILIZATION_THRESHOLD,
            "maximum_variability_for_downsizing": (
                MAX_ACCEPTABLE_VARIABILITY
            ),
            "scale_up_target_utilization": (
                SCALE_UP_TARGET_UTILIZATION
            ),
            "downsize_target_utilization": (
                DOWNSIZE_TARGET_UTILIZATION
            ),
            "maximum_downsize_percentage": (
                MAX_DOWNSIZE_PERCENTAGE
            ),
        },
    }
