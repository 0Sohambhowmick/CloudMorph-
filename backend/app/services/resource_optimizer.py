
"""
CloudMorph forecast-aware resource optimizer with ML workload classification.

The classifier is advisory. Observed/forecast pressure and conservative
resizing checks remain the safety gate. This module never changes cloud resources.
"""

import logging
import math

from ml.inference.optimization_classifier import predict_workload_condition


OPTIMIZER_NAME = "CloudMorph Resource Optimizer"
OPTIMIZER_VERSION = "2.1.0"
UNDERUTILIZATION_THRESHOLD = 35.0
HIGH_UTILIZATION_THRESHOLD = 85.0
MAX_RESIZING_VARIABILITY = 30.0
SUPPORTED_HORIZONS = ["15min", "30min"]
ML_MIN_CONFIDENCE = 0.60

logger = logging.getLogger("CloudMorphResourceOptimizer")


def validate_number(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")

    return number


def validate_percentage(value, field_name):
    number = validate_number(value, field_name)

    if not 0 <= number <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100.")

    return number


def validate_workload_analysis(analysis):
    """Validate workload_analyzer.py output and normalize required values."""

    if not isinstance(analysis, dict) or analysis.get("status") != "success":
        raise ValueError("A successful workload analysis is required.")

    statistics = analysis.get("resource_statistics")
    trends = analysis.get("workload_trends")
    variability = analysis.get("workload_variability")
    forecasts = analysis.get("forecasts")

    if not all(
        isinstance(x, dict)
        for x in (statistics, trends, variability, forecasts)
    ):
        raise ValueError("Workload analysis is missing required sections.")

    validated = {
        "statistics": {},
        "trends": {},
        "variability": {},
        "forecasts": {},
    }

    for resource in ("cpu", "memory"):
        stats = statistics.get(resource)
        trend = trends.get(resource)
        var = variability.get(resource)

        if not isinstance(stats, dict) or not isinstance(trend, dict) or not isinstance(var, dict):
            raise ValueError(f"Missing {resource} workload analysis details.")

        validated["statistics"][resource] = {
            field: validate_percentage(
                stats.get(field),
                f"{resource} {field}",
            )
            for field in ("current", "average", "peak")
        }

        direction = trend.get("direction")

        if direction not in ("INCREASING", "DECREASING", "STABLE"):
            raise ValueError(f"Invalid {resource} workload trend.")

        validated["trends"][resource] = direction

        cv = var.get("coefficient_of_variation")

        if cv is not None:
            cv = validate_number(
                cv,
                f"{resource} coefficient of variation",
            )

            if cv < 0:
                raise ValueError("Coefficient of variation cannot be negative.")

        validated["variability"][resource] = cv

    for horizon in SUPPORTED_HORIZONS:
        forecast = forecasts.get(horizon)

        if not isinstance(forecast, dict):
            raise ValueError(f"Missing {horizon} forecast.")

        validated["forecasts"][horizon] = {
            "cpu": validate_percentage(
                forecast.get("cpu_utilization"),
                f"{horizon} CPU forecast",
            ),
            "memory": validate_percentage(
                forecast.get("memory_utilization"),
                f"{horizon} memory forecast",
            ),
        }

    return validated


def analyze_resource_pressure(data):
    pressure = {}

    for resource in ("cpu", "memory"):
        current = data["statistics"][resource]["current"]
        peak = data["statistics"][resource]["peak"]

        maximum_forecast = max(
            data["forecasts"][h][resource]
            for h in SUPPORTED_HORIZONS
        )

        maximum_demand = max(
            current,
            peak,
            maximum_forecast,
        )

        pressure[resource] = {
            "current_utilization": round(current, 2),
            "peak_utilization": round(peak, 2),
            "maximum_forecast": round(maximum_forecast, 2),
            "maximum_observed_or_forecast": round(maximum_demand, 2),
            "high_pressure": maximum_demand >= HIGH_UTILIZATION_THRESHOLD,
        }

    return pressure


def generate_optimization_decision(data, pressure):
    """Original deterministic policy, retained as the inference fallback."""

    reasons = []

    high_resources = [
        r.upper()
        for r in ("cpu", "memory")
        if pressure[r]["high_pressure"]
    ]

    if high_resources:
        for resource in high_resources:
            value = pressure[resource.lower()]["maximum_observed_or_forecast"]

            reasons.append(
                f"{resource} demand reached {value:.2f}% across observed and forecast values."
            )

        reasons.append(
            "Consider increasing resource capacity after validating workload performance and available capacity."
        )

        return {
            "action": "SCALE_UP",
            "priority": "HIGH",
            "title": "High resource demand detected",
            "reasons": reasons,
            "affected_resources": high_resources,
        }

    resize_eligible = True

    for resource in ("cpu", "memory"):
        stats = data["statistics"][resource]
        trend = data["trends"][resource]
        cv = data["variability"][resource]

        forecast_values = [
            data["forecasts"][h][resource]
            for h in SUPPORTED_HORIZONS
        ]

        if any(
            v > UNDERUTILIZATION_THRESHOLD
            for v in [
                stats["current"],
                stats["average"],
                stats["peak"],
                *forecast_values,
            ]
        ):
            resize_eligible = False
            reasons.append(
                f"{resource.upper()} utilization does not meet the low-demand resizing condition."
            )

        if trend == "INCREASING":
            resize_eligible = False
            reasons.append(
                f"{resource.upper()} workload is increasing; resource reduction is not recommended."
            )

        if cv is None or cv > MAX_RESIZING_VARIABILITY:
            resize_eligible = False
            reasons.append(
                f"{resource.upper()} workload variability is too high or unavailable for safe resizing."
            )

    if resize_eligible:
        return {
            "action": "RESIZE",
            "priority": "MEDIUM",
            "title": "Resources appear consistently underutilized",
            "reasons": [
                f"CPU and memory utilization remain at or below {UNDERUTILIZATION_THRESHOLD:.0f}%.",
                "Both workloads are stable or decreasing within the analyzed observation window.",
                "Observed workload variability is within the configured resizing limit.",
                "Consider a smaller resource configuration after validating performance requirements.",
            ],
            "affected_resources": ["CPU", "MEMORY"],
        }

    if not reasons:
        reasons.append(
            "Current and forecasted resource demand does not indicate an immediate capacity issue."
        )

    reasons.append(
        "Maintain the current configuration and continue monitoring workload patterns."
    )

    return {
        "action": "KEEP",
        "priority": "LOW",
        "title": "Current resource allocation is acceptable",
        "reasons": reasons,
        "affected_resources": [],
    }


def _history_to_records(history):
    """Convert the route's 12x6 NumPy history into classifier input records."""

    if history is None:
        raise ValueError(
            "The 12-observation workload history is required for ML classification."
        )

    columns = [
        "util_CPU",
        "util_memory",
        "util_disk",
        "load1",
        "load5",
        "load15",
    ]

    if hasattr(history, "tolist"):
        rows = history.tolist()
    else:
        rows = history

    if len(rows) != 12:
        raise ValueError(
            f"Expected exactly 12 history observations; received {len(rows)}."
        )

    records = []

    for row in rows:
        if isinstance(row, dict):
            records.append(row)
        else:
            if len(row) != 6:
                raise ValueError(
                    "Each workload observation must contain exactly six metrics."
                )

            records.append({
                name: float(value)
                for name, value in zip(columns, row)
            })

    return records


def _safe_resize_eligible(data):
    for resource in ("cpu", "memory"):
        stats = data["statistics"][resource]

        if any(
            v > UNDERUTILIZATION_THRESHOLD
            for v in [
                stats["current"],
                stats["average"],
                stats["peak"],
                *[
                    data["forecasts"][h][resource]
                    for h in SUPPORTED_HORIZONS
                ],
            ]
        ):
            return False

        if data["trends"][resource] == "INCREASING":
            return False

        cv = data["variability"][resource]

        if cv is None or cv > MAX_RESIZING_VARIABILITY:
            return False

    return True


def _get_memory_risk_details(memory_risk_analysis):
    """
    Safely read the experimental memory-risk classifier output.

    Invalid or unavailable predictions are ignored.
    """

    if not isinstance(memory_risk_analysis, dict):
        return False, None

    if memory_risk_analysis.get("status") != "success":
        return False, None

    high_memory_risk = memory_risk_analysis.get("high_memory_risk")

    if not isinstance(high_memory_risk, bool):
        return False, None

    probability_percent = memory_risk_analysis.get(
        "risk_probability_percent"
    )

    try:
        if probability_percent is not None:
            probability_percent = validate_number(
                probability_percent,
                "Memory risk probability percentage",
            )

            if not 0 <= probability_percent <= 100:
                return False, None

        else:
            probability = memory_risk_analysis.get("risk_probability")

            if probability is not None:
                probability = validate_number(
                    probability,
                    "Memory risk probability",
                )

                if not 0 <= probability <= 1:
                    return False, None

                probability_percent = probability * 100

    except ValueError:
        return False, None

    return high_memory_risk, probability_percent


def _apply_memory_risk_safety_override(
    decision,
    memory_risk_analysis,
):
    """
    Prevent resource reduction when the experimental memory-risk model
    predicts high future memory demand.

    Existing SCALE_UP safety decisions always take precedence.
    """

    high_memory_risk, risk_probability_percent = (
        _get_memory_risk_details(memory_risk_analysis)
    )

    if not high_memory_risk:
        return decision, False

    # Never suppress an existing high-pressure SCALE_UP recommendation.
    if decision.get("action") == "SCALE_UP":
        return decision, False

    reasons = [
        "The experimental ML memory-risk classifier predicts a high risk of future memory demand.",
        "Resource reduction is blocked because predicted memory pressure may affect workload stability.",
        "Maintain the current resource configuration and monitor memory utilization.",
    ]

    if risk_probability_percent is not None:
        reasons.insert(
            1,
            f"Predicted memory-risk probability: {risk_probability_percent:.2f}%.",
        )

    updated_decision = {
        "action": "KEEP",
        "priority": "HIGH",
        "title": "High future memory risk detected; maintain resources",
        "reasons": reasons,
        "affected_resources": ["MEMORY"],
    }

    return updated_decision, True


def _ml_guided_decision(data, pressure, prediction):
    """Apply ML condition labels without allowing them to bypass safety gates."""

    # Hard safety override: observed or forecast pressure always wins.
    safe_scale_decision = generate_optimization_decision(data, pressure)

    if safe_scale_decision["action"] == "SCALE_UP":
        safe_scale_decision["reasons"].insert(
            0,
            "Safety override: observed/forecast utilization reached the high-pressure threshold; ML cannot suppress this warning.",
        )

        return safe_scale_decision, "forecast_safety_override"

    condition = prediction["predicted_condition"]

    if condition == "HIGH_DEMAND":
        return {
            "action": "KEEP",
            "priority": "MEDIUM",
            "title": "Potential high demand predicted; monitor before changing capacity",
            "reasons": [
                "The ML classifier predicts HIGH_DEMAND within its 30-minute prediction horizon.",
                "Current observed and numeric forecast values have not crossed the configured high-pressure threshold.",
                "Capacity increase is not recommended automatically from the classifier label alone; validate workload metrics and monitor closely.",
            ],
            "affected_resources": [],
        }, "ml_high_demand_monitoring"

    if condition == "LOW_DEMAND":
        if _safe_resize_eligible(data):
            return {
                "action": "RESIZE",
                "priority": "MEDIUM",
                "title": "ML predicts low demand and safety checks permit resizing",
                "reasons": [
                    "The ML classifier predicts LOW_DEMAND within its 30-minute prediction horizon.",
                    f"CPU and memory current, average, peak and forecast utilization are at or below {UNDERUTILIZATION_THRESHOLD:.0f}%.",
                    "Neither workload trend is increasing, and both variability checks are within the configured limit.",
                    "Consider a smaller resource configuration only after validating performance requirements.",
                ],
                "affected_resources": ["CPU", "MEMORY"],
            }, "ml_low_demand_with_safety_checks"

        return {
            "action": "KEEP",
            "priority": "LOW",
            "title": "Low demand predicted, but resizing safety checks were not satisfied",
            "reasons": [
                "The ML classifier predicts LOW_DEMAND, but one or more conservative utilization, trend or variability checks prevent resizing.",
                "Maintain the current configuration and continue monitoring workload patterns.",
            ],
            "affected_resources": [],
        }, "ml_low_demand_blocked_by_safety_checks"

    return {
        "action": "KEEP",
        "priority": "LOW",
        "title": "Normal workload demand predicted",
        "reasons": [
            "The ML classifier predicts NORMAL_DEMAND within its 30-minute prediction horizon.",
            "No observed or numeric forecast value has crossed the high-pressure threshold, and no safe resize has been established by the ML-guided policy.",
            "Maintain the current configuration and continue monitoring workload patterns.",
        ],
        "affected_resources": [],
    }, "ml_normal_demand"


def optimize_resources(
    workload_analysis,
    history=None,
    memory_risk_analysis=None,
):
    """Create an advisory ML-informed recommendation with deterministic fallback."""

    data = validate_workload_analysis(workload_analysis)
    pressure = analyze_resource_pressure(data)

    ml_analysis = {
        "status": "unavailable",
        "prediction": None,
        "minimum_confidence_required": ML_MIN_CONFIDENCE,
        "fallback_reason": None,
    }

    try:
        records = _history_to_records(history)

        prediction = predict_workload_condition(records)

        confidence = validate_number(
            prediction.get("confidence"),
            "ML confidence",
        )

        if not 0 <= confidence <= 1:
            raise ValueError("ML confidence must be between 0 and 1.")

        if prediction.get("predicted_condition") not in (
            "LOW_DEMAND",
            "NORMAL_DEMAND",
            "HIGH_DEMAND",
        ):
            raise ValueError(
                "Classifier returned an unsupported workload condition."
            )

        ml_analysis.update({
            "status": "success",
            "prediction": prediction,
        })

        if confidence < ML_MIN_CONFIDENCE:
            ml_analysis["status"] = "low_confidence"

            ml_analysis["fallback_reason"] = (
                f"Classifier confidence {confidence:.4f} is below the configured "
                f"{ML_MIN_CONFIDENCE:.2f} threshold."
            )

            decision = generate_optimization_decision(data, pressure)

            decision_source = (
                "forecast_aware_policy_fallback_low_ml_confidence"
            )

            decision_engine = (
                "forecast_aware_optimization_policy_fallback"
            )

        else:
            decision, decision_source = _ml_guided_decision(
                data,
                pressure,
                prediction,
            )

            decision_engine = (
                "ml_classifier_with_forecast_safety_policy"
            )

    except Exception as exc:
        # A classifier/model issue must not take down the existing advisory flow.
        logger.exception(
            "Optimization classifier failed; using deterministic policy."
        )

        ml_analysis["status"] = "error"
        ml_analysis["fallback_reason"] = str(exc)

        decision = generate_optimization_decision(data, pressure)

        decision_source = (
            "forecast_aware_policy_fallback_ml_error"
        )

        decision_engine = (
            "forecast_aware_optimization_policy_fallback"
        )

    # Apply memory-risk safety after the existing decision.
    # This also protects the deterministic fallback from unsafe resizing.
    decision, memory_risk_model_used = (
        _apply_memory_risk_safety_override(
            decision,
            memory_risk_analysis,
        )
    )

    if memory_risk_model_used:
        decision_source = "ml_memory_high_risk_monitoring"

    memory_risk_available = (
        isinstance(memory_risk_analysis, dict)
        and memory_risk_analysis.get("status") == "success"
        and isinstance(
            memory_risk_analysis.get("high_memory_risk"),
            bool,
        )
    )

    return {
        "status": "success",
        "optimizer": OPTIMIZER_NAME,
        "optimizer_version": OPTIMIZER_VERSION,
        "decision_engine": decision_engine,
        "decision_source": decision_source,
        "recommendation": decision,
        "resource_pressure": pressure,
        "ml_analysis": ml_analysis,

        "decision_factors": {
            "underutilization_threshold": UNDERUTILIZATION_THRESHOLD,
            "high_utilization_threshold": HIGH_UTILIZATION_THRESHOLD,
            "maximum_resizing_variability": MAX_RESIZING_VARIABILITY,
            "ml_minimum_confidence": ML_MIN_CONFIDENCE,
            "cpu_trend": data["trends"]["cpu"],
            "memory_trend": data["trends"]["memory"],
            "forecast_horizons": SUPPORTED_HORIZONS,
        },

        "analysis_metadata": {
            "requires_manual_validation": True,
            "automatic_resource_changes": False,
            "provider_pricing_included": False,

            "ml_classifier_used": (
                ml_analysis["status"] in ("success", "low_confidence")
            ),

            "ml_fallback_used": (
                decision_engine
                == "forecast_aware_optimization_policy_fallback"
            ),

            "memory_risk_model_available": memory_risk_available,
            "memory_risk_model_used": memory_risk_model_used,
        },
    }
