"""
CloudMorph Cost and Resource Optimization Engine.

This module evaluates application utilization and explicitly supplied cost
estimates. It does not invent provider prices, resize live resources, or claim
that estimates are verified by AWS/GCP/Azure.

Backward compatibility:
- Keeps analyze_application(application).
- Preserves the existing top-level response keys and recommendation shape.
- Keeps calculate_savings(current_cost, optimized_cost).
"""

from datetime import datetime, timezone
import math


# ============================================================
# CONFIGURATION
# ============================================================

UNDERUTILIZATION_THRESHOLD = 35.0
HIGH_UTILIZATION_THRESHOLD = 85.0

ENGINE_NAME = "CloudMorph Cost Optimization Engine"
ENGINE_VERSION = "2.0.0"

DEFAULT_CURRENCY = "USD"
SUPPORTED_COST_PERIODS = {"monthly", "annual"}


# ============================================================
# VALIDATION HELPERS
# ============================================================

def parse_optional_number(value, field_name):
    """Convert a value into a finite float. Missing values remain None."""

    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a valid number.")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")

    return number


def validate_utilization(value):
    """Validate utilization as a percentage from 0 to 100."""

    utilization = parse_optional_number(value, "utilization")

    if utilization is None:
        return None

    if not 0 <= utilization <= 100:
        raise ValueError("Utilization must be between 0 and 100.")

    return utilization


def validate_cost(value, field_name):
    """Validate a non-negative cost value."""

    cost = parse_optional_number(value, field_name)

    if cost is None:
        return None

    if cost < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return round(cost, 4)


def normalize_cost_period(value):
    """Normalize the supplied cost period to monthly or annual."""

    period = str(value or "monthly").strip().lower()

    aliases = {
        "month": "monthly",
        "monthly": "monthly",
        "year": "annual",
        "yearly": "annual",
        "annual": "annual",
        "annually": "annual",
    }

    if period not in aliases:
        raise ValueError("cost_period must be 'monthly' or 'annual'.")

    return aliases[period]


def normalize_currency(value):
    """Return a simple ISO-like currency label without assuming conversion."""

    currency = str(value or DEFAULT_CURRENCY).strip().upper()

    if not currency or len(currency) != 3 or not currency.isalpha():
        raise ValueError("currency must be a three-letter currency code.")

    return currency


def to_monthly_cost(value, period):
    """Convert a supplied monthly or annual amount into a monthly amount."""

    if value is None:
        return None

    return value / 12.0 if period == "annual" else value


# ============================================================
# RESOURCE RECOMMENDATION
# ============================================================

def generate_recommendation(utilization):
    """
    Generate a simple advisory recommendation using application utilization.

    Rules:
        <= 35% : RESIZE
        >= 85% : SCALE_UP
        35-85% : KEEP
        Missing: INVESTIGATE

    The forecast-aware ML policy in resource_optimizer.py remains the
    authoritative workload decision for analyze-workload requests.
    """

    if utilization is None:
        return {
            "action": "INVESTIGATE",
            "priority": "MEDIUM",
            "title": "Insufficient utilization data",
            "reasons": [
                "Utilization data is not available.",
                "Collect resource metrics before making an optimization decision.",
            ],
        }

    if utilization <= UNDERUTILIZATION_THRESHOLD:
        return {
            "action": "RESIZE",
            "priority": "MEDIUM",
            "title": "Resource may be underutilized",
            "reasons": [
                f"Current utilization is {utilization:.2f}%.",
                (
                    "Utilization is at or below the "
                    f"{UNDERUTILIZATION_THRESHOLD:.0f}% underutilization threshold."
                ),
                (
                    "Consider a smaller resource configuration only after "
                    "checking workload, availability and performance requirements."
                ),
            ],
        }

    if utilization >= HIGH_UTILIZATION_THRESHOLD:
        return {
            "action": "SCALE_UP",
            "priority": "HIGH",
            "title": "Resource utilization is high",
            "reasons": [
                f"Current utilization is {utilization:.2f}%.",
                (
                    "Utilization has reached or exceeded the "
                    f"{HIGH_UTILIZATION_THRESHOLD:.0f}% high-utilization threshold."
                ),
                (
                    "Consider additional capacity after checking workload "
                    "performance and the cost impact."
                ),
            ],
        }

    return {
        "action": "KEEP",
        "priority": "LOW",
        "title": "Resource utilization is within the configured range",
        "reasons": [
            f"Current utilization is {utilization:.2f}%.",
            "No immediate resizing recommendation is indicated.",
            "Continue monitoring resource utilization.",
        ],
    }


# ============================================================
# COST CALCULATION
# ============================================================

def calculate_savings(current_cost, optimized_cost):
    """
    Compare two explicitly supplied monthly cost values.

    No optimized price is generated by this function.
    """

    unavailable = {
        "estimated_monthly_savings": None,
        "estimated_annual_savings": None,
        "estimated_savings_percentage": None,
        "estimated_monthly_cost_increase": None,
        "estimated_cost_increase_percentage": None,
        "savings_status": "insufficient_data",
        "cost_efficiency_score": None,
    }

    if (
        current_cost is None
        or optimized_cost is None
        or current_cost <= 0
        or optimized_cost < 0
    ):
        return unavailable.copy()

    if optimized_cost < current_cost:
        monthly_savings = current_cost - optimized_cost
        annual_savings = monthly_savings * 12
        savings_percentage = (monthly_savings / current_cost) * 100

        return {
            "estimated_monthly_savings": round(monthly_savings, 2),
            "estimated_annual_savings": round(annual_savings, 2),
            "estimated_savings_percentage": round(savings_percentage, 2),
            "estimated_monthly_cost_increase": 0.0,
            "estimated_cost_increase_percentage": 0.0,
            "savings_status": "savings_available",
            "cost_efficiency_score": round(
                max(0.0, min(100.0, savings_percentage)), 2
            ),
        }

    if optimized_cost == current_cost:
        return {
            "estimated_monthly_savings": 0.0,
            "estimated_annual_savings": 0.0,
            "estimated_savings_percentage": 0.0,
            "estimated_monthly_cost_increase": 0.0,
            "estimated_cost_increase_percentage": 0.0,
            "savings_status": "no_savings",
            "cost_efficiency_score": 0.0,
        }

    monthly_increase = optimized_cost - current_cost
    increase_percentage = (monthly_increase / current_cost) * 100

    return {
        "estimated_monthly_savings": 0.0,
        "estimated_annual_savings": 0.0,
        "estimated_savings_percentage": 0.0,
        "estimated_monthly_cost_increase": round(monthly_increase, 2),
        "estimated_cost_increase_percentage": round(increase_percentage, 2),
        "savings_status": "cost_increase",
        "cost_efficiency_score": 0.0,
    }


def build_cost_recommendation(cost_analysis):
    """Create a separate cost recommendation without overriding workload action."""

    status = cost_analysis["savings_status"]

    if status == "savings_available":
        percentage = cost_analysis["estimated_savings_percentage"]
        return {
            "action": "REVIEW_SAVINGS",
            "priority": "MEDIUM" if percentage < 20 else "HIGH",
            "title": "Lower supplied cost estimate is available",
            "reasons": [
                (
                    f"The supplied optimized estimate is "
                    f"{percentage:.2f}% lower than the current estimate."
                ),
                "Validate the proposed configuration and service requirements before adoption.",
            ],
        }

    if status == "cost_increase":
        increase = cost_analysis["estimated_cost_increase_percentage"]
        return {
            "action": "REVIEW_COST_INCREASE",
            "priority": "HIGH",
            "title": "Proposed configuration has a higher estimated cost",
            "reasons": [
                f"The supplied optimized estimate is {increase:.2f}% higher.",
                "Review the capacity and performance benefit before accepting the higher cost.",
            ],
        }

    if status == "no_savings":
        return {
            "action": "NO_COST_CHANGE",
            "priority": "LOW",
            "title": "No estimated cost reduction",
            "reasons": [
                "The supplied current and optimized monthly costs are equal.",
                "Continue monitoring usage and review alternative configurations when valid pricing data is available.",
            ],
        }

    return {
        "action": "INSUFFICIENT_COST_DATA",
        "priority": "MEDIUM",
        "title": "More cost information is required",
        "reasons": [
            "A valid current cost and an explicit optimized cost are required to calculate savings.",
            "CloudMorph will not invent an optimized price or savings percentage.",
        ],
    }


def build_cost_comparison(current_monthly_cost, optimized_monthly_cost, currency):
    """Build a consistent current-versus-proposed cost comparison."""

    return {
        "currency": currency,
        "billing_period": "monthly",
        "current": {
            "monthly": (
                round(current_monthly_cost, 2)
                if current_monthly_cost is not None else None
            ),
            "annual": (
                round(current_monthly_cost * 12, 2)
                if current_monthly_cost is not None else None
            ),
        },
        "proposed": {
            "monthly": (
                round(optimized_monthly_cost, 2)
                if optimized_monthly_cost is not None else None
            ),
            "annual": (
                round(optimized_monthly_cost * 12, 2)
                if optimized_monthly_cost is not None else None
            ),
        },
    }


# ============================================================
# MAIN OPTIMIZATION ENGINE
# ============================================================

def analyze_application(application):
    """
    Analyze an application and return resource and cost recommendations.

    Expected fields:
        id, name, provider, service, region, utilization,
        cost, optimized_cost

    Optional fields:
        cost_period: monthly or annual (defaults to monthly)
        currency: three-letter code (defaults to USD)
        cost_data_source: descriptive source label
        pricing_verified: boolean explicitly supplied by the caller

    Cost values are estimates unless the caller explicitly marks the
    source as verified. This module does not call provider pricing APIs.
    """

    if not isinstance(application, dict):
        raise ValueError("Application must be a dictionary.")

    application_id = application.get("id")
    application_name = application.get("name", "Unnamed application")

    if not isinstance(application_name, str):
        application_name = "Unnamed application"

    provider = application.get("provider", "UNKNOWN")
    service = application.get("service", "")
    region = application.get("region", "")

    utilization = validate_utilization(application.get("utilization"))

    cost_period = normalize_cost_period(application.get("cost_period"))
    currency = normalize_currency(application.get("currency"))

    supplied_current_cost = validate_cost(application.get("cost"), "cost")
    supplied_optimized_cost = validate_cost(
        application.get("optimized_cost"), "optimized_cost"
    )

    current_monthly_cost = to_monthly_cost(
        supplied_current_cost, cost_period
    )
    optimized_monthly_cost = to_monthly_cost(
        supplied_optimized_cost, cost_period
    )

    recommendation = generate_recommendation(utilization)
    cost_metrics = calculate_savings(
        current_monthly_cost, optimized_monthly_cost
    )
    cost_recommendation = build_cost_recommendation(cost_metrics)

    if cost_metrics["savings_status"] == "savings_available":
        recommendation["reasons"].append(
            "The supplied optimized cost estimate indicates "
            f"{cost_metrics['estimated_savings_percentage']:.2f}% estimated savings."
        )
    elif cost_metrics["savings_status"] == "no_savings":
        recommendation["reasons"].append(
            "The supplied optimized cost estimate equals the current cost; no savings are estimated."
        )
    elif cost_metrics["savings_status"] == "cost_increase":
        recommendation["reasons"].append(
            "The supplied optimized cost estimate is higher than the current cost; review the proposed configuration."
        )
    else:
        recommendation["reasons"].append(
            "Cost efficiency could not be evaluated because complete cost information is unavailable."
        )

    cost_data_source = str(
        application.get("cost_data_source") or "application_record"
    )
    pricing_verified = application.get("pricing_verified") is True

    cost_analysis = {
        "current_monthly_cost": (
            round(current_monthly_cost, 2)
            if current_monthly_cost is not None else None
        ),
        "optimized_monthly_cost": (
            round(optimized_monthly_cost, 2)
            if optimized_monthly_cost is not None else None
        ),
        **cost_metrics,
        "cost_comparison": build_cost_comparison(
            current_monthly_cost,
            optimized_monthly_cost,
            currency,
        ),
        "cost_recommendation": cost_recommendation,
        "currency": currency,
        "original_cost_period": cost_period,
        "cost_data_source": cost_data_source,
        "pricing_verified": pricing_verified,
    }

    return {
        "status": "success",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "application": {
            "id": application_id,
            "name": application_name,
            "provider": provider,
            "service": service,
            "region": region,
        },
        "resource_analysis": {
            "utilization": utilization,
            "utilization_unit": "percentage",
            "underutilization_threshold": UNDERUTILIZATION_THRESHOLD,
            "high_utilization_threshold": HIGH_UTILIZATION_THRESHOLD,
        },
        "recommendation": {
            "action": recommendation["action"],
            "priority": recommendation["priority"],
            "title": recommendation["title"],
            "reasons": recommendation["reasons"],
        },
        "cost_analysis": cost_analysis,
        "analysis_metadata": {
            "analysis_type": "rule_based_resource_and_cost_analysis",
            "cost_data_source": cost_data_source,
            "cost_estimates_are_provider_verified": pricing_verified,
            "provider_pricing_api_used": False,
            "requires_manual_validation": True,
            "automatic_resource_changes": False,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
