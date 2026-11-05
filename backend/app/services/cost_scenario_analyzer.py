
import math


# ============================================================
# CONFIGURATION
# ============================================================

ENGINE_NAME = "CloudMorph Cost Scenario Analyzer"
ENGINE_VERSION = "1.0.0"

SUPPORTED_CURRENCY = "USD"
SUPPORTED_BILLING_PERIOD = "monthly"


# ============================================================
# VALIDATION
# ============================================================

def validate_cost(value, field_name):
    """Validate a manually supplied cost value."""

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field_name} must be numeric."
        ) from exc

    if not math.isfinite(amount):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if amount < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return amount


# ============================================================
# COST SCENARIO ANALYSIS
# ============================================================

def analyze_cost_scenario(
    current_cost,
    optimized_cost,
    currency="USD",
    billing_period="monthly",
):
    """
    Compare manually supplied current and optimized costs.

    Both values must represent costs for the same billing period.

    No provider pricing APIs or estimated cloud prices are used.
    """

    if currency != SUPPORTED_CURRENCY:
        raise ValueError(
            "Only USD currency is currently supported."
        )

    if billing_period != SUPPORTED_BILLING_PERIOD:
        raise ValueError(
            "Only monthly billing values are currently supported."
        )

    current = validate_cost(
        current_cost,
        "Current cost",
    )

    optimized = validate_cost(
        optimized_cost,
        "Optimized cost",
    )

    monthly_difference = current - optimized

    if current == 0 and optimized == 0:
        savings_percentage = 0.0

    elif current == 0:
        savings_percentage = None

    else:
        savings_percentage = (
            monthly_difference / current
        ) * 100

    annual_current = current * 12
    annual_optimized = optimized * 12
    annual_difference = monthly_difference * 12

    if monthly_difference > 0:
        status = "SAVINGS_AVAILABLE"
        interpretation = (
            "The manually entered optimized cost "
            "is lower than the current cost."
        )

    elif monthly_difference < 0:
        status = "COST_INCREASE"
        interpretation = (
            "The manually entered optimized cost "
            "is higher than the current cost."
        )

    else:
        status = "NO_SAVINGS"
        interpretation = (
            "Current and optimized costs are equal."
        )

    return {
        "status": "success",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "currency": currency,
        "billing_period": billing_period,
        "cost_source": "MANUAL_USER_INPUT",

        "current_cost": round(current, 2),
        "optimized_cost": round(optimized, 2),

        "monthly": {
            "current_cost": round(current, 2),
            "optimized_cost": round(optimized, 2),
            "savings": round(monthly_difference, 2),
            "savings_percentage": (
                round(savings_percentage, 2)
                if savings_percentage is not None
                else None
            ),
        },

        "annual": {
            "current_cost": round(annual_current, 2),
            "optimized_cost": round(annual_optimized, 2),
            "savings": round(annual_difference, 2),
        },

        "scenario_status": status,
        "interpretation": interpretation,

        "metadata": {
            "pricing_api_used": False,
            "provider_prices_verified": False,
            "optimized_cost_automatically_generated": False,
            "annual_values_are_annualized_estimates": True,
        },
    }
