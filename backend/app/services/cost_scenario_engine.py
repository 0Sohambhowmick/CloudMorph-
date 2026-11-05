"""
CloudMorph Cost Scenario Engine.

Evaluates proposed resource configurations using explicitly supplied costs,
complete user-supplied unit rates, or verified provider retail-price results.
It never invents prices or modifies cloud resources.
"""

import math
from datetime import datetime, timezone


ENGINE_NAME = "CloudMorph Cost Scenario Engine"
ENGINE_VERSION = "1.1.0"

SUPPORTED_CURRENCIES = {
    "USD", "INR", "EUR", "GBP", "AUD", "CAD", "SGD", "JPY"
}


def _number(value, field, required=False):
    if value is None:
        if required:
            raise ValueError(f"{field} is required.")
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    if result < 0:
        raise ValueError(f"{field} cannot be negative.")
    return result


def _positive_resource(value, field):
    result = _number(value, field, required=True)
    if result <= 0:
        raise ValueError(f"{field} must be greater than zero.")
    return result


def _normalize_configuration(configuration, field_name):
    if configuration is None:
        return None
    if not isinstance(configuration, dict):
        raise ValueError(f"{field_name} must be an object.")

    normalized = {
        "provider": configuration.get("provider"),
        "machine_type": configuration.get("machine_type"),
        "sku": configuration.get("sku") or configuration.get("machine_type"),
        "os_type": configuration.get("os_type", "linux"),
        "vcpu": _number(configuration.get("vcpu"), f"{field_name}.vcpu"),
        "memory_gb": _number(
            configuration.get("memory_gb"),
            f"{field_name}.memory_gb",
        ),
        "storage_gb": _number(
            configuration.get("storage_gb"),
            f"{field_name}.storage_gb",
        ),
        "region": configuration.get("region"),
    }

    for key in ("vcpu", "memory_gb", "storage_gb"):
        if normalized[key] is not None and normalized[key] <= 0:
            raise ValueError(f"{field_name}.{key} must be greater than zero.")

    return normalized


def _monthly_cost_from_pricing(configuration, pricing):
    """Calculate only when all applicable unit rates are explicitly supplied."""
    if not isinstance(configuration, dict) or not isinstance(pricing, dict):
        return None

    vcpu = configuration.get("vcpu")
    memory_gb = configuration.get("memory_gb")
    storage_gb = configuration.get("storage_gb")

    cpu_rate = _number(pricing.get("monthly_per_vcpu"), "pricing.monthly_per_vcpu")
    memory_rate = _number(
        pricing.get("monthly_per_gb_memory"),
        "pricing.monthly_per_gb_memory",
    )
    storage_rate = _number(
        pricing.get("monthly_per_gb_storage"),
        "pricing.monthly_per_gb_storage",
    )
    fixed_monthly = _number(
        pricing.get("monthly_fixed_cost", 0),
        "pricing.monthly_fixed_cost",
    )

    if vcpu is not None and cpu_rate is None:
        return None
    if memory_gb is not None and memory_rate is None:
        return None
    if storage_gb is not None and storage_rate is None:
        return None

    total = fixed_monthly or 0.0
    if vcpu is not None:
        total += vcpu * cpu_rate
    if memory_gb is not None:
        total += memory_gb * memory_rate
    if storage_gb is not None:
        total += storage_gb * storage_rate
    return round(total, 2)


def _safety_assessment(safety_context, current_cost, proposed_cost):
    context = safety_context if isinstance(safety_context, dict) else {}
    action = str(context.get("action") or "").upper()
    resource_pressure = context.get("resource_pressure") or {}

    high_pressure = False
    for resource in ("cpu", "memory"):
        details = resource_pressure.get(resource)
        if isinstance(details, dict) and details.get("high_pressure") is True:
            high_pressure = True

    memory_risk = context.get("high_memory_risk") is True

    if proposed_cost is None or current_cost is None:
        status = "NOT_ASSESSABLE"
        reasons = ["Comparable current and proposed monthly costs are required for cost comparison."]
    elif proposed_cost >= current_cost:
        status = "NO_COST_REDUCTION"
        reasons = ["The proposed configuration does not reduce the supplied monthly cost."]
    elif action == "SCALE_UP" or high_pressure:
        status = "BLOCKED_HIGH_RESOURCE_PRESSURE"
        reasons = [
            "The latest workload analysis indicates high resource pressure.",
            "A lower-cost configuration must not be treated as safe while scale-up pressure exists.",
        ]
    elif memory_risk:
        status = "BLOCKED_MEMORY_RISK"
        reasons = [
            "The latest memory-risk analysis indicates high future memory demand.",
            "Resource reduction is not supported by the available safety evidence.",
        ]
    elif action == "RESIZE":
        status = "RESIZE_SUPPORTED_REQUIRES_MANUAL_VALIDATION"
        reasons = [
            "The latest resource optimizer returned RESIZE.",
            "This is advisory evidence only; validate service performance and configuration limits before applying changes.",
        ]
    elif action == "SWITCH":
        status = "SWITCH_REQUIRES_COMPATIBILITY_AND_MIGRATION_VALIDATION"
        reasons = [
            "The latest resource optimizer returned SWITCH.",
            "The alternative provider or VM must be checked for workload, runtime, storage, networking, security, and regional compatibility.",
            "Migration downtime, data transfer, operational effort, and rollback planning have not been assessed by this cost estimate.",
            "Public VM compute pricing alone is not sufficient to approve a cloud migration.",
        ]
    elif action == "KEEP":
        status = "NOT_APPROVED_BY_WORKLOAD_OPTIMIZER"
        reasons = [
            "The latest resource optimizer returned KEEP, not RESIZE.",
            "A lower-cost scenario can be evaluated, but the workload optimizer has not approved resource reduction.",
        ]
    else:
        status = "NOT_VERIFIED"
        reasons = [
            "No usable workload safety decision is available.",
            "The proposed configuration requires independent performance and capacity validation.",
        ]

    return {
        "status": status,
        "resource_optimizer_action": action or None,
        "high_resource_pressure": high_pressure,
        "high_memory_risk": memory_risk,
        "manual_validation_required": True,
        "automatic_resource_changes": False,
        "decision_policy": {
            "KEEP": "Preserve the current configuration unless a new workload analysis changes the decision.",
            "RESIZE": "Treat as advisory; validate target capacity and workload performance before applying.",
            "SWITCH": "Require compatibility, migration, security, data-transfer, downtime, and rollback assessment before migration.",
        }.get(action, "Manual validation is required before any infrastructure change."),
        "reasons": reasons,
    }


def evaluate_cost_scenario(
    application,
    proposed_configuration=None,
    proposed_monthly_cost=None,
    pricing=None,
    currency=None,
    scenario_name=None,
    safety_context=None,
    provider_pricing_result=None,
):
    """
    Evaluate one proposed configuration against an application's current cost.

    Provider retail pricing is accepted only when the provider adapter reports
    status=success and pricing_verified=True. Provider VM retail estimates cover
    compute only; savings are not calculated against an unknown/full application
    bill unless the current application cost is explicitly marked compute-only.
    """
    if not isinstance(application, dict):
        raise ValueError("Application data is required.")

    current_monthly = _number(
        application.get("cost"),
        "application.cost",
        required=True,
    )
    if current_monthly <= 0:
        raise ValueError("application.cost must be greater than zero.")

    normalized_currency = str(
        currency or application.get("currency") or "USD"
    ).strip().upper()
    if normalized_currency not in SUPPORTED_CURRENCIES:
        raise ValueError(
            "Unsupported currency. Use one of: "
            + ", ".join(sorted(SUPPORTED_CURRENCIES))
        )

    configuration = _normalize_configuration(
        proposed_configuration,
        "proposed_configuration",
    )

    explicit_proposed_cost = _number(
        proposed_monthly_cost,
        "proposed_monthly_cost",
    )

    calculated_proposed_cost = None
    pricing_used = False
    provider_pricing_used = False
    provider_pricing = (
        provider_pricing_result
        if isinstance(provider_pricing_result, dict)
        else None
    )
    provider_pricing_verified = bool(
        provider_pricing
        and provider_pricing.get("status") == "success"
        and provider_pricing.get("pricing_verified") is True
    )
    estimate_scope = "full_monthly_cost"

    if explicit_proposed_cost is not None:
        proposed_cost = explicit_proposed_cost
        cost_source = "user_supplied_proposed_monthly_cost"
    else:
        # Preserve the existing manual unit-pricing path before using
        # an external provider estimate.
        calculated_proposed_cost = _monthly_cost_from_pricing(
            configuration,
            pricing,
        )

        if calculated_proposed_cost is not None:
            proposed_cost = calculated_proposed_cost
            pricing_used = True
            cost_source = "user_supplied_unit_pricing"
        elif provider_pricing_verified:
            provider_currency = str(
                provider_pricing.get("currency") or ""
            ).upper()
            if provider_currency != normalized_currency:
                raise ValueError(
                    "Provider pricing currency does not match the scenario currency."
                )
            calculated_proposed_cost = _number(
                provider_pricing.get("monthly_compute_estimate"),
                "provider_pricing.monthly_compute_estimate",
                required=True,
            )
            if calculated_proposed_cost <= 0:
                raise ValueError(
                    "Provider monthly compute estimate must be greater than zero."
                )
            proposed_cost = calculated_proposed_cost
            provider_pricing_used = True
            cost_source = "verified_provider_public_retail_compute_estimate"
            estimate_scope = str(
                provider_pricing.get("estimate_scope") or "VM compute only"
            )
        else:
            proposed_cost = None
            cost_source = "unavailable"

    if proposed_cost is not None and proposed_cost <= 0:
        raise ValueError("Proposed monthly cost must be greater than zero.")

    # Provider retail data is VM compute-only. Do not compare it with an
    # application-wide bill unless the current cost scope is explicitly compute-only.
    current_cost_scope = str(
        application.get("cost_scope") or ""
    ).strip().lower()
    provider_comparison_comparable = (
        not provider_pricing_used
        or current_cost_scope in {"compute_only", "vm_compute_only"}
    )

    if proposed_cost is None:
        monthly_savings = None
        annual_savings = None
        savings_percent = None
        comparison_status = "INSUFFICIENT_COST_INPUT"
    elif not provider_comparison_comparable:
        monthly_savings = None
        annual_savings = None
        savings_percent = None
        comparison_status = "COST_SCOPE_NOT_COMPARABLE"
    else:
        monthly_savings = round(current_monthly - proposed_cost, 2)
        annual_savings = round(monthly_savings * 12, 2)
        savings_percent = round(
            (monthly_savings / current_monthly) * 100,
            2,
        )
        comparison_status = (
            "SAVINGS"
            if monthly_savings > 0
            else "COST_INCREASE"
            if monthly_savings < 0
            else "NO_CHANGE"
        )

    safety_cost = proposed_cost if provider_comparison_comparable else None
    safety = _safety_assessment(
        safety_context,
        current_monthly,
        safety_cost,
    )
    if provider_pricing_used and not provider_comparison_comparable:
        safety["reasons"].append(
            "The proposed provider estimate covers VM compute only, while the current application cost scope is not confirmed as compute-only. Savings and resize cost safety cannot be compared."
        )

    now = datetime.now(timezone.utc).isoformat()

    pricing_metadata = {
        "pricing_source": cost_source,
        "user_supplied_unit_pricing_used": pricing_used,
        "provider_pricing_verified": provider_pricing_verified,
        "provider_pricing_api_used": provider_pricing_used,
        "currency": normalized_currency,
        "estimate_scope": estimate_scope if provider_pricing_used else None,
        "comparison_comparable": provider_comparison_comparable,
        "current_cost_scope": current_cost_scope or "unspecified",
        "provider_pricing": provider_pricing,
    }

    return {
        "status": "success",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "scenario": {
            "name": str(scenario_name or "Untitled cost scenario").strip()[:120],
            "created_at": now,
            "currency": normalized_currency,
        },
        "application": {
            "id": str(application.get("_id") or application.get("id") or ""),
            "name": application.get("name", ""),
            "provider": application.get("provider", "UNKNOWN"),
            "service": application.get("service", ""),
            "region": application.get("region", ""),
        },
        "configuration": {
            "current": {
                "machine_type": application.get("machine_type"),
                "vcpu": _number(application.get("vcpu"), "application.vcpu"),
                "memory_gb": _number(
                    application.get("memory_gb"),
                    "application.memory_gb",
                ),
                "storage_gb": _number(
                    application.get("storage_gb"),
                    "application.storage_gb",
                ),
                "region": application.get("region"),
            },
            "proposed": configuration,
        },
        "cost_comparison": {
            "period": "monthly",
            "current_monthly_cost": round(current_monthly, 2),
            "proposed_monthly_cost": (
                round(proposed_cost, 2)
                if proposed_cost is not None
                else None
            ),
            "current_annual_cost": round(current_monthly * 12, 2),
            "proposed_annual_cost": (
                round(proposed_cost * 12, 2)
                if proposed_cost is not None
                else None
            ),
            "estimated_monthly_savings": monthly_savings,
            "estimated_annual_savings": annual_savings,
            "estimated_savings_percent": savings_percent,
            "comparison_status": comparison_status,
            "comparison_note": (
                "Provider price is VM compute-only. Savings are withheld because the current application cost is not explicitly marked compute-only."
                if provider_pricing_used and not provider_comparison_comparable
                else None
            ),
        },
        "safety_assessment": safety,
        "pricing_metadata": pricing_metadata,
        "analysis_metadata": {
            "requires_manual_validation": True,
            "automatic_resource_changes": False,
            "resource_optimizer_modified": False,
            "created_at": now,
        },
    }
