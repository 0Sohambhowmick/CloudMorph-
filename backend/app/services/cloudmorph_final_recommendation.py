"""
CloudMorph Final Recommendation Orchestrator.
Combines a saved forecast-aware resource recommendation with the latest
saved cost scenario. Advisory only; never changes infrastructure.
"""
from datetime import datetime, timezone

ENGINE_NAME = "CloudMorph Final Recommendation Orchestrator"
ENGINE_VERSION = "1.0.0"
RESIZE_STATUS = "RESIZE_SUPPORTED_REQUIRES_MANUAL_VALIDATION"
SWITCH_STATUS = "SWITCH_REQUIRES_COMPATIBILITY_AND_MIGRATION_VALIDATION"


def _parse_time(value):
    if not value:
        return None
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def build_final_recommendation(
    resource_recommendation,
    cost_scenario=None,
    resource_analysis_metadata=None,
    recommendation_created_at=None,
    scenario_created_at=None,
):
    """Combine current resource decision and a scenario evaluated after it."""
    resource = resource_recommendation if isinstance(resource_recommendation, dict) else {}
    metadata = resource_analysis_metadata if isinstance(resource_analysis_metadata, dict) else {}
    scenario = cost_scenario if isinstance(cost_scenario, dict) else {}
    result = scenario.get("result") if isinstance(scenario.get("result"), dict) else scenario
    safety = result.get("safety_assessment") or {}
    comparison = result.get("cost_comparison") or {}
    pricing = result.get("pricing_metadata") or {}

    action = str(resource.get("action") or "").strip().upper()
    if action not in {"KEEP", "RESIZE", "SWITCH"}:
        action = "UNKNOWN"

    rec_time = _parse_time(recommendation_created_at)
    cost_time = _parse_time(scenario_created_at or scenario.get("created_at"))
    fresh = bool(result and rec_time and cost_time and cost_time >= rec_time)
    scenario_action = str(safety.get("resource_optimizer_action") or "").upper()
    action_matches = action in {"KEEP", "RESIZE", "SWITCH"} and scenario_action == action
    usable = bool(result and fresh and action_matches)

    comparison_status = comparison.get("comparison_status")
    comparable = comparison_status not in (None, "INSUFFICIENT_COST_INPUT", "COST_SCOPE_NOT_COMPARABLE")
    savings = comparison.get("estimated_monthly_savings") if usable and comparable else None
    savings_pct = comparison.get("estimated_savings_percent") if usable and comparable else None

    reasons, validation = [], []
    if action == "KEEP":
        final_action, status = "KEEP", "KEEP_CURRENT_CONFIGURATION"
        title = "Maintain the current resource configuration"
        reasons += [
            "The latest forecast-aware resource optimizer returned KEEP.",
            "A lower-cost scenario cannot override the workload optimizer's KEEP decision.",
        ]
        if savings is not None and savings > 0:
            reasons.append("A lower cost estimate exists, but it is potential savings only and is not approved as a safe resource change.")
        validation.append("Continue monitoring workload utilization and rerun analysis when workload conditions change.")
    elif action == "RESIZE":
        if not usable:
            final_action, status = "HOLD", "COST_SCENARIO_MISSING_STALE_OR_MISMATCHED"
            title = "Resize suggested; matching cost scenario needs evaluation"
            reasons.append("No newer saved cost scenario evaluated against this same RESIZE decision is available.")
        elif (
            safety.get("high_resource_pressure") is True
            or safety.get("high_memory_risk") is True
            or safety.get("status") not in {
                RESIZE_STATUS, "NO_COST_REDUCTION", "NOT_ASSESSABLE"
            }
        ):
            final_action, status = "HOLD", "RESIZE_SAFETY_NOT_SATISFIED"
            title = "Resize is not cleared by the safety assessment"
            reasons.append("The saved scenario reports a workload safety block or an unrecognized safety state.")
        else:
            final_action, status = "RESIZE", "RESIZE_REVIEW_REQUIRED"
            title = "Review a resource resize"
            reasons += ["The latest resource optimizer returned RESIZE.", "The saved scenario was evaluated against the same RESIZE decision."]
            validation += [
                "Validate target CPU and memory capacity against representative peak workload.",
                "Check application performance, availability, and rollback requirements before applying a change.",
            ]
    elif action == "SWITCH":
        if not usable:
            final_action, status = "HOLD", "SWITCH_SCENARIO_MISSING_STALE_OR_MISMATCHED"
            title = "Switch suggested; matching cost scenario needs evaluation"
            reasons.append("No newer saved cost scenario evaluated against this same SWITCH decision is available.")
        elif (
            safety.get("high_resource_pressure") is True
            or safety.get("high_memory_risk") is True
            or safety.get("status") not in {
                SWITCH_STATUS, "NO_COST_REDUCTION", "NOT_ASSESSABLE"
            }
        ):
            final_action, status = "HOLD", "SWITCH_SAFETY_NOT_SATISFIED"
            title = "Switch is not cleared by the safety assessment"
            reasons.append("The saved scenario reports a workload safety block or an unrecognized safety state.")
        else:
            final_action, status = "SWITCH", "SWITCH_MIGRATION_REVIEW_REQUIRED"
            title = "Review an alternative VM or cloud provider"
            reasons += ["The latest resource optimizer returned SWITCH.", "The saved scenario was evaluated against the same SWITCH decision."]
            validation += [
                "Validate runtime, operating system, CPU architecture, storage, networking, security, and regional compatibility.",
                "Estimate data transfer, migration effort, downtime, operational costs, and rollback requirements.",
            ]
    else:
        final_action, status = "HOLD", "RESOURCE_DECISION_UNAVAILABLE"
        title = "A current resource recommendation is required"
        reasons.append("No recognized KEEP, RESIZE, or SWITCH recommendation was supplied.")

    if not result:
        reasons.append("No saved cost scenario is available for this application.")
    elif not fresh:
        reasons.append("The saved cost scenario is older than the latest resource analysis or its timestamp could not be verified.")
    elif not action_matches:
        reasons.append("The saved cost scenario's workload decision does not match the latest resource optimizer decision.")

    if result and not comparable:
        reasons.append(comparison.get("comparison_note") or "Current and proposed cost scopes are not comparable; savings are withheld.")
    elif result and pricing.get("provider_pricing_verified") is not True:
        reasons.append("The cost scenario does not use verified provider retail pricing; supplied cost differences are estimates.")

    workload_source = metadata.get("workload_source")
    if workload_source == "Alibaba Cluster Trace 2017":
        validation.append("The workload input is Alibaba Cluster Trace 2017 demonstration data, not confirmed live telemetry from this application.")

    return {
        "status": "success",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "final_recommendation": {
            "action": final_action,
            "status": status,
            "title": title,
            "priority": str(resource.get("priority") or "UNSPECIFIED").upper(),
            "resource_optimizer_action": action,
            "cost_scenario_usable": usable,
            "potential_monthly_savings": savings,
            "potential_savings_percent": savings_pct,
            "savings_are_provider_verified": bool(usable and pricing.get("provider_pricing_verified") is True),
            "reasons": reasons,
            "required_validation": validation,
        },
        "source_context": {
            "workload_source": workload_source,
            "resource_recommendation_created_at": recommendation_created_at,
            "cost_scenario_created_at": scenario_created_at or scenario.get("created_at"),
            "cost_scenario_fresh_for_latest_resource_decision": fresh,
            "cost_scenario_action_matches": action_matches,
            "cost_comparison_status": comparison_status,
            "pricing_source": pricing.get("pricing_source"),
            "provider_pricing_verified": pricing.get("provider_pricing_verified") is True,
            "cost_scope_comparable": comparable,
        },
        "safety": {
            "automatic_infrastructure_changes": False,
            "manual_validation_required": True,
            "resource_optimizer_decision_overridden": False,
        },
    }
