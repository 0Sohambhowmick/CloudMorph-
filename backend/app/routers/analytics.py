
import math
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app import auth
from app.database import applications_collection, db


router = APIRouter(
    prefix="/analytics",
    tags=["Analytics Dashboard"],
)

security = HTTPBearer()

recommendations_collection = db["optimization_recommendations"]


# =========================================================
# AUTHENTICATION
# =========================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Return the authenticated user's email."""

    try:
        payload = jwt.decode(
            credentials.credentials,
            auth.SECRET_KEY,
            algorithms=[auth.ALGORITHM],
        )

        email = payload.get("sub")

        if not email:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
            )

        return email

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )


# =========================================================
# VALIDATION HELPERS
# =========================================================

def safe_number(value):
    """Return a finite float or None for missing/invalid values."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_date(value):
    """Convert supported date values into ISO strings."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.isoformat()

    if isinstance(value, str):
        return value

    return None


def round_value(value):
    """Round a numeric value to two decimal places."""

    if value is None:
        return None

    return round(float(value), 2)


# =========================================================
# MAIN ANALYTICS ENDPOINT
# =========================================================

@router.get("/overview")
async def get_analytics_overview(
    user_email: str = Depends(get_current_user),
):
    """
    Return dashboard analytics for the authenticated user.

    All cost figures are based on stored application records.
    They are not verified cloud-provider billing figures.
    """

    try:
        applications = []

        cursor = applications_collection.find(
            {"user_email": user_email}
        )

        async for application in cursor:
            applications.append(application)

        # -------------------------------------------------
        # APPLICATION SUMMARY
        # -------------------------------------------------

        total_applications = len(applications)

        active_applications = sum(
            1
            for app in applications
            if str(app.get("status", "ACTIVE")).upper() == "ACTIVE"
        )

        inactive_applications = sum(
            1
            for app in applications
            if str(app.get("status", "")).upper() == "INACTIVE"
        )

        pending_applications = sum(
            1
            for app in applications
            if str(app.get("status", "")).upper() == "PENDING"
        )

        # -------------------------------------------------
        # COST AND SAVINGS
        # -------------------------------------------------

        total_current_cost = 0.0
        total_optimized_cost = 0.0

        applications_with_cost = 0
        applications_with_optimized_cost = 0

        total_monthly_savings = 0.0
        total_monthly_cost_increase = 0.0

        cost_efficiency_scores = []

        for app in applications:
            current_cost = safe_number(app.get("cost"))
            optimized_cost = safe_number(
                app.get("optimized_cost")
            )

            if current_cost is not None and current_cost >= 0:
                total_current_cost += current_cost
                applications_with_cost += 1

            if (
                optimized_cost is not None
                and optimized_cost >= 0
            ):
                total_optimized_cost += optimized_cost
                applications_with_optimized_cost += 1

            if (
                current_cost is not None
                and optimized_cost is not None
                and current_cost > 0
                and optimized_cost >= 0
            ):
                difference = current_cost - optimized_cost

                if difference > 0:
                    total_monthly_savings += difference
                elif difference < 0:
                    total_monthly_cost_increase += abs(difference)

                score = max(
                    0.0,
                    min(100.0, (difference / current_cost) * 100),
                )

                cost_efficiency_scores.append(score)

        annual_savings = total_monthly_savings * 12

        overall_savings_percentage = (
            (total_monthly_savings / total_current_cost) * 100
            if total_current_cost > 0
            else None
        )

        average_cost_efficiency = (
            sum(cost_efficiency_scores)
            / len(cost_efficiency_scores)
            if cost_efficiency_scores
            else None
        )

        # -------------------------------------------------
        # UTILIZATION AND OPTIMIZATION SCORES
        # -------------------------------------------------

        utilization_values = []
        utilization_fit_scores = []
        overall_optimization_scores = []

        for app in applications:
            utilization = safe_number(app.get("utilization"))

            if (
                utilization is not None
                and 0 <= utilization <= 100
            ):
                utilization_values.append(utilization)

                if utilization <= 35:
                    score = max(0.0, 100 - ((35 - utilization) * 2))
                elif utilization >= 85:
                    score = max(0.0, 100 - ((utilization - 85) * 2))
                else:
                    score = 100.0

                utilization_fit_scores.append(score)

                current_cost = safe_number(app.get("cost"))
                optimized_cost = safe_number(
                    app.get("optimized_cost")
                )

                if (
                    current_cost is not None
                    and optimized_cost is not None
                    and current_cost > 0
                    and optimized_cost >= 0
                ):
                    difference = current_cost - optimized_cost

                    cost_score = max(
                        0.0,
                        min(100.0, (difference / current_cost) * 100),
                    )

                    overall_optimization_scores.append(
                        (score + cost_score) / 2
                    )

        average_utilization = (
            sum(utilization_values) / len(utilization_values)
            if utilization_values
            else None
        )

        average_utilization_fit = (
            sum(utilization_fit_scores)
            / len(utilization_fit_scores)
            if utilization_fit_scores
            else None
        )

        average_overall_score = (
            sum(overall_optimization_scores)
            / len(overall_optimization_scores)
            if overall_optimization_scores
            else None
        )

        # -------------------------------------------------
        # UTILIZATION DISTRIBUTION
        # -------------------------------------------------

        utilization_distribution = {
            "underutilized": 0,
            "optimal": 0,
            "high": 0,
            "unavailable": total_applications - len(utilization_values),
        }

        for utilization in utilization_values:
            if utilization <= 35:
                utilization_distribution["underutilized"] += 1
            elif utilization >= 85:
                utilization_distribution["high"] += 1
            else:
                utilization_distribution["optimal"] += 1

        # -------------------------------------------------
        # PROVIDER DISTRIBUTION
        # -------------------------------------------------

        provider_distribution = {}

        for app in applications:
            provider = str(
                app.get("provider", "UNKNOWN")
            ).upper()

            if provider not in provider_distribution:
                provider_distribution[provider] = {
                    "applications": 0,
                    "current_monthly_cost": 0.0,
                    "optimized_monthly_cost": 0.0,
                }

            provider_distribution[provider]["applications"] += 1

            current_cost = safe_number(app.get("cost"))
            optimized_cost = safe_number(
                app.get("optimized_cost")
            )

            if current_cost is not None and current_cost >= 0:
                provider_distribution[provider][
                    "current_monthly_cost"
                ] += current_cost

            if optimized_cost is not None and optimized_cost >= 0:
                provider_distribution[provider][
                    "optimized_monthly_cost"
                ] += optimized_cost

        for provider_data in provider_distribution.values():
            for field in (
                "current_monthly_cost",
                "optimized_monthly_cost",
            ):
                provider_data[field] = round_value(
                    provider_data[field]
                )

        # -------------------------------------------------
        # APPLICATION-WISE COST COMPARISON
        # -------------------------------------------------

        application_costs = []

        for app in applications:
            current_cost = safe_number(app.get("cost"))
            optimized_cost = safe_number(
                app.get("optimized_cost")
            )

            if current_cost is None and optimized_cost is None:
                continue

            difference = None

            if current_cost is not None and optimized_cost is not None:
                difference = current_cost - optimized_cost

            application_costs.append({
                "application_id": str(app["_id"]),
                "name": app.get("name", "Unnamed application"),
                "provider": app.get("provider", "UNKNOWN"),
                "current_monthly_cost": round_value(current_cost),
                "optimized_monthly_cost": round_value(optimized_cost),
                "estimated_monthly_savings": (
                    round_value(max(0.0, difference))
                    if difference is not None
                    else None
                ),
                "estimated_monthly_cost_increase": (
                    round_value(max(0.0, -difference))
                    if difference is not None
                    else None
                ),
            })

        application_costs.sort(
            key=lambda item: (
                item["current_monthly_cost"] is None,
                -(item["current_monthly_cost"] or 0),
            )
        )

        # -------------------------------------------------
        # DECISION DISTRIBUTION
        # -------------------------------------------------

        decision_distribution = {
            "KEEP": 0,
            "RESIZE": 0,
            "SWITCH": 0,
            "SCALE_UP": 0,
            "INVESTIGATE": 0,
            "OTHER": 0,
        }

        for app in applications:
            decision = str(
                app.get("decision", "ANALYZE")
            ).upper()

            if decision in decision_distribution:
                decision_distribution[decision] += 1
            else:
                decision_distribution["OTHER"] += 1

        # -------------------------------------------------
        # RECENT OPTIMIZATION HISTORY
        # -------------------------------------------------

        recent_recommendations = []

        recommendation_cursor = (
            recommendations_collection.find(
                {"user_email": user_email}
            )
            .sort("created_at", -1)
            .limit(10)
        )

        async for record in recommendation_cursor:
            recommendation = record.get(
                "recommendation",
                {},
            )

            cost_analysis = record.get(
                "cost_analysis",
                {},
            )

            recent_recommendations.append({
                "id": str(record.get("_id", "")),
                "application_id": str(
                    record.get("application_id", "")
                ),
                "application_name": record.get(
                    "application_name",
                    "",
                ),
                "action": recommendation.get(
                    "action",
                    record.get("decision", "UNKNOWN"),
                ),
                "priority": recommendation.get(
                    "priority",
                    "UNKNOWN",
                ),
                "estimated_monthly_savings": round_value(
                    safe_number(
                        cost_analysis.get(
                            "estimated_monthly_savings"
                        )
                    )
                ),
                "created_at": safe_date(
                    record.get("created_at")
                ),
            })

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {
            "status": "success",

            "summary": {
                "total_applications": total_applications,
                "active_applications": active_applications,
                "inactive_applications": inactive_applications,
                "pending_applications": pending_applications,
                "applications_with_cost_data": applications_with_cost,
                "applications_with_optimized_cost": (
                    applications_with_optimized_cost
                ),
            },

            "cost_analysis": {
                "total_current_monthly_cost": round_value(
                    total_current_cost
                ),
                "total_optimized_monthly_cost": round_value(
                    total_optimized_cost
                ),
                "estimated_monthly_savings": round_value(
                    total_monthly_savings
                ),
                "estimated_annual_savings": round_value(
                    annual_savings
                ),
                "estimated_monthly_cost_increase": round_value(
                    total_monthly_cost_increase
                ),
                "estimated_savings_percentage": round_value(
                    overall_savings_percentage
                ),
                "average_cost_efficiency_score": round_value(
                    average_cost_efficiency
                ),
            },

            "optimization_analysis": {
                "average_utilization": round_value(
                    average_utilization
                ),
                "average_utilization_fit_score": round_value(
                    average_utilization_fit
                ),
                "average_overall_optimization_score": round_value(
                    average_overall_score
                ),
                "utilization_distribution": (
                    utilization_distribution
                ),
                "decision_distribution": (
                    decision_distribution
                ),
            },

            "provider_distribution": provider_distribution,

            "application_costs": application_costs,

            "recent_recommendations": recent_recommendations,

            "metadata": {
                "data_source": "mongodb_application_records",
                "cost_data_source": "application_record",
                "cost_estimates_are_provider_verified": False,
                "historical_billing_available": False,
                "health_score_available": False,
                "requires_manual_validation": True,
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        }

    except HTTPException:
        raise

    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Failed to generate analytics overview."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate analytics overview",
        )
