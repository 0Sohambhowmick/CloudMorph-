
import math
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app import auth
from app.database import applications_collection, db


router = APIRouter(
    prefix="/analysis",
    tags=["Optimization Analysis"],
)

security = HTTPBearer()
notifications_collection = db["notifications"]


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Return the email of the authenticated user."""
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


def parse_metric(value, field_name, default=None):
    """Parse a finite numeric metric without silently accepting invalid data."""
    if value is None:
        return default

    if isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} contains invalid data",
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} contains invalid data",
        )

    if not math.isfinite(number):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} contains invalid data",
        )

    return number


@router.post("/applications/{application_id}/decision")
async def run_optimization_decision(
    application_id: str,
    user_email: str = Depends(get_current_user),
):
    """
    Evaluate stored application metrics using transparent rules.
    This is a rule-based engine, not a trained ML model.
    """
    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found",
        )

    monthly_cost = parse_metric(
        application.get(
            "cost",
            application.get("monthly_cost", 0),
        ),
        "Application cost",
        default=0.0,
    )

    stored_optimized_cost = parse_metric(
        application.get("optimized_cost", 0),
        "Optimized cost",
        default=0.0,
    )

    raw_utilization = application.get("utilization")
    utilization = parse_metric(
        raw_utilization,
        "Utilization",
        default=None,
    )

    if monthly_cost < 0 or stored_optimized_cost < 0:
        raise HTTPException(
            status_code=422,
            detail="Application costs cannot be negative",
        )

    if utilization is not None and not 0 <= utilization <= 100:
        raise HTTPException(
            status_code=422,
            detail="Utilization must be between 0 and 100",
        )

    # Make a conservative recommendation using available data.
    # Missing utilization must not be treated as 0%.
    if (
        monthly_cost > 0
        and stored_optimized_cost > 0
        and stored_optimized_cost < monthly_cost * 0.80
    ):
        decision = "SWITCH"
        projected_cost = stored_optimized_cost
        reasons = [
            "The stored optimized cost is at least 20% lower than current cost.",
            "The alternative configuration requires compatibility validation.",
            "Migration effort and operational risk must be assessed before switching.",
        ]

    elif (
        monthly_cost > 0
        and utilization is not None
        and utilization <= 35
    ):
        decision = "RESIZE"
        projected_cost = monthly_cost * 0.80
        reasons = [
            f"Recorded utilization is {utilization:g}%, indicating possible underutilization.",
            "A 20% cost reduction is an estimate, not a measured saving.",
            "Validate workload peaks and performance requirements before resizing.",
        ]

    else:
        decision = "KEEP"
        projected_cost = monthly_cost

        if utilization is None:
            reasons = [
                "Utilization data is missing, so underutilization cannot be established.",
                "Collect workload utilization metrics before considering a resize.",
                "Retain the current configuration pending sufficient evidence.",
            ]
        else:
            reasons = [
                "The available metrics do not establish a sufficiently strong cost-saving case.",
                "Retain the current configuration pending more workload and performance data.",
            ]

    estimated_savings = max(
        0.0,
        monthly_cost - projected_cost,
    )
    now = datetime.utcnow().isoformat()

    await applications_collection.update_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        },
        {
            "$set": {
                "decision": decision,
                "optimized_cost": projected_cost,
                "updated_at": now,
            }
        },
    )

    # Notify the user about the completed decision.
    try:
        await notifications_collection.insert_one(
            {
                "user_email": user_email,
                "title": f"Optimization Decision: {decision}",
                "message": (
                    f"CloudMorph evaluated {application.get('name', 'your application')} "
                    f"and recommended {decision}. Estimated monthly savings: "
                    f"${estimated_savings:.2f}. Review the recommendation before "
                    "making infrastructure changes."
                ),
                "type": "AI",
                "read": False,
                "created_at": now,
                "updated_at": now,
            }
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Failed to create an optimization notification."
        )

    return {
        "application_id": str(application["_id"]),
        "application_name": application.get("name", ""),
        "decision": decision,
        "decision_engine": "rule_based",
        "current_monthly_cost": round(monthly_cost, 2),
        "projected_monthly_cost": round(projected_cost, 2),
        "estimated_monthly_savings": round(estimated_savings, 2),
        "estimated_annual_savings": round(estimated_savings * 12, 2),
        "utilization_percent": (
            round(utilization, 2)
            if utilization is not None
            else None
        ),
        "reasons": reasons,
        "requires_manual_validation": True,
        "analyzed_at": now,
    }
