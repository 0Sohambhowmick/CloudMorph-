
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
    prefix="/applications",
    tags=["Applications"],
)

security = HTTPBearer()
notifications_collection = db["notifications"]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

UNDERUTILIZATION_THRESHOLD = 35.0
HIGH_UTILIZATION_THRESHOLD = 85.0
UTILIZATION_SCORE_PENALTY = 2.0


def calculate_optimization_score(utilization):
    """
    Calculate a rule-based utilization fit score.

    35% to 85% utilization: 100 points.
    Below 35%: lose 2 points per percentage point.
    Above 85%: lose 2 points per percentage point.
    Missing or invalid utilization: None.
    """

    if utilization is None:
        return None

    try:
        utilization = float(utilization)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(utilization) or not 0 <= utilization <= 100:
        return None

    if (
        UNDERUTILIZATION_THRESHOLD
        <= utilization
        <= HIGH_UTILIZATION_THRESHOLD
    ):
        return 100.0

    if utilization < UNDERUTILIZATION_THRESHOLD:
        deviation = UNDERUTILIZATION_THRESHOLD - utilization
    else:
        deviation = utilization - HIGH_UTILIZATION_THRESHOLD

    score = 100 - (
        deviation * UTILIZATION_SCORE_PENALTY
    )

    return round(max(0.0, min(100.0, score)), 2)


def calculate_health_score(doc):
    """
    Health score is not calculated until genuine health
    metrics such as uptime, latency, and error rate exist.
    """

    return None



def calculate_cost_efficiency_score(current_cost, optimized_cost):
    """
    Calculate cost efficiency from estimated monthly savings.

    Score is the percentage reduction from current cost,
    limited to the range 0-100.

    Missing/invalid costs, or a non-positive current cost, return None.
    A zero optimized cost is valid when current_cost is positive.
    """

    try:
        current_cost = float(current_cost)
        optimized_cost = float(optimized_cost)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(current_cost)
        or not math.isfinite(optimized_cost)
        or current_cost <= 0
        or optimized_cost < 0
    ):
        return None

    savings_percentage = (
        (current_cost - optimized_cost) / current_cost
    ) * 100

    return round(max(0.0, min(100.0, savings_percentage)), 2)


def calculate_overall_optimization_score(
    utilization_score,
    cost_efficiency_score,
):
    """
    Combine utilization fit and cost efficiency equally.

    Both component scores must be available.
    """

    if utilization_score is None or cost_efficiency_score is None:
        return None

    return round(
        (utilization_score * 0.5)
        + (cost_efficiency_score * 0.5),
        2,
    )


def application_document(doc):
    """Convert a MongoDB application document into JSON-safe data."""

    if not doc:
        return None

    utilization = doc.get("utilization")

    optimization_score = calculate_optimization_score(
        utilization
    )

    health_score = calculate_health_score(doc)

    cost_efficiency_score = calculate_cost_efficiency_score(
        doc.get("cost"),
        doc.get("optimized_cost"),
    )

    overall_optimization_score = calculate_overall_optimization_score(
        optimization_score,
        cost_efficiency_score,
    )

    return {
        "id": str(doc["_id"]),
        "user_email": doc["user_email"],
        "name": doc.get("name", ""),
        "description": doc.get("description", ""),
        "provider": doc.get("provider", "AWS"),
        "service": doc.get("service", ""),
        "category": doc.get("category", ""),
        "services": doc.get("services", 1),
        "cost": doc.get("cost", 0),
        "optimized_cost": doc.get("optimized_cost", 0),

        # Do not interpret missing utilization as actual 0%.
        "utilization": utilization,

        # Rule-based utilization fit score.
        "optimization_score": optimization_score,
        "optimization_score_type": "utilization_fit_rule_based",

        # Cost efficiency score based on estimated monthly savings.
        "cost_efficiency_score": cost_efficiency_score,
        "cost_efficiency_score_type": "monthly_savings_percentage",

        # Combined overall optimization score.
        "overall_optimization_score": overall_optimization_score,
        "overall_optimization_score_status": (
            "available"
            if overall_optimization_score is not None
            else "pending"
        ),

        # Health score is pending until genuine health metrics exist.
        "health_score": health_score,
        "health_score_status": "pending",

        "decision": doc.get("decision", "ANALYZE"),
        "status": doc.get("status", "Active"),
        "region": doc.get("region", ""),
        "created_at": doc.get("created_at", ""),
        "updated_at": doc.get("updated_at"),
    }


def parse_application_id(application_id: str) -> ObjectId:
    """Validate and convert a string ID into a MongoDB ObjectId."""

    try:
        return ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID",
        )


def parse_number(
    value,
    field_name,
    default=0,
    integer=False,
    minimum=None,
    maximum=None,
):
    """Validate finite numeric values and optional bounds."""

    if value is None:
        if default is None:
            return None
        value = default

    if isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a valid number",
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a valid number",
        )

    if not math.isfinite(number):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a finite number",
        )

    if minimum is not None and number < minimum:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be at least {minimum}",
        )

    if maximum is not None and number > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must not exceed {maximum}",
        )

    if integer:
        if not number.is_integer():
            raise HTTPException(
                status_code=422,
                detail=f"{field_name} must be a whole number",
            )
        return int(number)

    return number


def validate_text(value, field_name, default=""):
    """Validate text fields without silently accepting other data types."""

    if value is None:
        return default

    if not isinstance(value, str):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be text",
        )

    return value.strip()


def validate_choice(value, field_name, allowed_values, default):
    """Validate fields that must use one of the supported values."""

    if value is None:
        value = default

    if not isinstance(value, str):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a valid choice",
        )

    normalized = value.strip().upper()

    if normalized not in allowed_values:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{field_name} must be one of: "
                f"{', '.join(sorted(allowed_values))}"
            ),
        )

    return normalized


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Decode the JWT and return the authenticated user's email."""

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            auth.SECRET_KEY,
            algorithms=[auth.ALGORITHM],
        )

        email = payload.get("sub")

        if not email:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )

        return email

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )


async def create_application_notification(
    user_email: str,
    title: str,
    message: str,
    notification_type: str = "info",
):
    """
    Save a notification for the specified user.

    Notification failures should not turn a successful application
    operation into a failed API request.
    """

    now = datetime.utcnow().isoformat()

    try:
        await notifications_collection.insert_one(
            {
                "user_email": user_email,
                "title": title,
                "message": message,
                "type": notification_type,
                "read": False,
                "created_at": now,
                "updated_at": now,
            }
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Failed to create an application notification."
        )


def validate_application_fields(fields):
    """Apply shared validation to application fields."""

    if "description" in fields:
        fields["description"] = validate_text(
            fields["description"], "description"
        )

    if "region" in fields:
        fields["region"] = validate_text(
            fields["region"], "region"
        )

    if "service" in fields:
        fields["service"] = validate_text(
            fields["service"], "service"
        )

    if "category" in fields:
        fields["category"] = validate_text(
            fields["category"], "category"
        )

    if "provider" in fields:
        fields["provider"] = validate_choice(
            fields["provider"],
            "provider",
            {"AWS", "GCP", "AZURE"},
            "AWS",
        )

    if "status" in fields:
        fields["status"] = validate_choice(
            fields["status"],
            "status",
            {"ACTIVE", "INACTIVE", "PENDING"},
            "ACTIVE",
        ).title()

    if "decision" in fields:
        fields["decision"] = validate_choice(
            fields["decision"],
            "decision",
            {"ANALYZE", "KEEP", "RESIZE", "SWITCH"},
            "ANALYZE",
        )

    if "cost" in fields:
        fields["cost"] = parse_number(
            fields["cost"],
            "cost",
            minimum=0,
        )

    if "optimized_cost" in fields:
        fields["optimized_cost"] = parse_number(
            fields["optimized_cost"],
            "optimized_cost",
            minimum=0,
        )

    if "services" in fields:
        fields["services"] = parse_number(
            fields["services"],
            "services",
            integer=True,
            minimum=1,
        )

    if "utilization" in fields:
        fields["utilization"] = parse_number(
            fields["utilization"],
            "utilization",
            default=None,
            integer=True,
            minimum=0,
            maximum=100,
        )

    return fields


# =========================================================
# CREATE APPLICATION
# =========================================================

@router.post("")
async def create_application(
    application: dict,
    user_email: str = Depends(get_current_user),
):

    name = application.get("name")

    if not isinstance(name, str) or not name.strip():
        raise HTTPException(
            status_code=422,
            detail="Application name is required",
        )

    now = datetime.utcnow().isoformat()

    new_application = {
        "user_email": user_email,
        "name": name.strip(),
        "description": application.get("description", ""),
        "provider": application.get("provider", "AWS"),
        "service": application.get(
            "service",
            application.get("category", ""),
        ),
        "category": application.get(
            "category",
            application.get("service", ""),
        ),
        "services": application.get("services", 1),
        "cost": application.get(
            "cost",
            application.get("monthly_cost", 0),
        ),
        "optimized_cost": application.get(
            "optimized_cost",
            application.get("optimizedCost", 0),
        ),

        # Missing utilization remains unknown, not 0%.
        "utilization": application.get("utilization"),

        "decision": application.get("decision", "ANALYZE"),
        "status": application.get("status", "Active"),
        "region": application.get("region", ""),
        "created_at": now,
        "updated_at": now,
    }

    new_application = validate_application_fields(new_application)

    result = await applications_collection.insert_one(
        new_application
    )

    created = await applications_collection.find_one(
        {"_id": result.inserted_id}
    )

    await create_application_notification(
        user_email=user_email,
        title="Application Added",
        message=(
            f"{created.get('name', 'Your application')} "
            "was successfully added to CloudMorph."
        ),
        notification_type="info",
    )

    decision = str(created.get("decision", "ANALYZE")).upper()

    if decision in {"KEEP", "RESIZE", "SWITCH"}:
        await create_application_notification(
            user_email=user_email,
            title=f"Optimization Decision: {decision}",
            message=(
                f"CloudMorph recorded a {decision} decision for "
                f"{created.get('name', 'your application')}."
            ),
            notification_type="AI",
        )

    return application_document(created)


# =========================================================
# GET ALL APPLICATIONS FOR THE LOGGED-IN USER
# =========================================================

@router.get("")
async def get_applications(
    user_email: str = Depends(get_current_user),
):

    apps = []

    cursor = applications_collection.find(
        {"user_email": user_email}
    ).sort("created_at", -1)

    async for app in cursor:
        apps.append(application_document(app))

    return apps


# =========================================================
# GET A SINGLE APPLICATION
# =========================================================

@router.get("/{application_id}")
async def get_application(
    application_id: str,
    user_email: str = Depends(get_current_user),
):

    mongo_id = parse_application_id(application_id)

    app = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not app:
        raise HTTPException(
            status_code=404,
            detail="Application not found",
        )

    return application_document(app)


# =========================================================
# UPDATE APPLICATION
# =========================================================

@router.put("/{application_id}")
async def update_application(
    application_id: str,
    payload: dict,
    user_email: str = Depends(get_current_user),
):

    mongo_id = parse_application_id(application_id)

    allowed_fields = [
        "name",
        "description",
        "provider",
        "service",
        "category",
        "services",
        "cost",
        "optimized_cost",
        "utilization",
        "decision",
        "status",
        "region",
    ]

    update_fields = {
        field: payload[field]
        for field in allowed_fields
        if field in payload
    }

    if "name" in update_fields:
        if (
            not isinstance(update_fields["name"], str)
            or not update_fields["name"].strip()
        ):
            raise HTTPException(
                status_code=422,
                detail="Application name cannot be empty",
            )

        update_fields["name"] = update_fields["name"].strip()

    update_fields = validate_application_fields(update_fields)

    existing = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Application not found",
        )

    if update_fields:
        update_fields["updated_at"] = datetime.utcnow().isoformat()

        await applications_collection.update_one(
            {
                "_id": mongo_id,
                "user_email": user_email,
            },
            {"$set": update_fields},
        )

    updated = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    old_decision = str(
        existing.get("decision", "ANALYZE")
    ).upper()

    new_decision = str(
        updated.get("decision", "ANALYZE")
    ).upper()

    if (
        "decision" in update_fields
        and new_decision in {"KEEP", "RESIZE", "SWITCH"}
        and new_decision != old_decision
    ):
        await create_application_notification(
            user_email=user_email,
            title=f"Optimization Decision: {new_decision}",
            message=(
                f"The decision for {updated.get('name', 'your application')} "
                f"changed from {old_decision} to {new_decision}."
            ),
            notification_type="AI",
        )

    return application_document(updated)


# =========================================================
# DELETE APPLICATION
# =========================================================

@router.delete("/{application_id}")
async def delete_application(
    application_id: str,
    user_email: str = Depends(get_current_user),
):

    mongo_id = parse_application_id(application_id)

    existing = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Application not found",
        )

    result = await applications_collection.delete_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Application not found",
        )

    await create_application_notification(
        user_email=user_email,
        title="Application Deleted",
        message=(
            f"{existing.get('name', 'Your application')} "
            "was removed from CloudMorph."
        ),
        notification_type="info",
    )

    return {
        "message": "Application deleted successfully",
    }
