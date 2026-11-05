from bson import ObjectId


def user_document(user) -> dict:
    """
    Convert MongoDB User document into JSON-safe response.
    """

    if not user:
        return None

    return {
        "id": str(user["_id"]),
        "name": user.get("name"),
        "email": user.get("email"),
        "company": user.get("company", "CloudMorph Labs"),
        "role": user.get("role", "Cloud Engineer"),
        "provider": user.get("provider", "AWS"),
        "ai_recommendations": user.get("ai_recommendations", "Balanced"),
        "monitoring_frequency": user.get("monitoring_frequency", "Daily"),
        "auto_optimization": user.get("auto_optimization", "Disabled"),
    }


def application_document(app) -> dict:
    """
    Convert MongoDB Application document into JSON-safe response.
    """

    if not app:
        return None

    return {
        "id": str(app["_id"]),
        "name": app.get("name"),
        "provider": app.get("provider"),
        "service": app.get("service"),
        "region": app.get("region"),
        "monthly_cost": app.get("monthly_cost", 0),
        "utilization": app.get("utilization", 0),
        "status": app.get("status", "Active"),
        "owner_email": app.get("owner_email"),
    }


def object_id(id: str):
    return ObjectId(id)