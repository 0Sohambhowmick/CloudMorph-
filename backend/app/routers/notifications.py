
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app import auth
from app.database import db

router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"],
)

security = HTTPBearer()
notifications_collection = db["notifications"]


# =========================================================
# AUTHENTICATION
# =========================================================

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
                detail="Invalid token",
            )

        return email

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def parse_notification_id(notification_id: str) -> ObjectId:
    """Validate a notification ID before querying MongoDB."""
    try:
        return ObjectId(notification_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid notification ID",
        )


def notification_document(doc: dict) -> dict:
    """Convert a MongoDB notification into JSON-safe data."""
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title", ""),
        "message": doc.get("message", ""),
        "type": doc.get("type", "info"),
        "read": doc.get("read", False),
        "created_at": doc.get("created_at", ""),
        "updated_at": doc.get("updated_at"),
    }


# =========================================================
# CREATE NOTIFICATION
# =========================================================

@router.post("")
async def create_notification(
    payload: dict,
    user_email: str = Depends(get_current_user),
):
    title = payload.get("title")
    message = payload.get("message")

    if not isinstance(title, str) or not title.strip():
        raise HTTPException(
            status_code=422,
            detail="Notification title is required",
        )

    if not isinstance(message, str) or not message.strip():
        raise HTTPException(
            status_code=422,
            detail="Notification message is required",
        )

    now = datetime.utcnow().isoformat()

    notification = {
        "user_email": user_email,
        "title": title.strip(),
        "message": message.strip(),
        "type": payload.get("type", "info"),
        "read": False,
        "created_at": now,
        "updated_at": now,
    }

    result = await notifications_collection.insert_one(notification)

    created = await notifications_collection.find_one(
        {
            "_id": result.inserted_id,
            "user_email": user_email,
        }
    )

    return notification_document(created)


# =========================================================
# GET NOTIFICATIONS FOR THE LOGGED-IN USER
# =========================================================

@router.get("")
async def get_notifications(
    user_email: str = Depends(get_current_user),
):
    notifications = []

    cursor = notifications_collection.find(
        {"user_email": user_email}
    ).sort("created_at", -1)

    async for notification in cursor:
        notifications.append(notification_document(notification))

    return notifications


# =========================================================
# MARK ONE NOTIFICATION AS READ
# =========================================================

@router.patch("/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    user_email: str = Depends(get_current_user),
):
    mongo_id = parse_notification_id(notification_id)

    result = await notifications_collection.update_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        },
        {
            "$set": {
                "read": True,
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Notification not found",
        )

    notification = await notifications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    return notification_document(notification)


# =========================================================
# MARK ALL NOTIFICATIONS AS READ
# =========================================================

@router.patch("/read-all")
async def mark_all_notifications_read(
    user_email: str = Depends(get_current_user),
):
    result = await notifications_collection.update_many(
        {"user_email": user_email},
        {
            "$set": {
                "read": True,
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )

    return {
        "message": "All notifications marked as read",
        "updated_count": result.modified_count,
    }


# =========================================================
# DELETE ONE NOTIFICATION
# =========================================================

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    user_email: str = Depends(get_current_user),
):
    mongo_id = parse_notification_id(notification_id)

    result = await notifications_collection.delete_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Notification not found",
        )

    return {
        "message": "Notification deleted successfully",
    }


# =========================================================
# DELETE ALL NOTIFICATIONS FOR THE LOGGED-IN USER
# =========================================================

@router.delete("")
async def delete_all_notifications(
    user_email: str = Depends(get_current_user),
):
    result = await notifications_collection.delete_many(
        {"user_email": user_email}
    )

    return {
        "message": "All notifications deleted successfully",
        "deleted_count": result.deleted_count,
    }
