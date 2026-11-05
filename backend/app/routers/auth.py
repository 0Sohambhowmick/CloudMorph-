from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app import auth, models, schemas
from app.database import users_collection

router = APIRouter(prefix="/auth", tags=["Authentication"])

# JWT Security
security = HTTPBearer()


# =========================================================
# SIGNUP
# =========================================================

@router.post("/signup", response_model=schemas.TokenResponse)
async def signup(user: schemas.SignupRequest):

    existing = await users_collection.find_one({"email": user.email})

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered",
        )

    new_user = {
        "name": user.name,
        "email": user.email,
        "password": auth.hash_password(user.password),
        "company": "CloudMorph Labs",
        "role": "Cloud Engineer",
        "provider": "AWS",
        "ai_recommendations": "Balanced",
        "monitoring_frequency": "Daily",
        "auto_optimization": "Disabled",
    }

    result = await users_collection.insert_one(new_user)

    created = await users_collection.find_one(
        {"_id": result.inserted_id}
    )

    token = auth.create_access_token(
        {"sub": user.email}
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": models.user_document(created),
    }


# =========================================================
# LOGIN
# =========================================================

@router.post("/login", response_model=schemas.TokenResponse)
async def login(user: schemas.LoginRequest):

    existing = await users_collection.find_one(
        {"email": user.email}
    )

    if not existing:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )

    if not auth.verify_password(
        user.password,
        existing["password"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )

    token = auth.create_access_token(
        {"sub": user.email}
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": models.user_document(existing),
    }


# =========================================================
# UPDATE USER PROFILE
# =========================================================

@router.put("/profile")
async def update_profile(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):

    token = credentials.credentials

    try:
        data = jwt.decode(
            token,
            auth.SECRET_KEY,
            algorithms=[auth.ALGORITHM],
        )

        email = data.get("sub")

        if not email:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload",
            )

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )

    user = await users_collection.find_one(
        {"email": email}
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    update_fields = {}

    for field in [
        "provider",
        "company",
        "role",
        "ai_recommendations",
        "monitoring_frequency",
        "auto_optimization",
    ]:
        if field in payload:
            update_fields[field] = payload[field]

    if update_fields:
        await users_collection.update_one(
            {"email": email},
            {"$set": update_fields},
        )

    updated_user = await users_collection.find_one(
        {"email": email}
    )

    return {
        "message": "Profile updated successfully",
        "user": models.user_document(updated_user),
    }