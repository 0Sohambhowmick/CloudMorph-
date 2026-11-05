from pydantic import BaseModel, EmailStr, ConfigDict


# =========================================================
# AUTH REQUESTS
# =========================================================

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


# =========================================================
# USER RESPONSE
# =========================================================

class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    company: str
    role: str
    provider: str
    ai_recommendations: str
    monitoring_frequency: str
    auto_optimization: str

    model_config = ConfigDict(from_attributes=True)


# =========================================================
# TOKEN RESPONSE
# =========================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse