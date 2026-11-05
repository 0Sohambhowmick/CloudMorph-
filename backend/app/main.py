
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    auth,
    applications,
    notifications,
    analysis,
    forecasting,
    optimization,
    analytics,
)

app = FastAPI(
    title="CloudMorph API",
    version="1.0.0",
)

# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# API ROUTERS
# =========================================================

app.include_router(auth.router)
app.include_router(applications.router)
app.include_router(notifications.router)
app.include_router(analysis.router)
app.include_router(forecasting.router)
app.include_router(optimization.router)
app.include_router(analytics.router)

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
async def root():
    return {
        "message": "CloudMorph API is running",
        "database": "MongoDB Atlas",
        "version": "1.0.0",
    }
