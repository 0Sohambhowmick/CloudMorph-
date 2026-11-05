import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_FILE)

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "cloudmorph")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL not found in .env file.")

# =========================================================
# MONGODB CLIENT
# =========================================================

client = AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]

# =========================================================
# COLLECTIONS
# =========================================================

users_collection = db["users"]
applications_collection = db["applications"]

# =========================================================
# DATABASE DEPENDENCY
# =========================================================

async def get_database():
    return db