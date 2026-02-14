# database.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "styletrans_db")

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Connected to MongoDB"""
    mongodb.client = AsyncIOMotorClient(MONGODB_URL)
    mongodb.database = mongodb.client[DB_NAME]
    print(f"✅ Successfully connected to MongoDB, database: {DB_NAME}")

async def close_mongo_connection():
    """Close MongoDB connection"""
    if mongodb.client:
        mongodb.client.close()
        print("❌ MongoDB connection closed")

# Dependency to get the database instance
def get_db():
    return mongodb.database