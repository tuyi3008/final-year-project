import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration - use correct default values
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:admin123@localhost:27017/?authSource=admin")
DB_NAME = os.getenv("DB_NAME", "styletrans_db")

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Connect to MongoDB"""
    try:
        print(f"Connecting to MongoDB...")
        print(f"Using URL: {MONGODB_URL.replace('admin123', '******')}")
        
        mongodb.client = AsyncIOMotorClient(MONGODB_URL)
        mongodb.database = mongodb.client[DB_NAME]
        print(f"✅ Successfully connected to MongoDB, database: {DB_NAME}")
        
        # Test connection
        await mongodb.client.admin.command('ping')
        print("✅ MongoDB ping successful")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"Please check the MongoDB connection configuration")

async def close_mongo_connection():
    """Close MongoDB connection"""
    if mongodb.client:
        mongodb.client.close()
        print("❌ MongoDB connection closed")

# Dependency to get the database instance
def get_db():
    return mongodb.database

# Helper to get users collection
def get_users_collection():
    db = get_db()
    return db["users"] if db else None

# Helper to get history collection
def get_history_collection():
    db = get_db()
    if db is None:
        return None
    return db["history"]