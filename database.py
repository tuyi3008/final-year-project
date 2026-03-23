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
        
        await ensure_collections()

        await migrate_photo_styles()
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"Please check the MongoDB connection configuration")

async def close_mongo_connection():
    """Close MongoDB connection"""
    if mongodb.client:
        mongodb.client.close()
        print("❌ MongoDB connection closed")

async def migrate_photo_styles():
    try:
        if mongodb.database is None:
            return
        
        photos = mongodb.database["photos"]

        result = await photos.update_many(
            {"style": {"$exists": False}},
            {"$set": {"style": "original"}}
        )
        
        if result.modified_count > 0:
            print(f"✅ Updated {result.modified_count} photos with default style 'original'")
        else:
            print("✅ All photos already have style field")
        
    except Exception as e:
        print(f"⚠️ Error migrating photo styles: {e}")

async def ensure_collections():
    """Ensure all required collections exist with proper indexes"""
    if mongodb.database is None:
        return
    
    # List of collections to check/create
    collections = ["users", "history", "favorites", "gallery", "challenges", 
                   "challenge_participants", "challenge_submissions", "albums", 
                   "photos", "user_xp", "submission_likes", "gallery_likes"]
    
    existing_collections = await mongodb.database.list_collection_names()
    
    for collection_name in collections:
        if collection_name not in existing_collections:
            print(f"📁 Creating collection: {collection_name}")
            await mongodb.database.create_collection(collection_name)

    await create_indexes()

async def create_indexes():
    """Create indexes for better query performance"""
    if mongodb.database is None:
        return
    
    # Users collection indexes
    users = mongodb.database["users"]
    await users.create_index("email", unique=True)
    await users.create_index("username")
    
    # History collection indexes
    history = mongodb.database["history"]
    await history.create_index([("user_id", 1), ("created_at", -1)])
    
    # Favorites collection indexes
    favorites = mongodb.database["favorites"]
    await favorites.create_index([("user_id", 1), ("image_path", 1)], unique=True)
    
    # Gallery collection indexes
    gallery = mongodb.database["gallery"]
    await gallery.create_index([("user_id", 1), ("created_at", -1)])
    await gallery.create_index("likes")
    
    # Challenges collection indexes
    challenges = mongodb.database["challenges"]
    await challenges.create_index([("start_date", 1), ("end_date", 1)])
    
    # Challenge submissions indexes
    submissions = mongodb.database["challenge_submissions"]
    await submissions.create_index([("challenge_id", 1), ("likes", -1)])
    await submissions.create_index([("user_id", 1), ("challenge_id", 1)], unique=True)
    
    # User XP collection indexes
    user_xp = mongodb.database["user_xp"]
    await user_xp.create_index([("user_id", 1), ("created_at", -1)])
    
    # Like indexes
    submission_likes = mongodb.database["submission_likes"]
    await submission_likes.create_index([("submission_id", 1), ("user_id", 1)], unique=True)
    
    gallery_likes = mongodb.database["gallery_likes"]
    await gallery_likes.create_index([("image_id", 1), ("user_id", 1)], unique=True)
    
    # Albums and photos indexes
    albums = mongodb.database["albums"]
    await albums.create_index([("user_id", 1), ("created_at", -1)])
    
    photos = mongodb.database["photos"]
    await photos.create_index([("album_id", 1), ("uploaded_at", -1)])
    await photos.create_index("style")
    
    print("✅ All indexes created successfully")

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

def get_user_xp_collection():
    db = get_db()
    return db["user_xp"] if db else None

def get_challenges_collection():
    db = get_db()
    return db["challenges"] if db else None

def get_challenge_submissions_collection():
    db = get_db()
    return db["challenge_submissions"] if db else None

def get_gallery_collection():
    db = get_db()
    return db["gallery"] if db else None

def get_favorites_collection():
    db = get_db()
    return db["favorites"] if db else None

def get_photos_collection():
    db = get_db()
    return db["photos"] if db else None