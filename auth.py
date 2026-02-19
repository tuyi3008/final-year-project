from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from bson import ObjectId
from database import get_db, mongodb

# Load environment variables
load_dotenv()

# ==================== MongoDB Configuration from database.py ====================
# We'll use database.py's connection instead of creating a new one
DB_NAME = os.getenv("DB_NAME", "styletrans_db")

# ==================== JWT Configuration ====================
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable is not set")
    
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ==================== Password Encryption ====================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

# ==================== Data Models ====================
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserInDB(BaseModel):
    id: Optional[str] = None
    username: str
    email: Optional[str] = None
    hashed_password: str
    created_at: Optional[datetime] = None

# ==================== Helper Functions ====================
def get_users_collection():
    """Get users collection from database"""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    return db["users"]

# ==================== Utility Functions ====================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

async def get_user(email: str) -> Optional[UserInDB]:
    """Get user by email from MongoDB"""
    try:
        users_collection = get_users_collection()
        if users_collection is None:
            print("❌ Users collection is None")
            return None
            
        user_dict = await users_collection.find_one({"email": email})
        if user_dict:
            # Convert ObjectId to string
            user_dict["id"] = str(user_dict["_id"])
            return UserInDB(**user_dict)
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ==================== Core Authentication Functions ====================
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[UserInDB]:
    """
    Get current user from token.
    Returns UserInDB if token is valid, None if no token or invalid token.
    This function never raises HTTPException.
    """
    print(f"🔍 get_current_user called with token: {token[:20] if token else 'None'}...")
    
    # If no token provided, return None (anonymous user)
    if not token:
        print("ℹ️ No token provided, returning None (anonymous user)")
        return None
    
    try:
        # Try to decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"✅ Token decoded successfully: {payload}")
        
        email: str = payload.get("sub")
        if email is None:
            print("⚠️ No email in token, returning None")
            return None
        
        # Get user from database
        user = await get_user(email)
        if user is None:
            print(f"⚠️ User not found for email: {email}, returning None")
            return None
            
        print(f"✅ Authenticated user: {user.email}")
        return user
        
    except JWTError as e:
        # Token invalid, but don't raise exception - just return None
        print(f"⚠️ JWT decode error: {e}, returning None (anonymous user)")
        return None

# Optional: Keep a strict version if needed elsewhere
async def get_current_user_strict(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    Strict version that requires authentication.
    Use this for endpoints that absolutely need login.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    user = await get_current_user(token)
    if user is None:
        raise credentials_exception
    return user

# ==================== Authentication Interface Functions ====================
async def register_user(user: UserCreate):
    print("="*50)
    print(f"New User Registered: {user.email}")
    print(f"Database Name: {DB_NAME}")
    
    # Check if MongoDB is connected
    if mongodb.database is None:
        error_msg = "Database not connected"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Test MongoDB connection
    try:
        # Send ping to test connection
        await mongodb.client.admin.command('ping')
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    
    # Check if email already exists
    existing_user = await get_user(user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user document
    new_user = {
        "username": user.username,
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "created_at": datetime.utcnow()
    }
    
    # Insert into MongoDB
    try:
        users_collection = get_users_collection()
        result = await users_collection.insert_one(new_user)
        print(f"✅ User inserted successfully, ID: {result.inserted_id}")
        
        return {
            "code": 200, 
            "msg": "Registration successful", 
            "username": user.username,
            "email": user.email,
            "id": str(result.inserted_id)
        }
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database insert error: {str(e)}")

async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user(form_data.username)  # form_data.username is email
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}