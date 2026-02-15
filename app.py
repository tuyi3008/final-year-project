import io
import base64
import os
import uuid
from datetime import datetime
from typing import Dict

import torch
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as T
from dotenv import load_dotenv

# Import database connection functions
from database import connect_to_mongo, close_mongo_connection, get_db

# Import authentication functions and models
from auth import (
    register_user,
    login_for_access_token,
    get_current_user,
    UserCreate,
    UserInDB,
    OAuth2PasswordRequestForm
)

# Load environment variables from .env file
load_dotenv()

# =============================
# FastAPI init
# =============================
app = FastAPI(title="Multi-Style Transfer API (PyTorch)")

# =============================
# CORS middleware configuration
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if not exists
os.makedirs("uploads/originals", exist_ok=True)
os.makedirs("uploads/transformed", exist_ok=True)

# =============================
# Database lifecycle management
# =============================
@app.on_event("startup")
async def startup_event():
    """Connect to MongoDB when the application starts"""
    await connect_to_mongo()
    print("✅ Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection when the application shuts down"""
    await close_mongo_connection()
    print("✅ Application shutdown complete")

# =============================
# Serve frontend
# =============================
app.mount("/static", StaticFiles(directory="public"), name="static")

# =============================
# Authentication routes (public)
# =============================
@app.post("/register", summary="User registration")
async def register(user: UserCreate):
    return await register_user(user)

@app.post("/login", summary="User login to get token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return await login_for_access_token(form_data)

@app.get("/profile", summary="Get current user info (login required)")
async def get_profile(current_user: UserInDB = Depends(get_current_user)):
    return {
        "code": 200,
        "username": current_user.username,
        "email": current_user.email,
        "id": current_user.id,
        "msg": "Logged in"
    }

# =============================
# Page routes
# =============================
@app.get("/")
@app.get("/index")
@app.get("/index.html")
async def index():
    return FileResponse("public/index.html")

@app.get("/gallery")
@app.get("/gallery.html")
async def gallery():
    return FileResponse("public/gallery.html")

@app.get("/community")
@app.get("/community.html")
async def community():
    return FileResponse("public/community.html")

@app.get("/profile")
@app.get("/profile.html")
async def profile():
    return FileResponse("public/profile.html")

# =============================
# History routes
# =============================
@app.get("/history", summary="Get user transformation history")
async def get_user_history(current_user: UserInDB = Depends(get_current_user)):
    """Get user's transformation history"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get history from database, sorted by created_at descending, limit 20
        cursor = db.history.find({"user_id": current_user.id}).sort("created_at", -1).limit(20)
        history = await cursor.to_list(length=20)
        
        # Convert ObjectId to string for JSON serialization
        for item in history:
            item["_id"] = str(item["_id"])
        
        return {"code": 200, "history": history}
    
    except Exception as e:
        print(f"Error loading history: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

# =============================
# Public interfaces (no login required)
# =============================
@app.post("/stylize/")
async def stylize(
    content: UploadFile = File(...),
    style: str = Form(...),
    current_user: UserInDB = Depends(get_current_user)  # Optional, None if not logged in
):
    try:
        if style not in models:
            return JSONResponse({"error": "Invalid style"}, status_code=400)

        # Read original image bytes for saving
        original_bytes = await content.read()
        await content.seek(0)  # Reset file pointer for load_image
        
        # Process image
        content_tensor = load_image(content)

        with torch.no_grad():
            output = models[style](content_tensor)

        b64 = tensor_to_base64(output)
        
        # ========== Save history if user is logged in ==========
        if current_user:
            try:
                # Generate unique filenames
                file_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Original image
                original_filename = f"{current_user.id}_{timestamp}_{file_id}_original.jpg"
                original_path = f"uploads/originals/{original_filename}"
                
                # Result image
                result_filename = f"{current_user.id}_{timestamp}_{file_id}_{style}.png"
                result_path = f"uploads/transformed/{result_filename}"
                
                # Save original image
                with open(original_path, "wb") as f:
                    f.write(original_bytes)
                
                # Save result image (convert base64 back to image)
                image_data = base64.b64decode(b64)
                result_image = Image.open(io.BytesIO(image_data))
                result_image.save(result_path, "PNG")
                
                # Get database connection
                db = get_db()
                
                # Create history record
                history_record = {
                    "user_id": current_user.id,
                    "user_email": current_user.email,
                    "username": current_user.username,
                    "style": style,
                    "original_image_path": original_path,
                    "result_image_path": result_path,
                    "original_filename": original_filename,
                    "result_filename": result_filename,
                    "file_size": len(original_bytes),
                    "created_at": datetime.utcnow()
                }
                
                # Insert into history collection
                await db.history.insert_one(history_record)
                print(f"✅ History saved for user: {current_user.email}")
                
            except Exception as e:
                print(f"⚠️ History save failed (non-critical): {e}")
        # =======================================================

        return {"image_base64": b64}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================
# Serve uploaded files (for development)
# =============================
from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# =============================
# Image helpers
# =============================
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

def load_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor

def tensor_to_base64(tensor: torch.Tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)
    img = T.ToPILImage()(tensor.cpu())

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()

# =============================
# Dummy lightweight style models
# =============================
class SimpleStyleNet(torch.nn.Module):
    """Very small CNN just for demo / CPU use."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# =============================
# Load models (CPU only)
# =============================
STYLE_NAMES = ["sketch", "anime", "ink"]
models: Dict[str, torch.nn.Module] = {}

print("Loading PyTorch style models...")

for name in STYLE_NAMES:
    model = SimpleStyleNet()

    model_path = f"models/{name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded trained model: {name}")
    else:
        print(f"Using untrained demo model for: {name}")

    model.eval()
    models[name] = model

print("All models ready!\n")