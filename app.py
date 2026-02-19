import io
import base64
import os
import uuid
from datetime import datetime
from typing import Dict

import torch
from fastapi import FastAPI, UploadFile, File, Form, Depends, Request
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
    
@app.get("/gallery/images", summary="Get all gallery images")
async def get_gallery_images():
    """Get all published gallery images"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get all images from gallery, sorted by newest first
        cursor = db.gallery.find().sort("created_at", -1).limit(50)
        images = await cursor.to_list(length=50)
        
        # Convert ObjectId to string for JSON serialization
        for img in images:
            img["_id"] = str(img["_id"])
        
        print(f"Found {len(images)} gallery images")  # Debug log
        
        return {"code": 200, "images": images}
        
    except Exception as e:
        print(f"Error loading gallery: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.post("/gallery/publish", summary="Publish image to gallery")
async def publish_to_gallery(
    request: Request,
    current_user: UserInDB = Depends(get_current_user)
):
    """Publish a transformed image to the public gallery"""
    try:
        data = await request.json()
        image_base64 = data.get('image_base64')
        style = data.get('style')
        title = data.get('title', 'Untitled')
        description = data.get('description', '')
        
        if not image_base64 or not style:
            return JSONResponse({"code": 400, "error": "Missing image or style"}, status_code=400)
        
        # Generate filename
        import uuid
        from datetime import datetime
        import base64
        from PIL import Image
        import io
        
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gallery_{current_user.id}_{timestamp}_{file_id}_{style}.png"
        filepath = f"uploads/gallery/{filename}"
        
        # Ensure directory exists
        os.makedirs("uploads/gallery", exist_ok=True)
        
        # Save image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image.save(filepath, "PNG")
        
        # Save to database
        db = get_db()
        gallery_record = {
            "user_id": current_user.id,
            "user_email": current_user.email,
            "username": current_user.username,
            "style": style,
            "title": title,
            "description": description,
            "image_path": filepath,
            "filename": filename,
            "likes": 0,
            "views": 0,
            "created_at": datetime.utcnow()
        }
        
        result = await db.gallery.insert_one(gallery_record)
        
        return {
            "code": 200,
            "message": "Published to gallery successfully",
            "gallery_id": str(result.inserted_id)
        }
        
    except Exception as e:
        print(f"Publish error: {e}")
        import traceback
        traceback.print_exc()
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
        # Validate style
        if style not in models:
            return JSONResponse({"error": "Invalid style"}, status_code=400)

        # Read original image bytes for potential saving
        original_bytes = await content.read()
        await content.seek(0)  # Reset file pointer for load_image
        
        # Process image through style transfer model
        content_tensor = load_image(content)

        with torch.no_grad():
            output = models[style](content_tensor)

        # Debug: print output range
        print(f"Output range for {style}: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")

        # Convert tensor to base64 string
        b64 = tensor_to_base64(output)
        
        # ========== Save to history ONLY for authenticated users ==========
        if current_user:
            try:
                # Generate unique identifiers for files
                file_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save original image
                original_filename = f"{current_user.id}_{timestamp}_{file_id}_original.jpg"
                original_path = f"uploads/originals/{original_filename}"
                
                with open(original_path, "wb") as f:
                    f.write(original_bytes)
                
                # Save transformed result image
                result_filename = f"{current_user.id}_{timestamp}_{file_id}_{style}.png"
                result_path = f"uploads/transformed/{result_filename}"
                
                # Convert base64 back to image and save
                image_data = base64.b64decode(b64)
                result_image = Image.open(io.BytesIO(image_data))
                result_image.save(result_path, "PNG")
                
                # Get database connection
                db = get_db()
                
                # Create history record for database
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
                
                # Insert into MongoDB history collection
                await db.history.insert_one(history_record)
                print(f"✅ History saved for authenticated user: {current_user.email}")
                
            except Exception as e:
                # Non-critical error - don't fail the request, just log it
                print(f"⚠️ Failed to save history (non-critical): {e}")
        else:
            # Anonymous user - skip database save
            print(f"ℹ️ Anonymous user request - history not saved")
        # ================================================================

        # Return transformed image to client
        return {"image_base64": b64}

    except Exception as e:
        # Log error and return 500 response
        print(f"❌ Error during image transformation: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================
# Serve uploaded files (for development)
# =============================
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# =============================
# Favorite routes
# =============================
@app.post("/favorites/add", summary="Add image to favorites")
async def add_to_favorites(
    request: Request,
    current_user: UserInDB = Depends(get_current_user)
):
    """Add a transformed image to user's favorites"""
    try:
        data = await request.json()
        image_path = data.get('image_path')
        style = data.get('style')
        
        if not image_path or not style:
            return JSONResponse({"code": 400, "error": "Missing image_path or style"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Check if already in favorites
        existing = await db.favorites.find_one({
            "user_id": current_user.id,
            "image_path": image_path
        })
        
        if existing:
            return JSONResponse({"code": 400, "error": "Already in favorites"}, status_code=400)
        
        # Add to favorites
        favorite_record = {
            "user_id": current_user.id,
            "user_email": current_user.email,
            "username": current_user.username,
            "image_path": image_path,
            "style": style,
            "original_filename": data.get('original_filename'),
            "result_filename": data.get('result_filename'),
            "created_at": datetime.utcnow()
        }
        
        result = await db.favorites.insert_one(favorite_record)
        
        return {
            "code": 200,
            "message": "Added to favorites",
            "favorite_id": str(result.inserted_id)
        }
        
    except Exception as e:
        print(f"Error adding to favorites: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.delete("/favorites/remove", summary="Remove from favorites")
async def remove_from_favorites(
    request: Request,
    current_user: UserInDB = Depends(get_current_user)
):
    """Remove an image from favorites"""
    try:
        data = await request.json()
        image_path = data.get('image_path')
        
        if not image_path:
            return JSONResponse({"code": 400, "error": "Missing image_path"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Remove from favorites
        result = await db.favorites.delete_one({
            "user_id": current_user.id,
            "image_path": image_path
        })
        
        if result.deleted_count > 0:
            return {"code": 200, "message": "Removed from favorites"}
        else:
            return JSONResponse({"code": 404, "error": "Favorite not found"}, status_code=404)
        
    except Exception as e:
        print(f"Error removing from favorites: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/favorites", summary="Get user's favorites")
async def get_favorites(current_user: UserInDB = Depends(get_current_user)):
    """Get user's favorite images"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get favorites from database
        cursor = db.favorites.find({"user_id": current_user.id}).sort("created_at", -1)
        favorites = await cursor.to_list(length=100)
        
        # Convert ObjectId to string
        for item in favorites:
            item["_id"] = str(item["_id"])
        
        return {"code": 200, "favorites": favorites}
        
    except Exception as e:
        print(f"Error loading favorites: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/favorites/check", summary="Check if image is favorited")
async def check_favorite(
    image_path: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Check if an image is in user's favorites"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        favorite = await db.favorites.find_one({
            "user_id": current_user.id,
            "image_path": image_path
        })
        
        return {"code": 200, "is_favorite": favorite is not None}
        
    except Exception as e:
        print(f"Error checking favorite: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

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
    """Convert model output tensor to base64 image string"""
    tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Normalize to [0, 1] range
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    
    # Convert to PIL Image
    img = T.ToPILImage()(tensor.cpu())

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()

# =============================
# Style Transfer Network (PyTorch official)
# =============================
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

# =============================
# Load models (CPU only) with version compatibility
# =============================
STYLE_NAMES = ["sketch", "anime", "ink"]
models: Dict[str, torch.nn.Module] = {}

print("Loading PyTorch style models...")

def filter_state_dict(state_dict):
    """Remove running_mean and running_var keys from state_dict"""
    filtered = {}
    for key, value in state_dict.items():
        # Skip keys that end with running_mean or running_var
        if not (key.endswith('running_mean') or key.endswith('running_var')):
            filtered[key] = value
    return filtered

for name in STYLE_NAMES:
    # Use TransformerNet instead of SimpleStyleNet
    model = TransformerNet()

    model_path = f"models/{name}.pt"
    if os.path.exists(model_path):
        try:
            # Load pretrained weights
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Try loading with strict=True first
            model.load_state_dict(state_dict)
            print(f"✅ Loaded trained model: {name}")
            
        except RuntimeError as e:
            # If strict loading fails, filter the state_dict and try with strict=False
            print(f"⚠️ Version mismatch for {name}, filtering state_dict...")
            try:
                # Filter out problematic keys
                filtered_dict = filter_state_dict(state_dict)
                model.load_state_dict(filtered_dict, strict=False)
                print(f"✅ Loaded trained model: {name} (with filtered state_dict)")
                print(f"   Removed {len(state_dict) - len(filtered_dict)} unexpected keys")
            except Exception as e2:
                print(f"❌ Failed to load model {name}: {e2}")
                print(f"Using untrained demo model for: {name}")
    else:
        print(f"⚠️ Model file not found: {model_path}, using random weights")
        print(f"Using untrained demo model for: {name}")

    model.eval()
    models[name] = model

print("All models ready!\n")