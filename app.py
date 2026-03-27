import io
import base64
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, Depends, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as T
from dotenv import load_dotenv
from bson import ObjectId
from model import Generator, UNet, TransformerNet, UkiyoEGenerator

# Import database connection functions
from database import connect_to_mongo, close_mongo_connection, get_db

# Import authentication functions and models
from auth import (
    register_user,
    login_for_access_token,
    get_current_user,
    get_current_user_strict,
    UserCreate,
    UserInDB,
    OAuth2PasswordRequestForm
)

# Load environment variables from .env file
load_dotenv()


# =============================
# Saliency-based Cropper Class - ULTRA FAST VERSION
# =============================
class SaliencyCropper:
    """
    Fast cropping based on face detection and center crop
    Removed slow saliency detection for speed
    """
    
    def __init__(self):
        # Initialize face detector only (remove saliency)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Statistics
        self.stats = {
            'total': 0,
            'face_detected': 0,
            'center_used': 0,
            'fallback_used': 0
        }
    
    def process(self, 
                pil_image: Image.Image, 
                target_ratio: str = "1:1") -> Tuple[Image.Image, Dict]:
        """
        Ultra fast processing - no saliency detection
        """
        start_time = time.time()
        self.stats['total'] += 1
        
        # Record processing info
        info = {
            'method': 'unknown',
            'face_count': 0,
            'processing_time': 0,
            'crop_box': None,
            'original_size': pil_image.size
        }
        
        try:
            # Convert format for face detection
            img_np = np.array(pil_image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            
            # Parse target ratio
            target_w, target_h = map(int, target_ratio.split(':'))
            target_ratio_float = target_w / target_h
            
            # Fast face detection (less sensitive for speed)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,        # Increased for fewer false positives
                minNeighbors=8,          # Increased for fewer false positives
                minSize=(60, 60)         # Increased to ignore small false detections
            )
            
            info['face_count'] = len(faces)
            
            # Decision: use face detection or center crop
            if len(faces) > 0:
                # Use face-protected crop
                crop_box = self._face_protected_crop(
                    faces, (w, h), target_ratio_float
                )
                info['method'] = 'face_protected'
                self.stats['face_detected'] += 1
            else:
                # Use center crop
                crop_box = self._get_center_crop_box(
                    (w, h), target_ratio_float
                )
                info['method'] = 'center_crop'
                self.stats['center_used'] += 1
            
            # FIX: Convert numpy ints to Python ints for JSON serialization
            if crop_box:
                info['crop_box'] = tuple(int(x) for x in crop_box)
            
            # Execute crop
            x1, y1, x2, y2 = crop_box
            cropped = pil_image.crop((x1, y1, x2, y2))
            
            info['processing_time'] = time.time() - start_time
            
            return cropped, info
            
        except Exception as e:
            print(f"Processing error: {e}, using fallback")
            info['method'] = 'error_fallback'
            self.stats['fallback_used'] += 1
            # Simple center crop fallback
            result = self._center_crop(pil_image, target_ratio_float)
            info['processing_time'] = time.time() - start_time
            return result, info
    
    def _face_protected_crop(self, 
                            faces: List, 
                            img_size: Tuple[int, int], 
                            target_ratio: float) -> Tuple[int, int, int, int]:
        """
        Face-protected cropping - returns Python ints
        """
        w, h = img_size
        
        # Calculate bounding box containing all faces
        min_x = min(x for (x, y, fw, fh) in faces)
        max_x = max(x + fw for (x, y, fw, fh) in faces)
        min_y = min(y for (x, y, fw, fh) in faces)
        max_y = max(y + fh for (x, y, fw, fh) in faces)
        
        # Expand 20% as buffer
        padding = 0.2
        width_pad = int((max_x - min_x) * padding)
        height_pad = int((max_y - min_y) * padding)
        
        face_center_x = (min_x + max_x) // 2
        face_center_y = (min_y + max_y) // 2
        
        # Adjust according to target ratio
        if w / h > target_ratio:
            crop_w = int(h * target_ratio)
            left = max(0, min(face_center_x - crop_w // 2, w - crop_w))
            # FIX: Ensure all values are Python ints
            return (int(left), 0, int(left + crop_w), int(h))
        else:
            crop_h = int(w / target_ratio)
            top = max(0, min(face_center_y - crop_h // 2, h - crop_h))
            # FIX: Ensure all values are Python ints
            return (0, int(top), int(w), int(top + crop_h))
    
    def _get_center_crop_box(self, 
                            img_size: Tuple[int, int], 
                            target_ratio: float) -> Tuple[int, int, int, int]:
        """Get center crop box - returns Python ints"""
        w, h = img_size
        
        if w / h > target_ratio:
            crop_w = int(h * target_ratio)
            left = (w - crop_w) // 2
            # FIX: Ensure all values are Python ints
            return (int(left), 0, int(left + crop_w), int(h))
        else:
            crop_h = int(w / target_ratio)
            top = (h - crop_h) // 2
            # FIX: Ensure all values are Python ints
            return (0, int(top), int(w), int(top + crop_h))
    
    def _center_crop(self, 
                    pil_image: Image.Image, 
                    target_ratio: float) -> Image.Image:
        """Center crop - returns cropped image"""
        w, h = pil_image.size
        
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            return pil_image.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            return pil_image.crop((0, top, w, top + new_h))
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return self.stats


# =============================
# Initialize SaliencyCropper
# =============================
cropper = SaliencyCropper()
    

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
os.makedirs("uploads/gallery", exist_ok=True)
os.makedirs("uploads/challenges", exist_ok=True)
os.makedirs("uploads/avatars", exist_ok=True)
os.makedirs("uploads/album_covers", exist_ok=True)
os.makedirs("uploads/album_photos", exist_ok=True)

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
async def get_profile(current_user: UserInDB = Depends(get_current_user_strict)):
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
# Challenge API Routes
# =============================
@app.post("/api/challenge/join", summary="Join current weekly challenge")
async def join_challenge(
    request: Request,
    current_user: UserInDB = Depends(get_current_user)
):
    """Join the current weekly challenge"""
    try:
        data = await request.json()
        challenge_id = data.get('challenge_id')
        
        if not challenge_id:
            return JSONResponse({"code": 400, "error": "Missing challenge_id"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Check if challenge exists
        challenge = await db.challenges.find_one({"_id": ObjectId(challenge_id)})
        if not challenge:
            return JSONResponse({"code": 404, "error": "Challenge not found"}, status_code=404)
        
        # Check if user already joined
        existing = await db.challenge_participants.find_one({
            "user_id": current_user.id,
            "challenge_id": challenge_id
        })
        
        if existing:
            return JSONResponse({"code": 400, "error": "Already joined this challenge"}, status_code=400)
        
        # Add user to participants
        participant = {
            "user_id": current_user.id,
            "user_email": current_user.email,
            "username": current_user.username,
            "challenge_id": challenge_id,
            "joined_at": datetime.utcnow()
        }
        
        await db.challenge_participants.insert_one(participant)
        
        # Update participant count
        await db.challenges.update_one(
            {"_id": ObjectId(challenge_id)},
            {"$inc": {"participants": 1}}
        )

        xp_reward = 20
        await db.user_xp.insert_one({
            "user_id": current_user.id,
            "amount": xp_reward,
            "source": "join_challenge",
            "challenge_id": challenge_id,
            "created_at": datetime.utcnow()
        })
        
        # Update user's total XP in users collection
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"total_xp": xp_reward}}
        )
        
        return {
            "code": 200,
            "message": "Successfully joined the challenge",
            "xp_reward": xp_reward
        }
        
    except Exception as e:
        print(f"Error joining challenge: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/api/challenge/check-joined", summary="Check if user joined challenge")
async def check_challenge_joined(
    challenge_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Check if the current user has joined a specific challenge"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Check if user is in participants
        participant = await db.challenge_participants.find_one({
            "user_id": current_user.id,
            "challenge_id": challenge_id
        })
        
        return {
            "code": 200,
            "joined": participant is not None
        }
        
    except Exception as e:
        print(f"Error checking join status: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/api/challenge/current", summary="Get current weekly challenge")
async def get_current_challenge():
    """Get the current active weekly challenge"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Find current active challenge (start_date <= now <= end_date)
        now = datetime.utcnow()
        challenge = await db.challenges.find_one({
            "start_date": {"$lte": now},
            "end_date": {"$gte": now},
            "active": True
        })
        
        if not challenge:
            # If no active challenge, create a default one
            challenge = await create_default_challenge(db)
        
        # Calculate days left
        if challenge and challenge.get('end_date'):
            days_left = (challenge['end_date'] - now).days
            challenge['days_left'] = max(0, days_left)
        
        # Convert ObjectId to string
        if challenge and '_id' in challenge:
            challenge['_id'] = str(challenge['_id'])
        
        # Get winners for this challenge (top 3 submissions by likes)
        winners = await get_challenge_winners(db, challenge['_id'] if challenge else None)
        if winners:
            challenge['winners'] = winners
        
        return {"code": 200, "challenge": challenge}
        
    except Exception as e:
        print(f"Error loading challenge: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.post("/api/challenge/submit", summary="Submit to weekly challenge")
async def submit_to_challenge(
    request: Request,
    current_user: UserInDB = Depends(get_current_user)
):
    """Submit an image to the current weekly challenge"""
    try:
        data = await request.json()
        image_path = data.get('image_path')
        style = data.get('style')
        description = data.get('description', '')
        challenge_id = data.get('challenge_id')
        
        if not image_path or not style:
            return JSONResponse({"code": 400, "error": "Missing image_path or style"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get current challenge if no challenge_id provided
        if not challenge_id:
            now = datetime.utcnow()
            current_challenge = await db.challenges.find_one({
                "start_date": {"$lte": now},
                "end_date": {"$gte": now},
                "active": True
            })
            if current_challenge:
                challenge_id = str(current_challenge['_id'])
            else:
                return JSONResponse({"code": 400, "error": "No active challenge"}, status_code=400)
        
        # Check if user already submitted to this challenge
        existing = await db.challenge_submissions.find_one({
            "user_id": current_user.id,
            "challenge_id": challenge_id
        })
        
        if existing:
            return JSONResponse({"code": 400, "error": "You have already submitted to this challenge"}, status_code=400)
        
        # Create submission record
        submission = {
            "user_id": current_user.id,
            "user_email": current_user.email,
            "username": current_user.username,
            "challenge_id": challenge_id,
            "image_path": image_path,
            "style": style,
            "description": description,
            "likes": 0,
            "views": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.challenge_submissions.insert_one(submission)
        
        # Update challenge submission count
        await db.challenges.update_one(
            {"_id": ObjectId(challenge_id)},
            {"$inc": {"submissions": 1, "participants": 1}}
        )

        xp_reward = 50
        await db.user_xp.insert_one({
            "user_id": current_user.id,
            "amount": xp_reward,
            "source": "submit_challenge",
            "challenge_id": challenge_id,
            "submission_id": str(result.inserted_id),
            "created_at": datetime.utcnow()
        })
        
        # Update user's total XP in users collection
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"total_xp": xp_reward}}
        )
        
        return {
            "code": 200,
            "message": "Successfully submitted to challenge",
            "submission_id": str(result.inserted_id),
            "xp_reward": xp_reward
        }
        
    except Exception as e:
        print(f"Error submitting to challenge: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/api/challenge/submissions", summary="Get challenge submissions")
async def get_challenge_submissions(
    challenge_id: Optional[str] = None,
    page: int = 1,
    limit: int = 8,
    style: Optional[str] = None,
    sort_by: str = "popular"
):
    """Get submissions for a challenge with pagination and filtering"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Build query
        query = {}
        if challenge_id:
            query["challenge_id"] = challenge_id
        if style and style != "all":
            query["style"] = style
        
        # Determine sort order
        sort_field = "likes" if sort_by == "popular" else "created_at"
        sort_order = -1  # descending
        
        # Get total count
        total = await db.challenge_submissions.count_documents(query)
        
        # Get paginated results
        skip = (page - 1) * limit
        cursor = db.challenge_submissions.find(query).sort(sort_field, sort_order).skip(skip).limit(limit)
        submissions = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for sub in submissions:
            sub["_id"] = str(sub["_id"])
        
        return {
            "code": 200,
            "submissions": submissions,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": total > (page * limit)
        }
        
    except Exception as e:
        print(f"Error loading submissions: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.post("/api/submission/{submission_id}/like", summary="Like a submission")
async def like_submission(
    submission_id: str,
    current_user: Optional[UserInDB] = Depends(get_current_user)
):
    """Like or unlike a challenge submission"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Check if submission exists
        submission = await db.challenge_submissions.find_one({"_id": ObjectId(submission_id)})
        if not submission:
            return JSONResponse({"code": 404, "error": "Submission not found"}, status_code=404)
        
        # If user is logged in, track unique likes
        if current_user:
            # Check if user already liked
            like = await db.submission_likes.find_one({
                "submission_id": submission_id,
                "user_id": current_user.id
            })
            
            if like:
                # Unlike
                await db.submission_likes.delete_one({"_id": like["_id"]})
                await db.challenge_submissions.update_one(
                    {"_id": ObjectId(submission_id)},
                    {"$inc": {"likes": -1}}
                )
                return {"code": 200, "message": "Unliked", "liked": False}
            else:
                # Like
                await db.submission_likes.insert_one({
                    "submission_id": submission_id,
                    "user_id": current_user.id,
                    "created_at": datetime.utcnow()
                })
                await db.challenge_submissions.update_one(
                    {"_id": ObjectId(submission_id)},
                    {"$inc": {"likes": 1}}
                )

                if str(submission.get('user_id')) != current_user.id:
                    await db.users.update_one(
                        {"_id": ObjectId(submission['user_id'])},
                        {"$inc": {"total_xp": 2}}
                    )
                    
                    await db.user_xp.insert_one({
                        "user_id": submission['user_id'],
                        "amount": 2,
                        "source": "like_received",
                        "submission_id": submission_id,
                        "created_at": datetime.utcnow()
                    })
                
                return {"code": 200, "message": "Liked", "liked": True}
        else:
            # Anonymous like - just increment counter
            await db.challenge_submissions.update_one(
                {"_id": ObjectId(submission_id)},
                {"$inc": {"likes": 1}}
            )
            return {"code": 200, "message": "Liked", "liked": True}
        
    except Exception as e:
        print(f"Error liking submission: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

# =============================
# Helper functions for challenges
# =============================
async def create_default_challenge(db):
    """Create a default weekly challenge if none exists"""
    now = datetime.utcnow()
    end_date = now + timedelta(days=7)
    
    default_challenge = {
        "theme": "Dreamscapes",
        "description": "Create surreal and dreamlike landscapes. Let your imagination run wild and transform ordinary scenes into extraordinary dreamscapes using our AI styles.",
        "start_date": now,
        "end_date": end_date,
        "active": True,
        "submissions": 0,
        "participants": 0,
        "created_at": now,
        "updated_at": now
    }
    
    result = await db.challenges.insert_one(default_challenge)
    default_challenge["_id"] = result.inserted_id
    return default_challenge

async def get_challenge_winners(db, challenge_id):
    """Get top 3 submissions for a challenge by likes"""
    if not challenge_id:
        return None
    
    cursor = db.challenge_submissions.find(
        {"challenge_id": challenge_id}
    ).sort("likes", -1).limit(3)
    
    winners = await cursor.to_list(length=3)
    
    return [{
        "username": w.get("username", "Anonymous"),
        "likes": w.get("likes", 0)
    } for w in winners]

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

        xp_reward = 15
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"total_xp": xp_reward}}
        )
        
        await db.user_xp.insert_one({
            "user_id": current_user.id,
            "amount": xp_reward,
            "source": "publish_gallery",
            "gallery_id": str(result.inserted_id),
            "created_at": datetime.utcnow()
        })
        
        return {
            "code": 200,
            "message": "Published to gallery successfully",
            "gallery_id": str(result.inserted_id),
            "xp_reward": xp_reward
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
    aspect_ratio: str = Form("1:1"),
    current_user: Optional[UserInDB] = Depends(get_current_user)
):
    start_time = time.time()
    
    print(f"\n========== NEW REQUEST ==========")
    print(f"Style: {style}")
    print(f"Target ratio: {aspect_ratio}")

    try:
        # Validate style
        valid_styles = ["sketch", "anime", "ink", "hayao", "shinkai", "paprika", "cyberpunk", "ukiyoe"]
        if style not in valid_styles:
            return JSONResponse({"error": f"Invalid style. Choose from: {valid_styles}"}, status_code=400)
        
        # Validate aspect ratio
        valid_ratios = ["auto", "1:1", "2:3", "3:2", "4:3", "3:4", "16:9", "9:16"]
        if aspect_ratio not in valid_ratios:
            return JSONResponse({"error": f"Invalid aspect ratio. Choose from: {valid_ratios}"}, status_code=400)

        # Read original image bytes for potential saving
        original_bytes = await content.read()
        await content.seek(0)
        
        # Load image with PIL
        pil_image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        
        # ===== Calculate display size =====
        max_display_size = 512
        
        if aspect_ratio == "auto":
            print("👉 Using auto mode (direct square)")
            processed_image = pil_image.resize((256, 256), Image.LANCZOS)
            model_input = processed_image
            display_size = (256, 256)
            crop_info = {'method': 'auto_resize', 'original_size': pil_image.size}
        else:
            # Parse user selected ratio
            target_w, target_h = map(int, aspect_ratio.split(':'))
            
            # Calculate display dimensions
            if target_w > target_h:
                display_w = max_display_size
                display_h = int(max_display_size * target_h / target_w)
            else:
                display_h = max_display_size
                display_w = int(max_display_size * target_w / target_h)
            
            display_size = (display_w, display_h)
            print(f"👉 Display size: {display_w}x{display_h}")
            print(f"👉 Using saliency cropping for ratio: {aspect_ratio}")
            
            # Use saliency cropping
            crop_start = time.time()
            cropped_image, crop_info = cropper.process(pil_image, aspect_ratio)
            
            print(f"✅ cropper.process() completed in: {time.time() - crop_start:.2f}s")
            print(f"   Cropping method: {crop_info.get('method')}")
            print(f"   Faces detected: {crop_info.get('face_count')}")
            
            # Create model input (256x256)
            model_input = cropped_image.resize((256, 256), Image.LANCZOS)
        
        print(f"Total processing time: {time.time() - start_time:.2f}s")
        
        # Convert to tensor
        content_tensor = image_to_tensor(model_input)

        # Run style transfer model
        with torch.no_grad():
            output = models[style](content_tensor)

        # Debug: print output range
        print(f"Output range for {style}: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")

        # Convert tensor to PIL image
        output_image = tensor_to_pil(output, style)
        
        # ===== KEY FIX: Resize output to correct display size =====
        if aspect_ratio != "auto":
            output_image = output_image.resize(display_size, Image.LANCZOS)
        
        # Convert to base64 for frontend
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # ========== Save to history ==========
        xp_reward = 0
        if current_user:
            try:
                # Save files...
                file_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save original image
                original_filename = f"{current_user.id}_{timestamp}_{file_id}_original.jpg"
                original_path = f"uploads/originals/{original_filename}"
                with open(original_path, "wb") as f:
                    f.write(original_bytes)
                
                # Save result image (at display size)
                result_filename = f"{current_user.id}_{timestamp}_{file_id}_{style}_{aspect_ratio}.png"
                result_path = f"uploads/transformed/{result_filename}"
                output_image.save(result_path, "PNG")
                
                # Database record
                db = get_db()
                history_record = {
                    "user_id": current_user.id,
                    "user_email": current_user.email,
                    "username": current_user.username,
                    "style": style,
                    "aspect_ratio": aspect_ratio,
                    "crop_method": crop_info.get('method'),
                    "display_size": display_size,
                    "original_image_path": original_path,
                    "result_image_path": result_path,
                    "original_filename": original_filename,
                    "result_filename": result_filename,
                    "file_size": len(original_bytes),
                    "created_at": datetime.utcnow()
                }
                await db.history.insert_one(history_record)
                
                # Award XP
                xp_reward = 5
                await db.users.update_one(
                    {"_id": ObjectId(current_user.id)},
                    {"$inc": {"total_xp": xp_reward}}
                )
                
            except Exception as e:
                print(f"⚠️ Failed to save history: {e}")
        
        print("=================================\n")

        # Return transformed image
        return {
            "image_base64": b64,
            "xp_reward": xp_reward if current_user else 0,
            "aspect_ratio": aspect_ratio,
            "display_size": display_size,
            "crop_info": crop_info
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================
# Add this helper function
# =============================
def tensor_to_pil(tensor: torch.Tensor, style: str) -> Image.Image:
    """Convert model output tensor to PIL image"""
    tensor = tensor.squeeze(0).cpu()
    
    # Handle different model types
    if style == "cyberpunk":
        tensor = (tensor + 1) / 2  # Tanh output [-1, 1] -> [0, 1]
    elif style == "sketch":
        tensor = (tensor + 1) / 2
        # Convert to grayscale for sketch
        gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        tensor = torch.stack([gray, gray, gray])
    elif style == "ukiyoe":
        tensor = (tensor + 1) / 2  # Tanh output [-1, 1] -> [0, 1]
    else:
        # Normalize to [0, 1]
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_min < 0 or tensor_max > 1:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    
    tensor = torch.clamp(tensor, 0, 1)
    return T.ToPILImage()(tensor)

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

        xp_reward = 3
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"total_xp": xp_reward}}
        )
        
        await db.user_xp.insert_one({
            "user_id": current_user.id,
            "amount": xp_reward,
            "source": "add_favorite",
            "favorite_id": str(result.inserted_id),
            "created_at": datetime.utcnow()
        })
        
        return {
            "code": 200,
            "message": "Added to favorites",
            "favorite_id": str(result.inserted_id),
            "xp_reward": xp_reward
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
# Album API Routes
# =============================
@app.get("/api/albums", summary="Get user's albums")
async def get_albums(current_user: UserInDB = Depends(get_current_user_strict)):
    """Get all albums for the current user"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get albums from database
        cursor = db.albums.find({"user_id": current_user.id}).sort("created_at", -1)
        albums = await cursor.to_list(length=100)
        
        # Convert ObjectId to string and add photo count
        for album in albums:
            album["_id"] = str(album["_id"])
            album["id"] = album["_id"]
            
            # Count photos in album
            photo_count = await db.photos.count_documents({"album_id": album["_id"]})
            album["photo_count"] = photo_count
        
        return {"code": 200, "albums": albums}
        
    except Exception as e:
        print(f"Error loading albums: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/api/albums/{album_id}", summary="Get single album")
async def get_album(
    album_id: str,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Get a specific album with its photos"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get album
        album = await db.albums.find_one({"_id": ObjectId(album_id), "user_id": current_user.id})
        if not album:
            return JSONResponse({"code": 404, "error": "Album not found"}, status_code=404)
        
        album["_id"] = str(album["_id"])
        album["id"] = album["_id"]
        
        # Get photos in this album
        cursor = db.photos.find({"album_id": album_id}).sort("uploaded_at", -1)
        photos = await cursor.to_list(length=100)
        
        for photo in photos:
            photo["_id"] = str(photo["_id"])
            photo["id"] = photo["_id"]
        
        album["photos"] = photos
        album["photo_count"] = len(photos)
        
        return {"code": 200, "album": album}
        
    except Exception as e:
        print(f"Error loading album: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.post("/api/albums", summary="Create new album")
async def create_album(
    request: Request,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Create a new album"""
    try:
        data = await request.form()
        name = data.get('name')
        description = data.get('description', '')
        
        if not name:
            return JSONResponse({"code": 400, "error": "Album name is required"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Handle cover image
        cover_file = data.get('cover')
        cover_path = None
        if cover_file and cover_file.filename:
            # Save cover image
            file_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"album_cover_{current_user.id}_{timestamp}_{file_id}.jpg"
            filepath = f"uploads/album_covers/{filename}"
            
            os.makedirs("uploads/album_covers", exist_ok=True)
            
            content = await cover_file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            
            cover_path = filepath
        
        # Create album record
        album = {
            "user_id": current_user.id,
            "user_email": current_user.email,
            "username": current_user.username,
            "name": name,
            "description": description,
            "cover_image": cover_path,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.albums.insert_one(album)

        xp_reward = 10
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"total_xp": xp_reward}}
        )
        
        await db.user_xp.insert_one({
            "user_id": current_user.id,
            "amount": xp_reward,
            "source": "create_album",
            "album_id": str(result.inserted_id),
            "created_at": datetime.utcnow()
        })
        
        return {
            "code": 200,
            "message": "Album created successfully",
            "album_id": str(result.inserted_id),
            "xp_reward": xp_reward
        }
        
    except Exception as e:
        print(f"Error creating album: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.put("/api/albums/{album_id}", summary="Update album")
async def update_album(
    album_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Update album details"""
    try:
        data = await request.json()
        name = data.get('name')
        description = data.get('description', '')
        
        if not name:
            return JSONResponse({"code": 400, "error": "Album name is required"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Update album
        result = await db.albums.update_one(
            {"_id": ObjectId(album_id), "user_id": current_user.id},
            {"$set": {
                "name": name,
                "description": description,
                "updated_at": datetime.utcnow()
            }}
        )
        
        if result.modified_count == 0:
            return JSONResponse({"code": 404, "error": "Album not found"}, status_code=404)
        
        return {"code": 200, "message": "Album updated successfully"}
        
    except Exception as e:
        print(f"Error updating album: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)


@app.put("/api/photos/{photo_id}/style")
async def update_photo_style(
    photo_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Update photo style"""
    try:
        data = await request.json()
        new_style = data.get('style')
        
        if not new_style:
            return JSONResponse({"code": 400, "error": "Style is required"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Find the photo
        photo = await db.photos.find_one({"_id": ObjectId(photo_id), "user_id": current_user.id})
        if not photo:
            return JSONResponse({"code": 404, "error": "Photo not found"}, status_code=404)
        
        # Update photo style
        await db.photos.update_one(
            {"_id": ObjectId(photo_id)},
            {"$set": {"style": new_style, "updated_at": datetime.utcnow()}}
        )
        
        # Award XP for updating style
        xp_reward = 2
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"total_xp": xp_reward}}
        )
        
        await db.user_xp.insert_one({
            "user_id": current_user.id,
            "amount": xp_reward,
            "source": "update_style",
            "photo_id": photo_id,
            "created_at": datetime.utcnow()
        })
        
        return {
            "code": 200,
            "message": "Style updated successfully",
            "xp_reward": xp_reward
        }
        
    except Exception as e:
        print(f"Error updating photo style: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.delete("/api/photos/{photo_id}", summary="Delete photo")
async def delete_photo(
    photo_id: str,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Delete a photo from album"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Find the photo
        photo = await db.photos.find_one({"_id": ObjectId(photo_id), "user_id": current_user.id})
        if not photo:
            return JSONResponse({"code": 404, "error": "Photo not found"}, status_code=404)
        
        # Get album info to update photo count later
        album_id = photo.get("album_id")
        
        # Delete the actual file from filesystem
        image_path = photo.get("image_path")
        if image_path:
            # Remove leading slash if present
            file_path = image_path.lstrip('/')
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✅ Deleted file: {file_path}")
            except Exception as e:
                print(f"⚠️ Error deleting file: {e}")
        
        # Delete photo from database
        await db.photos.delete_one({"_id": ObjectId(photo_id)})
        
        # Update album's photo count and updated_at
        if album_id:
            photo_count = await db.photos.count_documents({"album_id": album_id})
            await db.albums.update_one(
                {"_id": ObjectId(album_id)},
                {"$set": {"updated_at": datetime.utcnow(), "photo_count": photo_count}}
            )
        
        return {
            "code": 200,
            "message": "Photo deleted successfully"
        }
        
    except Exception as e:
        print(f"Error deleting photo: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)


@app.post("/api/photos/batch-delete", summary="Batch delete photos")
async def batch_delete_photos(
    request: Request,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Delete multiple photos"""
    try:
        data = await request.json()
        photo_ids = data.get('photo_ids', [])
        
        if not photo_ids:
            return JSONResponse({"code": 400, "error": "No photo IDs provided"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        deleted_count = 0
        album_ids = set()
        
        for photo_id in photo_ids:
            # Find the photo
            photo = await db.photos.find_one({"_id": ObjectId(photo_id), "user_id": current_user.id})
            if photo:
                album_ids.add(photo.get("album_id"))
                
                # Delete file
                image_path = photo.get("image_path")
                if image_path:
                    file_path = image_path.lstrip('/')
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"⚠️ Error deleting file: {e}")
                
                # Delete from database
                await db.photos.delete_one({"_id": ObjectId(photo_id)})
                deleted_count += 1
        
        # Update affected albums
        for album_id in album_ids:
            if album_id:
                photo_count = await db.photos.count_documents({"album_id": album_id})
                await db.albums.update_one(
                    {"_id": ObjectId(album_id)},
                    {"$set": {"updated_at": datetime.utcnow(), "photo_count": photo_count}}
                )
        
        return {
            "code": 200,
            "message": f"Successfully deleted {deleted_count} photos",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        print(f"Error batch deleting photos: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.delete("/api/albums/{album_id}", summary="Delete album")
async def delete_album(
    album_id: str,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Delete an album and all its photos"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Delete all photos in the album first
        await db.photos.delete_many({"album_id": album_id})
        
        # Delete the album
        result = await db.albums.delete_one({"_id": ObjectId(album_id), "user_id": current_user.id})
        
        if result.deleted_count == 0:
            return JSONResponse({"code": 404, "error": "Album not found"}, status_code=404)
        
        return {"code": 200, "message": "Album deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting album: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.post("/api/albums/upload", summary="Upload images to album")
async def upload_images_to_album(
    request: Request,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Upload multiple images to an album"""
    try:
        form = await request.form()
        album_id = form.get('album_id')
        files = form.getlist('images')
        style = form.get('style', 'original')  # get style from form, default to 'original' if not provided
        
        if not album_id:
            return JSONResponse({"code": 400, "error": "Album ID is required"}, status_code=400)
        
        if not files:
            return JSONResponse({"code": 400, "error": "No images uploaded"}, status_code=400)
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Check if album exists and belongs to user
        album = await db.albums.find_one({"_id": ObjectId(album_id), "user_id": current_user.id})
        if not album:
            return JSONResponse({"code": 404, "error": "Album not found"}, status_code=404)
        
        uploaded_photos = []
        total_xp = 0
        
        for file in files:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"album_photo_{current_user.id}_{timestamp}_{file_id}.jpg"
            filepath = f"uploads/album_photos/{filename}"
            
            # Ensure directory exists
            os.makedirs("uploads/album_photos", exist_ok=True)
            
            # Save file
            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            
            # Create photo record with style
            photo = {
                "album_id": album_id,
                "user_id": current_user.id,
                "filename": filename,
                "image_path": f"/uploads/album_photos/{filename}",
                "file_size": len(content),
                "style": style,  #add style field
                "uploaded_at": datetime.utcnow()
            }
            
            result = await db.photos.insert_one(photo)
            photo["_id"] = str(result.inserted_id)
            photo["id"] = photo["_id"]
            uploaded_photos.append(photo)
            total_xp += 2
        
        # Update album's updated_at
        await db.albums.update_one(
            {"_id": ObjectId(album_id)},
            {"$set": {"updated_at": datetime.utcnow()}}
        )

        if total_xp > 0:
            await db.users.update_one(
                {"_id": ObjectId(current_user.id)},
                {"$inc": {"total_xp": total_xp}}
            )
            
            await db.user_xp.insert_one({
                "user_id": current_user.id,
                "amount": total_xp,
                "source": "upload_photos",
                "album_id": album_id,
                "photo_count": len(uploaded_photos),
                "created_at": datetime.utcnow()
            })
        
        return {
            "code": 200,
            "message": f"Successfully uploaded {len(uploaded_photos)} images",
            "photos": uploaded_photos,
            "xp_reward": total_xp
        }
        
    except Exception as e:
        print(f"Error uploading images: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

# =============================
# User Stats API
# =============================
@app.get("/api/user/stats", summary="Get user statistics")
async def get_user_stats(current_user: UserInDB = Depends(get_current_user_strict)):
    """Get user statistics: transform count, favorite count, share count, total XP"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Get transform count from history
        transform_count = await db.history.count_documents({"user_id": current_user.id})
        
        # Get favorites count
        favorite_count = await db.favorites.count_documents({"user_id": current_user.id})
        
        # Get share count (published to gallery)
        share_count = await db.gallery.count_documents({"user_id": current_user.id})
        
        # Get total XP from users collection
        user = await db.users.find_one({"_id": ObjectId(current_user.id)})
        total_xp = user.get("total_xp", 0) if user else 0
        
        return {
            "code": 200,
            "transformCount": transform_count,
            "favoriteCount": favorite_count,
            "shareCount": share_count,
            "totalXP": total_xp
        }
        
    except Exception as e:
        print(f"Error loading stats: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

# =============================
# Gallery Like API
# =============================
@app.post("/api/gallery/{image_id}/like", summary="Like/unlike gallery image")
async def like_gallery_image(
    image_id: str,
    current_user: Optional[UserInDB] = Depends(get_current_user)
):
    """Like or unlike a gallery image"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        # Find the image
        image = await db.gallery.find_one({"_id": ObjectId(image_id)})
        if not image:
            return JSONResponse({"code": 404, "error": "Image not found"}, status_code=404)
        
        # If user is logged in, track unique likes
        if current_user:
            # Check if user already liked
            like = await db.gallery_likes.find_one({
                "image_id": image_id,
                "user_id": current_user.id
            })
            
            if like:
                # Unlike
                await db.gallery_likes.delete_one({"_id": like["_id"]})
                await db.gallery.update_one(
                    {"_id": ObjectId(image_id)},
                    {"$inc": {"likes": -1}}
                )
                return {"code": 200, "message": "Unliked", "liked": False}
            else:
                # Like
                await db.gallery_likes.insert_one({
                    "image_id": image_id,
                    "user_id": current_user.id,
                    "created_at": datetime.utcnow()
                })
                await db.gallery.update_one(
                    {"_id": ObjectId(image_id)},
                    {"$inc": {"likes": 1}}
                )
                
                # 给作品作者加 XP
                if str(image.get('user_id')) != current_user.id:
                    await db.users.update_one(
                        {"_id": ObjectId(image['user_id'])},
                        {"$inc": {"total_xp": 2}}
                    )
                    
                    await db.user_xp.insert_one({
                        "user_id": image['user_id'],
                        "amount": 2,
                        "source": "gallery_like_received",
                        "image_id": image_id,
                        "created_at": datetime.utcnow()
                    })
                
                return {"code": 200, "message": "Liked", "liked": True}
        else:
            # Anonymous like - just increment counter
            await db.gallery.update_one(
                {"_id": ObjectId(image_id)},
                {"$inc": {"likes": 1}}
            )
            return {"code": 200, "message": "Liked", "liked": True}
        
    except Exception as e:
        print(f"Error liking image: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

# =============================
# User Profile API
# =============================
@app.put("/api/user/profile", summary="Update user profile")
async def update_user_profile(
    request: Request,
    current_user: UserInDB = Depends(get_current_user_strict)
):
    """Update user profile information"""
    try:
        form = await request.form()
        username = form.get('username')
        bio = form.get('bio')
        avatar_file = form.get('avatar')
        
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        update_data = {}
        
        if username:
            update_data["username"] = username
        
        if bio:
            update_data["bio"] = bio
        
        # Handle avatar upload
        if avatar_file and avatar_file.filename:
            # Save avatar image
            file_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"avatar_{current_user.id}_{timestamp}_{file_id}.jpg"
            filepath = f"uploads/avatars/{filename}"
            
            os.makedirs("uploads/avatars", exist_ok=True)
            
            content = await avatar_file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            
            update_data["avatar_path"] = filepath
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            
            await db.users.update_one(
                {"_id": ObjectId(current_user.id)},
                {"$set": update_data}
            )

            xp_reward = 5
            await db.users.update_one(
                {"_id": ObjectId(current_user.id)},
                {"$inc": {"total_xp": xp_reward}}
            )
            
            await db.user_xp.insert_one({
                "user_id": current_user.id,
                "amount": xp_reward,
                "source": "update_profile",
                "created_at": datetime.utcnow()
            })
            
            return {
                "code": 200,
                "message": "Profile updated successfully",
                "username": username or current_user.username,
                "bio": bio or "",
                "xp_reward": xp_reward
            }
        
        return {
            "code": 200,
            "message": "Profile updated successfully",
            "username": username or current_user.username,
            "bio": bio or ""
        }
        
    except Exception as e:
        print(f"Error updating profile: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)

@app.get("/api/user/profile", summary="Get user profile")
async def get_user_profile(current_user: UserInDB = Depends(get_current_user_strict)):
    """Get full user profile including bio and avatar"""
    try:
        db = get_db()
        if db is None:
            return JSONResponse({"code": 500, "error": "Database not connected"}, status_code=500)
        
        user = await db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user:
            return JSONResponse({"code": 404, "error": "User not found"}, status_code=404)
        
        return {
            "code": 200,
            "username": user.get("username"),
            "email": user.get("email"),
            "bio": user.get("bio", ""),
            "avatar_path": user.get("avatar_path"),
            "created_at": user.get("created_at"),
            "total_xp": user.get("total_xp", 0)
        }
        
    except Exception as e:
        print(f"Error getting profile: {e}")
        return JSONResponse({"code": 500, "error": str(e)}, status_code=500)


@app.get("/api/crop-stats")
async def get_crop_stats(current_user: UserInDB = Depends(get_current_user_strict)):
    """Get cropping statistics (admin/debug only)"""
    return {
        "code": 200,
        "stats": cropper.get_stats()
    }

# =============================
# Image helpers
# =============================
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    print(f"Input tensor stats - min: {tensor.min():.3f}, max: {tensor.max():.3f}, mean: {tensor.mean():.3f}")
    return tensor

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor - preserves original dimensions"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # No forced resize to 256x256

def tensor_to_base64(tensor: torch.Tensor, model_type="unet"):
    """Convert model output tensor to base64 image string"""
    tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Cyberpunk model handling (tanh output [-1, 1])
    if model_type == "cyberpunk":
        # Convert from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        img = T.ToPILImage()(tensor.cpu())
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    # Ukiyo-e model handling (tanh output [-1, 1])
    if model_type == "ukiyoe":
        # Convert from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        img = T.ToPILImage()(tensor.cpu())
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    # Sketch model handling
    if model_type == "unet":
        tensor = (tensor + 1) / 2
        # Convert to grayscale for sketch effect
        gray = 0.299 * tensor[0:1, :, :] + 0.587 * tensor[1:2, :, :] + 0.114 * tensor[2:3, :, :]
        tensor = torch.cat([gray, gray, gray], dim=0)
    else:
        # Other models - normalize to [0, 1] if needed
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_min < 0 or tensor_max > 1:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    img = T.ToPILImage()(tensor.cpu())
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# =============================
# U-Net Model (Sketch)
# =============================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64, batchnorm=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)
        self.enc7 = self.conv_block(512, 512)
        self.enc8 = self.conv_block(512, 512, batchnorm=False)
        
        # Decoder
        self.dec8 = self.upconv_block(512, 512, dropout=True)
        self.dec7 = self.upconv_block(1024, 512, dropout=True)
        self.dec6 = self.upconv_block(1024, 512, dropout=True)
        self.dec5 = self.upconv_block(1024, 512)
        self.dec4 = self.upconv_block(1024, 256)
        self.dec3 = self.upconv_block(512, 128)
        self.dec2 = self.upconv_block(256, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def conv_block(self, in_channels, out_channels, batchnorm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def upconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d8 = self.dec8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.dec7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.dec6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        return d1

# =============================
# Style Transfer Network (for anime/ink)
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

class ResidualBlockV1(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlockV1, self).__init__()
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

class TransformerNetV1(torch.nn.Module):
    def __init__(self):
        super(TransformerNetV1, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlockV1(128)
        self.res2 = ResidualBlockV1(128)
        self.res3 = ResidualBlockV1(128)
        self.res4 = ResidualBlockV1(128)
        self.res5 = ResidualBlockV1(128)
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
# TransformerNet V2 (for cyberpunk style)
# =============================
class ResidualBlockV2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.block(x)

class TransformerNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 9, 1, 4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res_blocks = nn.Sequential(*[ResidualBlockV2(128) for _ in range(5)])
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.conv_out = nn.Conv2d(32, 3, 9, 1, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res_blocks(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.conv_out(y)
        return torch.tanh(y)

# =============================
# Load all models
# =============================
STYLE_NAMES = ["sketch", "anime", "ink", "hayao", "shinkai", "paprika", "cyberpunk", "ukiyoe"]
models: Dict[str, torch.nn.Module] = {}

print("Loading PyTorch style models...")
print("=" * 50)

# from model import Generator, UNet

def filter_state_dict(state_dict):
    """Remove running_mean and running_var keys from state_dict"""
    filtered = {}
    for key, value in state_dict.items():
        # Skip keys that end with running_mean or running_var
        if not (key.endswith('running_mean') or key.endswith('running_var')):
            filtered[key] = value
    return filtered

def load_unet_model(model_path):
    """Load your trained U-Net model"""
    model = UNet()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    return model

def load_animegan_model(model_path):
    """Load AnimeGANv2 model (Hayao, Shinkai, Paprika)"""
    model = Generator()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model

def load_transformernet_v1_model(model_path):
    """Load TransformerNet V1 model (anime, ink)"""
    model = TransformerNetV1()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Try to load with strict=True first
    try:
        model.load_state_dict(state_dict)
        print(f"   Loaded with strict matching")
    except RuntimeError as e:
        print(f"   ⚠️ Strict loading failed, trying with filtered dict...")
        filtered_dict = filter_state_dict(state_dict)
        model.load_state_dict(filtered_dict, strict=False)
        print(f"   Loaded with filtered state_dict")
    
    return model

def load_transformernet_v2_model(model_path):
    """Load TransformerNet V2 model (cyberpunk)"""
    model = TransformerNetV2()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Try to load with strict=True first
    try:
        model.load_state_dict(state_dict)
        print(f"   Loaded with strict matching")
    except RuntimeError as e:
        print(f"   ⚠️ Strict loading failed, trying with filtered dict...")
        filtered_dict = filter_state_dict(state_dict)
        model.load_state_dict(filtered_dict, strict=False)
        print(f"   Loaded with filtered state_dict")
    
    return model

def load_ukiyoe_model(model_path):
    """Load your trained Ukiyo-e model"""
    from model import UkiyoEGenerator
    model = UkiyoEGenerator()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'G_BA_state_dict' in checkpoint:
            state_dict = checkpoint['G_BA_state_dict']
            print(f"   Loaded G_BA (photo -> ukiyo-e) weights")
        elif 'G_AB_state_dict' in checkpoint:
            state_dict = checkpoint['G_AB_state_dict']
            print(f"   Loaded G_AB (ukiyoe -> photo) weights")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    return model

for name in STYLE_NAMES:
    model_path = f"models/{name}.pt"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}, using random weights")
        if name == "sketch":
            model = UNet()
            print(f"   Using untrained UNet for: {name}")
        elif name in ["hayao", "shinkai", "paprika"]:
            model = Generator()
            print(f"   Using untrained AnimeGAN for: {name}")
        elif name == "cyberpunk":
            model = TransformerNetV2()
            print(f"   Using untrained TransformerNetV2 for: {name}")
        elif name == "ukiyoe":                              
            model = UkiyoEGenerator()                        
            print(f"   Using untrained UkiyoEGenerator for: {name}")  
        else:  # anime, ink
            model = TransformerNetV1()
            print(f"   Using untrained TransformerNetV1 for: {name}")
    else:
        try:
            if name == "sketch":
                model = load_unet_model(model_path)
                print(f"✅ Loaded U-Net model: {name}")
                
            elif name in ["hayao", "shinkai", "paprika"]:
                model = load_animegan_model(model_path)
                print(f"✅ Loaded AnimeGAN model: {name}")
                
            elif name == "cyberpunk":
                model = load_transformernet_v2_model(model_path)
                print(f"✅ Loaded Cyberpunk model: {name}")
                
            elif name == "ukiyoe":                          
                model = load_ukiyoe_model(model_path)        
                print(f"✅ Loaded Ukiyo-e model: {name}")    
                
            else:  # anime, ink
                model = load_transformernet_v1_model(model_path)
                print(f"✅ Loaded TransformerNet model: {name}")
                    
        except Exception as e:
            print(f"❌ Failed to load model {name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to untrained model
            if name == "sketch":
                model = UNet()
            elif name in ["hayao", "shinkai", "paprika"]:
                model = Generator()
            elif name == "cyberpunk":
                model = TransformerNetV2()
            elif name == "ukiyoe":                          
                model = UkiyoEGenerator()                    
            else:
                model = TransformerNetV1()
            print(f"Using untrained fallback model for: {name}")

    model.eval()
    models[name] = model

print("=" * 50)
print(f"\n✅ All models ready! Loaded {len(models)} models:")
for name, model in models.items():
    print(f"   - {name}: {type(model).__name__}")