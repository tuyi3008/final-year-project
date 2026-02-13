import io
import base64
import os
from typing import Dict

import torch
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torchvision.transforms as T
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import authentication functions and models
from auth import (
    register_user,
    login_for_access_token,
    get_current_user,
    UserCreate,
    UserInDB,
    OAuth2PasswordRequestForm
)

# =============================
# FastAPI init
# =============================
app = FastAPI(title="Multi-Style Transfer API (PyTorch)")

# Serve frontend
app.mount("/static", StaticFiles(directory="public"), name="static")

# =============================
# Authentication routes (public)
# =============================
@app.post("/register", summary="User registration")
async def register(user: UserCreate):
    return register_user(user)

@app.post("/login", summary="User login to get token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return login_for_access_token(form_data)

@app.get("/profile", summary="Get current user info (login required)")
async def get_profile(current_user: UserInDB = Depends(get_current_user)):
    return {
        "code": 200,
        "username": current_user.username,
        "email": current_user.email,
        "msg": "Logged in"
    }

# =============================
# Public interfaces (no login required)
# =============================
@app.get("/")
async def root():
    return FileResponse("public/index.html")

@app.post("/stylize/")
async def stylize(
    content: UploadFile = File(...),
    style: str = Form(...)
    # Note: No Depends(get_current_user), so login is NOT required
):
    try:
        if style not in models:
            return JSONResponse({"error": "Invalid style"}, status_code=400)

        content_tensor = load_image(content)

        with torch.no_grad():
            output = models[style](content_tensor)

        b64 = tensor_to_base64(output)

        # Return without user information
        return {"image_base64": b64}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================
# Image helpers
# =============================
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

def load_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
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