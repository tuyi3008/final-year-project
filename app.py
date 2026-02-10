import io
import base64
import os
from typing import Dict

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torchvision.transforms as T


# =============================
# FastAPI init
# =============================
app = FastAPI(title="Multi‑Style Transfer API (PyTorch)")

# Serve frontend
app.mount("/static", StaticFiles(directory="public"), name="static")


@app.get("/")
async def root():
    return FileResponse("public/index.html")


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
# (CPU‑friendly placeholder CNNs)
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


# =============================
# Stylize endpoint
# =============================
@app.post("/stylize/")
async def stylize(
    content: UploadFile = File(...),
    style: str = Form(...),
):
    try:
        if style not in models:
            return JSONResponse({"error": "Invalid style"}, status_code=400)

        content_tensor = load_image(content)

        with torch.no_grad():
            output = models[style](content_tensor)

        b64 = tensor_to_base64(output)

        return {"image_base64": b64}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
