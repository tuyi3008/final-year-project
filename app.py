import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Initialize FastAPI
app = FastAPI(title="Style Transfer API")

# Load TensorFlow Hub model (arbitrary style transfer)
MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
print("Loading TensorFlow Hub model...")
hub_model = hub.load(MODEL_URL)
print("Model loaded successfully!")

# Helper function: Load image as tensor
def load_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    img = img.resize((256, 256))  # Resize to speed up processing
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # [1, H, W, 3]
    return img

# Helper function: Convert tensor to Base64
def tensor_to_base64(tensor):
    img = tensor[0]  # Remove batch dimension
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# POST endpoint: Upload content and style images
@app.post("/stylize/")
async def stylize(content: UploadFile = File(...), style: UploadFile = File(...)):
    try:
        content_image = load_image(content)
        style_image = load_image(style)

        # Run style transfer
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

        # Convert to Base64 and return
        b64_image = tensor_to_base64(stylized_image)
        return JSONResponse(content={"image_base64": b64_image})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
