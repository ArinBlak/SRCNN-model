from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="SRCNN Image Enhancement API")

# -----------------------------
# Load SRCNN model once at startup
# -----------------------------
model = tf.keras.models.load_model("srcnn_model_backup.h5")


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image_bytes):
    # Load image from bytes and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Convert to numpy and normalize
    img_np = np.array(img).astype(np.float32) / 255.0

    # Add batch & channel dimensions -> (1, H, W, 1)
    img_np = np.expand_dims(img_np, axis=(0, -1))
    return img_np


# -----------------------------
# Image postprocessing
# -----------------------------
def postprocess_image(output):
    # Remove batch & channel dimensions
    output = np.squeeze(output)

    # Clip and rescale
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)

    return Image.fromarray(output, mode="L")


# -----------------------------
# API endpoint
# -----------------------------
@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Preprocess
    input_image = preprocess_image(image_bytes)

    # Inference
    output = model.predict(input_image)

    # Postprocess
    output_image = postprocess_image(output)

    # Convert PIL image to byte stream
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
