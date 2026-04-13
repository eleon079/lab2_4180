import base64
import io
import os

import numpy as np
import torch
import segmentation_models_pytorch as smp
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from PIL import Image

load_dotenv()

app = Flask(__name__)

# Configuration
PORT = int(os.getenv("PORT", "5000"))
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/best_model.pth")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# Decide whether to use CPU or GPU
device_env = os.getenv("DEVICE", "").strip().lower()
if device_env:
    if device_env.startswith("cuda") and not torch.cuda.is_available():
        print(f"DEVICE={device_env} requested but CUDA is unavailable. Falling back to cpu.")
        DEVICE = "cpu"
    else:
        DEVICE = device_env
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_NAME = os.getenv("ENCODER_NAME", "resnet34")

_model = None

# Model helpers
def build_model():
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    return model


def load_model():
    global _model
    if _model is not None:
        return _model

    if os.getenv("SKIP_MODEL_LOAD", "false").lower() == "true":
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                batch, _, h, w = x.shape
                return torch.zeros((batch, 1, h, w), dtype=torch.float32, device=x.device)

        _model = DummyModel().to(DEVICE).eval()
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Train the model first or update MODEL_PATH in .env."
        )

    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    _model = model
    return _model

# Image preprocessing / postprocessing
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    original_w, original_h = image.size
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor, (original_w, original_h)


def postprocess_mask(mask_probs: np.ndarray, original_size):
    mask = (mask_probs >= THRESHOLD).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return mask_img


def image_to_base64_png(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Routes
@app.get("/")
def home():
    return jsonify(
        message="House segmentation service is running.",
        endpoints={
            "health": "GET /health",
            "predict": "POST /predict with multipart/form-data field named 'image'"
        }
    )


@app.get("/health")
def health():
    model_status = "loaded"
    try:
        load_model()
    except Exception as exc:
        model_status = f"error: {str(exc)}"

    return jsonify(
        status="ok",
        task="house-segmentation",
        model_path=MODEL_PATH,
        encoder_name=ENCODER_NAME,
        image_size=IMAGE_SIZE,
        threshold=THRESHOLD,
        device=DEVICE,
        model_status=model_status,
    )


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(
            error="Invalid request. Send multipart/form-data with a file field named 'image'."
        ), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(error="No file selected."), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify(error="Uploaded file is not a valid image."), 400

    model = load_model()
    tensor, original_size = preprocess_image(image)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    mask_img = postprocess_mask(probs, original_size)
    mask_arr = np.array(mask_img)
    positive_pixels = int((mask_arr > 0).sum())
    total_pixels = int(mask_arr.size)
    house_pixel_ratio = float(positive_pixels / max(total_pixels, 1))

    return jsonify(
        task="house-segmentation",
        width=original_size[0],
        height=original_size[1],
        threshold=THRESHOLD,
        positive_pixels=positive_pixels,
        total_pixels=total_pixels,
        house_pixel_ratio=house_pixel_ratio,
        mask_png_base64=image_to_base64_png(mask_img),
    )


if __name__ == "__main__":
    # Local development server.
    # In Docker/production, Waitress is typically used instead.
    app.run(host="0.0.0.0", port=PORT, debug=True)