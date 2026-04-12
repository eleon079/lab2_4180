import io
import os
import sys
from pathlib import Path

from PIL import Image

# Add project root to Python path so "from app import app" works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Important: set this before importing app
os.environ["SKIP_MODEL_LOAD"] = "true"

from app import app


def make_test_image():
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200

    data = response.get_json()
    assert "message" in data
    assert "endpoints" in data


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "ok"
    assert data["task"] == "house-segmentation"
    assert "model_status" in data


def test_predict_missing_file():
    client = app.test_client()
    response = client.post("/predict", data={})
    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data


def test_predict_invalid_file():
    client = app.test_client()
    data = {
        "image": (io.BytesIO(b"not an image"), "bad.txt")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400

    json_data = response.get_json()
    assert "error" in json_data


def test_predict_valid_image():
    client = app.test_client()
    img_buf = make_test_image()

    data = {
        "image": (img_buf, "test.png")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    json_data = response.get_json()
    assert json_data["task"] == "house-segmentation"
    assert "mask_png_base64" in json_data
    assert "positive_pixels" in json_data
    assert "total_pixels" in json_data
    assert "house_pixel_ratio" in json_data