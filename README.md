# House Segmentation Service

This project trains and serves a U-Net segmentation model for aerial house/building segmentation using the INRIA Aerial Image Labeling dataset. The trained model is exposed through a Flask API and can be tested locally, containerized with Docker, and validated through CI/CD.

## Service overview

The API provides:

- `GET /` for a basic service message
- `GET /health` for configuration and model-load status
- `POST /predict` for segmentation inference on an uploaded image

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── artifacts/
│   ├── best_model.pth
│   ├── loss_curve.png
│   ├── metric_curves.png
│   ├── metrics.json
│   └── sample_predictions.png
├── data/
│   └── processed/
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
├── screen captures/
├── tests/
│   └── test_api.py
├── .dockerignore
├── .env.example
├── .gitignore
├── app.py
├── Dockerfile
├── prepare_dataset.py
├── README.md
├── requirements.txt
└── train_segmentation.py
```

## Installation

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are using a CUDA-enabled PyTorch setup and need a different wheel for your machine, install the matching PyTorch build recommended for your environment.

Example:
```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Configuration
This project uses environment variables loaded through .env with python-dotenv.

Key settings include:
- `DEVICE` for model training and API inference
- `SAM_DEVICE` for SAM-based dataset preparation
- `MODEL_PATH` for the trained checkpoint
- `DATA_ROOT` for processed dataset output
- `HF_DATASET_NAME` and `HF_DATASET_CONFIG` for dataset loading
- `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, and `LEARNING_RATE` for training
Use .env.example as a template for your local .env.

The code safely falls back to CPU if CUDA is requested but unavailable.

## Dataset preparation

This project uses:
- `blanchon/INRIA-Aerial-Image-Labeling`

The dataset preparation script:
1. loads INRIA aerial training images
2. locates the corresponding INRIA ground-truth masks
3. converts masks to binary building masks
4. derives connected-component bounding boxes from the masks
5. uses those bounding boxes as prompts to SAM through SamPredictor
6. merges the returned SAM masks into final segmentation masks
7. intersects the result with the original binary mask to reduce leakage
8. saves processed image/mask pairs into train/validation/test splits

Run:

```bash
python prepare_dataset.py
```

Output folders will be created under `data/processed/`:

```text
data/processed/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Model training

The training pipeline uses a U-Net with a `resnet34` encoder from `segmentation-models-pytorch`.

Training script features:
- binary segmentation training with `BCEWithLogitsLoss`
- validation during training
- checkpointing of the best model by validation Dice score
- final test-set evaluation with Dice and IoU
- saving plots and sample predictions
- CPU/GPU device fallback through environment-based configuration

Run:

```bash
python train_segmentation.py
```

Expected artifacts:

```text
artifacts/
├── best_model.pth
├── loss_curve.png
├── metric_curves.png
├── metrics.json
└── sample_predictions.png
```

## API usage

Start the service locally:

```bash
python app.py
```

### Endpoints

#### `GET /`
Returns a basic status message and endpoint summary.

#### `GET /health`
Returns service configuration and whether the model loaded successfully.

#### `POST /predict`
Accepts an image upload using multipart form data with a field named `image`.

Example using `curl`:

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@sample.png"
```

Example response fields:
- image width and height
- threshold used
- positive pixel count
- total pixel count
- predicted house-pixel ratio
- base64-encoded PNG segmentation mask

## Testing

Run the automated API tests with:

```bash
pytest -q
```

The tests cover:
- home endpoint
- health endpoint
- missing file handling
- invalid image handling
- valid prediction request handling

## Docker

Build the image:

```bash
docker build -t lab2-seg:latest .
```

Run the container:

```bash
docker run --rm -p 5000:5000 lab2-seg:latest
```

The Docker container starts the app with Waitress:

```bash
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

The container serves the Flask app with Waitress.

If the trained checkpoint is not available, you can use:

```bash
SKIP_MODEL_LOAD=true
```

for lightweight testing scenarios.

## CI/CD

The GitHub Actions workflow in `.github/workflows/ci-cd.yml` is configured to:
- install dependencies,
- run automated tests,
- build the Docker image,
- log in to Docker Hub on push events,
- push the Docker image.


## Notes

- The model is trained for binary house/building segmentation on aerial imagery.
- The best checkpoint is selected by highest validation Dice score.
- `DEVICE` controls training and API inference.
- `SAM_DEVICE` controls SAM execution during dataset preparation.
- Both training and inference support automatic fallback to CPU when CUDA is unavailable.
