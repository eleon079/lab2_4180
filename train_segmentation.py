import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from dotenv import load_dotenv
load_dotenv()


from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ----------------
# Configuration
# ----------------

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/processed"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))


# Select CPU or GPU from environment settings, with safe fallback
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

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------
# Dataset class
# ----------------
class HouseSegmentationDataset(Dataset):
    """
    Simple dataset wrapper for the processed split folders.

    Expected structure:
      split/
        images/
        masks/

    Each image should have a matching mask with the same filename stem.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.image_paths = sorted([
            p for p in self.image_dir.glob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        self.image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask

# ----------------
# Metrics
# ----------------

def dice_score(preds, targets, eps=1e-7):
    # Dice coefficient for binary segmentation.
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)


def iou_score(preds, targets, eps=1e-7):
    # Intersection over Union (IoU) for binary segmentation.
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


def evaluate(model, loader, criterion):
    # Evaluate the model on one dataloader. Returns average: loss, Dice UoU
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss += loss.item()
            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()
            total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "dice": total_dice / max(total_batches, 1),
        "iou": total_iou / max(total_batches, 1),
    }


# ----------------
# Plot saving
# ----------------

def save_training_curves(history):
    # Save the loss curve and validation metric curve to disk.
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_dice"], label="val_dice")
    plt.plot(epochs, history["val_iou"], label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Dice and IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "metric_curves.png")
    plt.close()


def save_sample_predictions(model, loader, num_samples=3):
    # Save small grid of input image, ground truth mask and predicted mask (used for report)
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    count = min(num_samples, images.size(0))
    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(count):
        image_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
        mask_np = masks[i, 0].detach().cpu().numpy()
        pred_np = preds[i, 0].detach().cpu().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_np, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "sample_predictions.png")
    plt.close()

# -------------------------------
# Main (training entry point)
# -------------------------------

def main():
    # Load processed dataset splits
    train_ds = HouseSegmentationDataset(DATA_ROOT / "train")
    val_ds = HouseSegmentationDataset(DATA_ROOT / "val")
    test_ds = HouseSegmentationDataset(DATA_ROOT / "test")

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            "Dataset splits are empty. Run prepare_dataset.py first and verify image/mask files exist."
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # U-Net with a pretrained ResNet34 encoder
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = -math.inf

    # Store curves over time
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        train_loss = epoch_loss / max(batches, 1)


        # Evaluate on validation set after each epoch
        val_metrics = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        # Keep the best checkpoint based on validation Dice
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_name": ENCODER_NAME,
                    "image_size": IMAGE_SIZE,
                    "best_val_dice": best_val_dice,
                },
                ARTIFACTS_DIR / "best_model.pth",
            )

    # Save training curves after training ends
    save_training_curves(history)

    # Reload the best checkpoint before evaluating on the test set
    checkpoint = torch.load(ARTIFACTS_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test-set evaluation
    test_metrics = evaluate(model, test_loader, criterion)

    # Save a small prediction grid for the report
    save_sample_predictions(model, test_loader)

    # Write metrics/config to JSON for easy reporting
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_metrics["loss"],
                "test_dice": test_metrics["dice"],
                "test_iou": test_metrics["iou"],
                "best_val_dice": best_val_dice,
                "encoder_name": ENCODER_NAME,
                "image_size": IMAGE_SIZE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
            f,
            indent=2,
        )

    print("Training complete.")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()