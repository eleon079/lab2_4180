import argparse
import os
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from io import BytesIO


DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/processed"))
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "blanchon/INRIA-Aerial-Image-Labeling")
HF_DATASET_CONFIG = os.getenv("HF_DATASET_CONFIG", "default")
HF_MAX_SAMPLES = int(os.getenv("HF_MAX_SAMPLES", "180"))
SEED = int(os.getenv("DATASET_SEED", "42"))

# Week 7 / SAM settings
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")

sam_device_env = os.getenv("SAM_DEVICE", "").strip().lower()
if sam_device_env:
    if sam_device_env.startswith("cuda") and not torch.cuda.is_available():
        print(f"SAM_DEVICE={sam_device_env} requested but CUDA is unavailable. Falling back to cpu.")
        SAM_DEVICE = "cpu"
    else:
        SAM_DEVICE = sam_device_env
else:
    SAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_MASK_AREA = int(os.getenv("MIN_MASK_AREA", "50"))
SAM_MAX_IMAGE_SIDE = int(os.getenv("SAM_MAX_IMAGE_SIDE", "1024"))

# Optional cache paths
HF_HOME = os.getenv("HF_HOME", ".hf_cache")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", f"{HF_HOME}/hub")
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", f"{HF_HOME}/datasets")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

Path(HF_HOME).mkdir(parents=True, exist_ok=True)
Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
Path(HF_DATASETS_CACHE).mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def debug_dataset_sample(train_ds, num_samples=10):
    print("\nDEBUGGING DATASET SAMPLES")
    print("Dataset length:", len(train_ds))

    for i in range(min(num_samples, len(train_ds))):
        item = train_ds[i]
        print(f"\n--- Sample {i} ---")
        print("Keys:", list(item.keys()))

        for key, value in item.items():
            print(f"  {key}: type={type(value)}")

            if isinstance(value, Image.Image):
                print(
                    f"    PIL image mode={value.mode}, size={value.size}, "
                    f"filename={getattr(value, 'filename', None)}"
                )

            elif isinstance(value, dict):
                print(f"    dict keys={list(value.keys())}")
                if "path" in value:
                    print(f"    path={value['path']}")
                if "bytes" in value and value["bytes"] is not None:
                    print(f"    bytes_length={len(value['bytes'])}")

            elif hasattr(value, "shape"):
                print(f"    shape={value.shape}")


def save_debug_raw_samples(train_ds, out_dir="debug_raw", num_samples=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(num_samples, len(train_ds))):
        item = train_ds[i]

        if "image" in item:
            img = item["image"]
            if isinstance(img, Image.Image):
                img.convert("RGB").save(out_dir / f"sample_{i:03d}_image.png")

        if "label" in item:
            try:
                mask = ensure_binary_mask(item["label"])
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                    out_dir / f"sample_{i:03d}_label.png"
                )
            except Exception as exc:
                print(f"Could not save label for sample {i}: {exc}")



def looks_like_binary_mask(image: Image.Image, unique_limit=8):
    arr = np.array(image.convert("L"))
    unique_vals = np.unique(arr)
    return len(unique_vals) <= unique_limit


def find_train_image_root():
    candidates = []

    for root in [Path(HF_DATASETS_CACHE), Path(HF_HOME), Path(".")]:
        if root.exists():
            candidates.extend(root.rglob("data/train/images"))
            candidates.extend(root.rglob("train/images"))
            candidates.extend(root.rglob("data/train/img"))
            candidates.extend(root.rglob("train/img"))

    candidates = [p for p in candidates if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(
            "Could not locate the INRIA training image folder (train/images or train/img)."
        )

    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return candidates[0]




def output_paths(output_dir, split_name, idx):
    img_dir = output_dir / split_name / "images"
    mask_dir = output_dir / split_name / "masks"
    out_name = f"{split_name}_{idx:05d}.png"
    return img_dir / out_name, mask_dir / out_name


def pair_already_exists(output_dir, split_name, idx):
    image_path, mask_path = output_paths(output_dir, split_name, idx)
    if not (image_path.is_file() and mask_path.is_file()):
        return False

    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(mask_path) as m:
            m.verify()
        return True
    except Exception:
        return False


def ensure_binary_mask(mask_like):
    if isinstance(mask_like, Image.Image):
        mask_img = mask_like.convert("L")

    elif isinstance(mask_like, (str, Path)):
        mask_img = Image.open(mask_like).convert("L")

    elif isinstance(mask_like, dict):
        if mask_like.get("bytes") is not None:
            mask_img = Image.open(BytesIO(mask_like["bytes"])).convert("L")
        elif mask_like.get("path") is not None:
            mask_img = Image.open(mask_like["path"]).convert("L")
        else:
            raise ValueError(f"Unsupported mask dict payload: keys={list(mask_like.keys())}")

    else:
        arr = np.array(mask_like)

        if arr.ndim == 0:
            raise ValueError(
                f"Mask is not image-like. Got scalar/object value of type {type(mask_like)}: {mask_like!r}"
            )

        mask_img = Image.fromarray(arr).convert("L")

    mask_array = np.array(mask_img)
    return (mask_array > 0).astype(np.uint8)


def get_label_mask(item):
    if "label" in item:
        return item["label"]

    raise KeyError(
        f"Sample does not contain a 'label' field. Available keys: {list(item.keys())}"
    )


def find_train_gt_root():
    candidates = []

    for root in [Path(HF_DATASETS_CACHE), Path(HF_HOME), Path(".")]:
        if root.exists():
            candidates.extend(root.rglob("data/train/gt"))
            candidates.extend(root.rglob("train/gt"))

    candidates = [p for p in candidates if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(
            "Could not locate the INRIA training ground-truth folder (train/gt) "
            "inside the Hugging Face cache or project directory."
        )

    # Prefer the shortest path / earliest hit
    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return candidates[0]


def get_image_filename(example_image):
    if isinstance(example_image, Image.Image):
        filename = getattr(example_image, "filename", None)
        if filename:
            return Path(filename).name

    if isinstance(example_image, dict):
        if example_image.get("path"):
            return Path(example_image["path"]).name

    raise ValueError(
        "Could not determine the original image filename from the dataset example."
    )


def get_example_image_and_mask(example, image_root, gt_root):
    if "image" not in example:
        raise KeyError("Example is missing 'image'.")

    example_image = example["image"]
    image_name = get_image_filename(example_image)

    image_path = image_root / image_name
    gt_path = gt_root / image_name

    if not image_path.exists():
        raise FileNotFoundError(
            f"Training image not found for '{image_name}'. Expected at: {image_path}"
        )

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found for '{image_name}'. Expected at: {gt_path}"
        )

    image = Image.open(image_path).convert("RGB")
    return image, gt_path


def resize_image_and_mask(image, binary_mask, max_side):
    """
    Downscale large INRIA images before running SAM, then keep track of scale so
    the final mask can be resized back to original resolution.
    """
    width, height = image.size
    longest_side = max(width, height)

    if max_side <= 0 or longest_side <= max_side:
        return image, binary_mask, 1.0

    scale = max_side / float(longest_side)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    image_small = image.resize((new_w, new_h), Image.BILINEAR)
    mask_small = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L").resize(
        (new_w, new_h), Image.NEAREST
    )
    mask_small = (np.array(mask_small) > 0).astype(np.uint8)
    return image_small, mask_small, scale


def make_bbox_mask(labelled_bbox, image_shape):
    """
    Week 7-style helper:
    Convert one labelled bounding box into a binary mask.
    bbox format: [x_min, y_min, width, height]
    """
    height, width = image_shape
    x_min, y_min, box_w, box_h = labelled_bbox
    x_min, y_min, box_w, box_h = int(x_min), int(y_min), int(box_w), int(box_h)

    mask = np.zeros((height, width), dtype=np.uint8)

    x_max = min(x_min + box_w, width)
    y_max = min(y_min + box_h, height)

    if x_min < x_max and y_min < y_max:
        mask[y_min:y_max, x_min:x_max] = 1

    return mask


def bbox_union_mask(bboxes, image_shape):
    final_mask = np.zeros(image_shape, dtype=np.uint8)
    for bbox in bboxes:
        final_mask = np.logical_or(final_mask, make_bbox_mask(bbox, image_shape)).astype(np.uint8)
    return final_mask


def iou_score(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def extract_component_bboxes(binary_mask, min_area):
    """
    Since INRIA provides pixel masks rather than labelled boxes, derive boxes
    from connected components, then feed those boxes into the Week 7 SAM flow.
    """
    binary_mask = binary_mask.astype(np.uint8)
    height, width = binary_mask.shape
    visited = np.zeros((height, width), dtype=bool)
    bboxes = []

    foreground = np.argwhere(binary_mask > 0)
    for start_y, start_x in foreground:
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        coords = []

        while stack:
            y, x = stack.pop()
            coords.append((y, x))

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width:
                    if not visited[ny, nx] and binary_mask[ny, nx] > 0:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        if len(coords) < min_area:
            continue

        ys = [p[0] for p in coords]
        xs = [p[1] for p in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bboxes.append([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1])

    return bboxes


def build_sam_predictor():
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "The 'segment_anything' package is required for dataset preparation.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "Then re-run prepare_dataset.py."
        ) from exc

    if not Path(SAM_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at '{SAM_CHECKPOINT}'. "
            "Download it first and update SAM_CHECKPOINT in .env if needed."
        )

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=SAM_DEVICE)
    predictor = SamPredictor(sam)
    return predictor



def generate_week7_box_prompt_mask(image, derived_bboxes, predictor):
    """
    Adapt Week 7-style SAM box prompting to INRIA:
    - convert the provided label mask to a binary mask
    - derive connected-component bounding boxes from the binary mask
    - optionally downscale the image and mask for faster SAM inference
    - prompt SAM with each derived bounding box using SamPredictor
    - union the returned masks into one final binary mask
    - resize back to original size if needed
    - intersect with the original binary mask to limit mask leakage
    """
    image_np = np.array(image.convert("RGB"))
    predictor.set_image(image_np)

    final_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for bbox in derived_bboxes:
        x_min, y_min, box_w, box_h = bbox
        x_max = x_min + box_w
        y_max = y_min + box_h

        input_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            continue

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)

        final_mask = np.logical_or(final_mask, best_mask).astype(np.uint8)

    if final_mask.sum() == 0:
        final_mask = bbox_union_mask(derived_bboxes, (image.height, image.width))

    return final_mask



def generate_inria_week7_sam_mask(image, label_mask, predictor):
    """
    Adapt Week 7 SAM logic to INRIA:
    - derive component boxes from the provided pixel labels
    - optionally downscale for SAM speed
    - run SAM and keep masks that overlap enough with derived label boxes
    - resize back to original size
    - intersect with original binary mask to avoid leaking too far outside labels
    """
    binary_mask = ensure_binary_mask(label_mask)
    original_h, original_w = binary_mask.shape

    image_small, mask_small, scale = resize_image_and_mask(image, binary_mask, SAM_MAX_IMAGE_SIDE)
    derived_bboxes = extract_component_bboxes(mask_small, min_area=MIN_MASK_AREA)

    if not derived_bboxes:
        final_small = mask_small
    else:
        final_small = generate_week7_box_prompt_mask(image_small, derived_bboxes, predictor)
        final_small = np.logical_and(final_small, mask_small).astype(np.uint8)

    if scale != 1.0:
        restored = Image.fromarray((final_small * 255).astype(np.uint8), mode="L").resize(
            (original_w, original_h), Image.NEAREST
        )
        final_mask = (np.array(restored) > 0).astype(np.uint8)
    else:
        final_mask = final_small.astype(np.uint8)

    final_mask = np.logical_and(final_mask, binary_mask).astype(np.uint8)
    return (final_mask * 255).astype(np.uint8), derived_bboxes


def collect_labelled_examples(train_ds, image_root, gt_root):
    labelled_examples = []
    skipped = 0

    for idx, item in enumerate(train_ds):
        try:
            image, mask_path = get_example_image_and_mask(item, image_root, gt_root)
            _ = ensure_binary_mask(mask_path)  # validate the mask file
            labelled_examples.append({"image": image, "label": mask_path})

            print(
                f"DEBUG collected sample {idx}: "
                f"image_mode={image.mode}, image_size={image.size}, "
                f"image_filename={getattr(image, 'filename', None)}, "
                f"mask_path={mask_path}"
            )
            

        except Exception as exc:
            skipped += 1
            print(f"  skipping train example {idx}: {exc}")

        if (idx + 1) % 25 == 0:
            print(
                f"  scanned {idx + 1} training examples "
                f"({len(labelled_examples)} usable, {skipped} skipped)"
            )

    if not labelled_examples:
        raise RuntimeError(
            "No usable labelled training examples were found. "
            "Check image filenames and train/gt mask paths."
        )

    print(f"Collected {len(labelled_examples)} usable labelled examples; skipped {skipped}.")
    return labelled_examples



def save_pair(image, mask_array, output_dir, split_name, idx):
    image_path, mask_path = output_paths(output_dir, split_name, idx)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_image = image.convert("RGB")

    if looks_like_binary_mask(rgb_image):
        print(
            f"WARNING: image for {split_name}_{idx:05d} looks mask-like. "
            f"filename={getattr(image, 'filename', None)}"
        )

    rgb_image.save(image_path)
    Image.fromarray(mask_array, mode="L").save(mask_path)


def collect_image_mask_pairs(image_root, gt_root, max_samples=0):
    image_paths = sorted(
        [p for p in image_root.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}]
    )
    pairs = []

    for image_path in image_paths:
        gt_path = gt_root / image_path.name
        if not gt_path.exists():
            print(f"Skipping {image_path.name}: missing gt file")
            continue

        pairs.append({
            "image_path": image_path,
            "mask_path": gt_path,
        })

    if max_samples > 0:
        pairs = pairs[: min(len(pairs), max_samples)]

    if not pairs:
        raise RuntimeError("No valid image/mask pairs found.")

    print(f"Collected {len(pairs)} valid RGB image / GT mask pairs.")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--max-samples", type=int, default=HF_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing processed image/mask pairs."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Ensuring dataset is available: {HF_DATASET_NAME} ({HF_DATASET_CONFIG})")
    _ = load_dataset(HF_DATASET_NAME, name=HF_DATASET_CONFIG, split="train")

    gt_root = find_train_gt_root()
    print(f"Using INRIA training GT masks from: {gt_root}")

    image_root = find_train_image_root()
    print(f"Using INRIA training RGB images from: {image_root}")

    print("Building SAM predictor...")
    predictor = build_sam_predictor()

    print("Collecting labelled training examples from INRIA...")
    examples = collect_image_mask_pairs(image_root, gt_root, max_samples=args.max_samples)
    random.shuffle(examples)

    if args.max_samples > 0:
        examples = examples[: min(len(examples), args.max_samples)]

    n = len(examples)
    if n < 3:
        raise RuntimeError("Not enough labelled samples to create train/val/test splits.")

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    split_examples = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    print(
        f"Final sizes -> train: {len(split_examples['train'])}, "
        f"val: {len(split_examples['val'])}, "
        f"test: {len(split_examples['test'])}"
    )

    total_bboxes = 0
    for split_name, split_data in split_examples.items():
        print(f"Saving {split_name} images and Week 7 SAM-processed masks...")
        skipped_existing = 0
        processed_now = 0

        for idx, item in enumerate(split_data):
            if not args.force and pair_already_exists(output_dir, split_name, idx):
                skipped_existing += 1
                if (idx + 1) % 10 == 0:
                    print(
                        f"  reached {idx + 1} {split_name} samples "
                        f"({processed_now} processed now, {skipped_existing} skipped existing)"
                    )
                continue

            image = Image.open(item["image_path"]).convert("RGB")
            label_mask = item["mask_path"]

            mask_array, derived_bboxes = generate_inria_week7_sam_mask(
                image=image,
                label_mask=label_mask,
                predictor=predictor,
            )
            total_bboxes += len(derived_bboxes)
            save_pair(image, mask_array, output_dir, split_name, idx)
            processed_now += 1

            if (idx + 1) % 10 == 0:
                print(
                    f"  reached {idx + 1} {split_name} samples "
                    f"({processed_now} processed now, {skipped_existing} skipped existing)"
                )

        print(
            f"Finished {split_name}: {processed_now} processed now, "
            f"{skipped_existing} skipped because outputs already existed."
        )

    print("INRIA aerial segmentation dataset prepared successfully.")
    print(
        {
            "dataset": HF_DATASET_NAME,
            "config": HF_DATASET_CONFIG,
            "source_split_used": "train",
            "sam_model_type": SAM_MODEL_TYPE,
            "sam_device": SAM_DEVICE,
            "sam_max_image_side": SAM_MAX_IMAGE_SIDE,
            "max_samples": args.max_samples,
            "train": len(split_examples["train"]),
            "val": len(split_examples["val"]),
            "test": len(split_examples["test"]),
            "week7_style": {
                "min_mask_area": MIN_MASK_AREA,
                "total_component_bboxes": total_bboxes,
            },
            "output_dir": str(output_dir),
            "cache_dirs": {
                "HF_HOME": HF_HOME,
                "HF_HUB_CACHE": HF_HUB_CACHE,
                "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
                "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ".cache"),
            },
        }
    )


if __name__ == "__main__":
    main()
