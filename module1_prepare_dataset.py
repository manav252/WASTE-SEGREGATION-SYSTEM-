"""
================================================================================
MODULE 1 — DATASET PREPARATION
================================================================================
Purpose : Extracts the labeled dataset from the zip file, reads YOLO bounding
          box annotations, crops each labeled object from every image, and
          splits the result into train / val / test folders.

Why crop : The YOLO labels tell us exactly where each sanitary pad is inside
           each image. By cropping those regions, we train the CNN on tight
           crops of the actual pad — not the background noise around it.
           This makes classification far more accurate.

Output structure after running:
    ahp_dataset/
        train/
            pad/          ← cropped sanitary pad images (positive class)
            no_pad/       ← background crops (negative class, auto-generated)
        val/
            pad/
            no_pad/
        test/
            pad/
            no_pad/

Run:
    python module1_prepare_dataset.py
================================================================================
"""

import os
import sys
import zipfile
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── SETTINGS ──────────────────────────────────────────────────────────────────
ZIP_PATH    = "project-1-at-2026-02-17-21-44-c758d2de.zip"   # your dataset zip
EXTRACT_DIR = "ahp_raw"          # where zip is extracted (temporary)
DATASET_DIR = "ahp_dataset"      # final organised dataset folder
TRAIN_RATIO = 0.70               # 70% train
VAL_RATIO   = 0.15               # 15% val  → remaining 15% = test
CROP_PAD    = 0.10               # add 10% padding around each YOLO bounding box
RANDOM_SEED = 42
IMG_SIZE    = 224                # resize all crops to 224 x 224
# ──────────────────────────────────────────────────────────────────────────────


def extract_zip():
    """Extract the dataset zip into EXTRACT_DIR."""
    raw = Path(EXTRACT_DIR)

    if (raw / "images").exists():
        print(f"  Already extracted at '{EXTRACT_DIR}' — skipping.")
        return

    if not Path(ZIP_PATH).exists():
        print(f"\n  ERROR: Zip file not found: '{ZIP_PATH}'")
        print(f"  Make sure it is in the same folder as this script.\n")
        sys.exit(1)

    print(f"  Extracting '{ZIP_PATH}' ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(str(raw))
    print(f"  Done. Files extracted to '{EXTRACT_DIR}/'")


def read_yolo_labels(label_path):
    """
    Read a YOLO-format label file.

    Each line: class_id  cx  cy  w  h   (all normalised 0-1)
    Returns a list of dicts with those fields.
    """
    boxes = []
    if not Path(label_path).exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            boxes.append({
                "class_id": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w":  float(parts[3]),
                "h":  float(parts[4]),
            })
    return boxes


def yolo_to_pixels(box, img_w, img_h):
    """
    Convert a YOLO box (normalised cx, cy, w, h) into pixel coordinates.
    Returns (x1, y1, x2, y2) — top-left and bottom-right corners.
    """
    cx = box["cx"] * img_w
    cy = box["cy"] * img_h
    bw = box["w"]  * img_w
    bh = box["h"]  * img_h

    # Add padding around the box
    pad_x = bw * CROP_PAD
    pad_y = bh * CROP_PAD

    x1 = int(max(0,        cx - bw / 2 - pad_x))
    y1 = int(max(0,        cy - bh / 2 - pad_y))
    x2 = int(min(img_w,    cx + bw / 2 + pad_x))
    y2 = int(min(img_h,    cy + bh / 2 + pad_y))
    return x1, y1, x2, y2


def generate_negative_crop(img, existing_boxes_px):
    """
    Generate one background (no-pad) crop from a region that does NOT
    overlap with any labelled sanitary pad box.
    Returns a cropped image or None if no valid region found.
    """
    h, w = img.shape[:2]
    min_size = min(w, h) // 4   # minimum negative crop size

    for _ in range(30):          # try up to 30 random positions
        cw = random.randint(min_size, w // 2)
        ch = random.randint(min_size, h // 2)
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        x2, y2 = x1 + cw, y1 + ch

        # Check that this region does not overlap any pad box
        overlap = False
        for (bx1, by1, bx2, by2) in existing_boxes_px:
            if x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1:
                overlap = True
                break

        if not overlap:
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return None


def build_crops():
    """
    For every image in the dataset:
      1. Read all YOLO bounding boxes (sanitary pad positions).
      2. Crop each box → positive sample (label = pad).
      3. Crop a random background region → negative sample (label = no_pad).
    Returns two lists: pad_crops, nopad_crops  — each item is a numpy array.
    """
    img_dir = Path(EXTRACT_DIR) / "images"
    lbl_dir = Path(EXTRACT_DIR) / "labels"

    all_images = sorted(list(img_dir.glob("*.jpg")) +
                        list(img_dir.glob("*.jpeg")) +
                        list(img_dir.glob("*.png")))

    if not all_images:
        print(f"\n  ERROR: No images found in '{img_dir}'")
        sys.exit(1)

    pad_crops   = []   # positive class
    nopad_crops = []   # negative class

    print(f"\n  Processing {len(all_images)} images ...")

    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Find matching label file
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes    = read_yolo_labels(str(lbl_path))

        boxes_px = []   # pixel coords for overlap check

        for box in boxes:
            x1, y1, x2, y2 = yolo_to_pixels(box, w, h)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            pad_crops.append(crop_resized)
            boxes_px.append((x1, y1, x2, y2))

        # Generate one negative crop per image (keeps dataset balanced)
        neg = generate_negative_crop(img, boxes_px)
        if neg is not None:
            nopad_crops.append(neg)

    print(f"  Positive (pad) crops    : {len(pad_crops)}")
    print(f"  Negative (no-pad) crops : {len(nopad_crops)}")
    return pad_crops, nopad_crops


def save_split(crops, class_name, split_counts):
    """Save a list of crops into the correct train/val/test subfolders."""
    random.shuffle(crops)
    n   = len(crops)
    ntr = split_counts["train"]
    nva = split_counts["val"]

    splits = {
        "train": crops[:ntr],
        "val"  : crops[ntr:ntr + nva],
        "test" : crops[ntr + nva:],
    }

    for split_name, items in splits.items():
        out_dir = Path(DATASET_DIR) / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, crop in enumerate(items):
            fname = out_dir / f"{class_name}_{split_name}_{idx:04d}.jpg"
            cv2.imwrite(str(fname), crop)

    return {k: len(v) for k, v in splits.items()}


def prepare_dataset():
    """Main function — runs the full dataset preparation pipeline."""
    print("=" * 60)
    print("  MODULE 1 — DATASET PREPARATION")
    print("=" * 60)

    # Clean old dataset if it exists
    if Path(DATASET_DIR).exists():
        print(f"\n  Removing old dataset at '{DATASET_DIR}' ...")
        shutil.rmtree(DATASET_DIR)

    # Step 1 — Extract zip
    print("\n  [1/4] Extracting zip ...")
    extract_zip()

    # Step 2 — Build crops from YOLO annotations
    print("\n  [2/4] Cropping annotated regions ...")
    pad_crops, nopad_crops = build_crops()

    # Step 3 — Calculate split sizes
    print("\n  [3/4] Splitting into train / val / test ...")

    def get_split_counts(n):
        ntr = int(n * TRAIN_RATIO)
        nva = int(n * VAL_RATIO)
        return {"train": ntr, "val": nva, "test": n - ntr - nva}

    pad_counts   = get_split_counts(len(pad_crops))
    nopad_counts = get_split_counts(len(nopad_crops))

    # Step 4 — Save
    print("\n  [4/4] Saving crops ...")
    random.seed(RANDOM_SEED)
    save_split(pad_crops,   "pad",    pad_counts)
    save_split(nopad_crops, "no_pad", nopad_counts)

    # Summary
    print("\n" + "─" * 60)
    print("  DATASET READY")
    print("─" * 60)
    for split in ("train", "val", "test"):
        p = len(list((Path(DATASET_DIR) / split / "pad").glob("*.jpg")))
        n = len(list((Path(DATASET_DIR) / split / "no_pad").glob("*.jpg")))
        print(f"  {split:6s}  pad={p:>4}   no_pad={n:>4}   total={p+n:>4}")

    print(f"\n  Dataset saved to  →  '{DATASET_DIR}/'")
    print(f"\n  Next step: python module2_train.py\n")


if __name__ == "__main__":
    prepare_dataset()
