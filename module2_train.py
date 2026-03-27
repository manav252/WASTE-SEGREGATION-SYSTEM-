"""
================================================================================
MODULE 2 — CNN MODEL TRAINING
================================================================================
Purpose : Trains a MobileNetV2-based binary classifier on your labeled dataset.
          MobileNetV2 is chosen because:
            • Very fast inference (ideal for real-time conveyor belt)
            • Small model size (~3.4M parameters)
            • Pretrained on ImageNet — strong feature extractor out of the box
            • Used in real industrial vision systems

The model learns two things simultaneously:
    1. Binary classification  →  pad (1)  or  no_pad (0)
    2. Feature embedding      →  a 256-D vector representing the object's
                                  visual features (used for dataset comparison)

After training the model saves:
    ahp_model.pth          ← trained model weights
    ahp_embeddings.npy     ← 256-D feature vectors for all training images
    training_log.json      ← epoch-by-epoch accuracy / loss log

Run:
    python module2_train.py

Expected training time:
    CPU only  : ~15–30 minutes (40 epochs)
    GPU       : ~3–8 minutes
================================================================================
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DATASET_DIR  = "ahp_dataset"       # output from module1_prepare_dataset.py
MODEL_FILE   = "ahp_model.pth"     # where trained weights are saved
EMBED_FILE   = "ahp_embeddings.npy"
LOG_FILE     = "training_log.json"
EMBED_DIM    = 256                 # size of feature vector
EPOCHS       = 40                  # increase to 60 for higher accuracy
BATCH_SIZE   = 16
LEARNING_RATE = 0.0005
PATIENCE     = 12                  # stop early if val accuracy stops improving
IMG_SIZE     = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


# ── IMAGE TRANSFORMS ──────────────────────────────────────────────────────────

# Training: heavy augmentation to prevent overfitting on 190 images
TRAIN_TRANSFORM = T.Compose([
    T.Resize((240, 240)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                std= [0.229, 0.224, 0.225]),   # ImageNet std
    T.RandomErasing(p=0.15),                   # randomly erase small patches
])

# Validation / Test / Inference: no augmentation, just resize + normalise
EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


# ── DATASET CLASS ─────────────────────────────────────────────────────────────

class AHPDataset(Dataset):
    """
    Loads images from ahp_dataset/split/class_name/ structure.
    Label mapping:  pad → 1,   no_pad → 0
    """

    LABEL_MAP = {"pad": 1, "no_pad": 0}

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []   # list of (path, label) tuples

        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: '{root_dir}'\n"
                f"Run module1_prepare_dataset.py first."
            )

        for class_name, label in self.LABEL_MAP.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((str(img_path), label))

        if not self.samples:
            raise FileNotFoundError(f"No images found in '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────

class AHPClassifier(nn.Module):
    """
    MobileNetV2 backbone + custom classification head.

    Architecture:
        MobileNetV2 (pretrained, layers 1-14 frozen)
            ↓
        Global Average Pooling  →  1280-D feature vector
            ↓
        Embedding head  →  256-D feature vector  (used for similarity comparison)
            ↓
        Classifier head →  2 logits  (pad=1, no_pad=0)

    The 256-D embedding is what gets saved to ahp_embeddings.npy and used
    later to compare live camera crops against training images.
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        # Load MobileNetV2 pretrained on ImageNet
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze early layers (keep ImageNet features, only fine-tune late layers)
        # MobileNetV2 has 19 feature blocks; freeze first 14
        for i, layer in enumerate(base.features):
            if i < 14:
                for param in layer.parameters():
                    param.requires_grad = False

        # Keep the feature extractor, remove the original classifier
        self.backbone = base.features          # outputs (batch, 1280, 7, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1) # → (batch, 1280, 1, 1)

        # Embedding head: 1280 → 256
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim),
        )

        # Binary classifier: 256 → 2  (pad / no_pad)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """
        Returns:
            logits    : raw scores for [no_pad, pad]  — shape (B, 2)
            embedding : 256-D feature vector           — shape (B, 256)
        """
        features  = self.pool(self.backbone(x))
        embedding = self.embed_head(features)
        logits    = self.classifier(embedding)
        return logits, embedding


# ── TRAINING HELPERS ──────────────────────────────────────────────────────────

def evaluate(model, dataloader):
    """Run model on a dataloader. Returns accuracy (0-1)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, _    = model(imgs)
            preds        = logits.argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
    return correct / max(total, 1)


def build_training_embeddings(model, dataset):
    """
    Run all training images through the model and save their 256-D embeddings.
    These are used later (in module3 and module4) to compare live crops
    against known sanitary pad images.
    """
    print("\n  Building feature embeddings for training images ...")
    model.eval()
    dl  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_embeddings = []

    with torch.no_grad():
        for imgs, _ in dl:
            _, emb = model(imgs.to(DEVICE))
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    np.save(EMBED_FILE, embeddings)
    print(f"  Saved {len(embeddings)} embeddings → '{EMBED_FILE}'")


# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  MODULE 2 — CNN MODEL TRAINING")
    print("=" * 60)
    print(f"\n  Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Model      : MobileNetV2 (pretrained)")

    # ── Load datasets ─────────────────────────────────────────
    train_ds = AHPDataset(f"{DATASET_DIR}/train", TRAIN_TRANSFORM)
    val_ds   = AHPDataset(f"{DATASET_DIR}/val",   EVAL_TRANSFORM)
    test_ds  = AHPDataset(f"{DATASET_DIR}/test",  EVAL_TRANSFORM)

    # Count class distribution
    pad_count   = sum(1 for _, l in train_ds.samples if l == 1)
    nopad_count = sum(1 for _, l in train_ds.samples if l == 0)
    print(f"\n  Train: {len(train_ds)} images  (pad={pad_count}, no_pad={nopad_count})")
    print(f"  Val  : {len(val_ds)} images")
    print(f"  Test : {len(test_ds)} images\n")

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build model ───────────────────────────────────────────
    model = AHPClassifier(EMBED_DIM).to(DEVICE)

    # Only optimise parameters that are NOT frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable):,}")

    # ── Loss and optimiser ────────────────────────────────────
    # Use class weights to handle any imbalance between pad / no_pad
    total = pad_count + nopad_count
    weights = torch.tensor([total / max(nopad_count, 1),
                             total / max(pad_count, 1)],
                            dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)

    # OneCycleLR gives fast convergence and good generalisation
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE * 10,
        steps_per_epoch=len(train_dl), epochs=EPOCHS
    )

    # ── Training loop ─────────────────────────────────────────
    best_val_acc  = 0.0
    patience_ctr  = 0
    training_log  = []

    print(f"  {'Ep':>3}  {'Loss':>8}  {'Train':>7}  {'Val':>7}  {'Time':>6}  Note")
    print("  " + "─" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = correct = total = 0
        t_start = time.time()

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss      = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / max(total, 1)
        val_acc   = evaluate(model, val_dl)
        elapsed   = time.time() - t_start
        avg_loss  = epoch_loss / max(len(train_dl), 1)

        # Save if best
        note = ""
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_FILE)
            note = "✅ best saved"
        else:
            patience_ctr += 1
            note = f"patience {patience_ctr}/{PATIENCE}"

        print(f"  {epoch:>3}  {avg_loss:>8.4f}  {train_acc:>7.3f}  {val_acc:>7.3f}  {elapsed:>5.1f}s  {note}")

        training_log.append({
            "epoch": epoch, "loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
        })

        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # ── Final test evaluation ──────────────────────────────────
    print("\n" + "─" * 60)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    test_acc = evaluate(model, test_dl)

    print(f"  Best validation accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"  Final test accuracy      : {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # ── Save training log ──────────────────────────────────────
    log_data = {
        "best_val_acc": round(best_val_acc, 4),
        "test_acc":     round(test_acc, 4),
        "epochs_run":   len(training_log),
        "model":        "MobileNetV2",
        "embed_dim":    EMBED_DIM,
        "history":      training_log,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  Training log saved       : '{LOG_FILE}'")

    # ── Build and save embeddings ──────────────────────────────
    build_training_embeddings(model, train_ds)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model saved        →  '{MODEL_FILE}'")
    print(f"  Embeddings saved   →  '{EMBED_FILE}'")
    print(f"\n  Next step: python module3_live_detection.py")
    print(f"         OR: python module4_snapshot_detection.py\n")


if __name__ == "__main__":
    train()
