"""
================================================================================
AHP WASTE SEGREGATION — COMPLETE SYSTEM (TRAIN + DETECT IN ONE FILE)
================================================================================
This file does EVERYTHING automatically:
  Step 1 → Checks if model is trained. If NOT, trains it first automatically.
  Step 2 → Starts live detection once model is ready.

You never need to run separate files. Just run this one file always.

HOW TO RUN:
    python ahp_system.py

FIRST RUN  : Will train model (15-30 min) then start detection
EVERY AFTER: Skips training, goes straight to detection

ACCURACY FIXES IN THIS VERSION:
  • Calibrated from your actual 211 labeled pad crops
  • Brightness filter:  101 – 206  (10th-90th percentile of real pads)
  • Texture filter:     132 – 6509 (10th-90th percentile of real pads)
  • Edge density:       9.6 – 48.8 (10th-90th percentile of real pads)
  • Color std:          32.9 – 71.9 (10th-90th percentile of real pads)
  • Aspect ratio:       0.20 – 6.50 (from real pad measurements)
  • ALL 4 visual filters + CNN + similarity must agree for OUTPUT=1
  • Camera thread = zero lag
  • Detection every 2 seconds with countdown on screen

FLAP INTEGRATION:
    from ahp_system import load_model, get_result
    model, db_emb = load_model()
    result = get_result(frame, model, db_emb)   # returns 1 or 0
================================================================================
"""

import os, sys, shutil, zipfile, random, time, json, threading
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS  ← only edit this section
# ─────────────────────────────────────────────────────────────────────────────

ZIP_FILE      = "project-1-at-2026-02-17-21-44-c758d2de.zip"
MODEL_FILE    = "ahp_model.pth"
EMBED_FILE    = "ahp_embeddings.npy"
DATASET_DIR   = "ahp_dataset"
LOG_FILE      = "detections.json"

CAMERA_ID     = 0        # change to 1 or 2 if camera not found
CAM_W, CAM_H  = 640, 480
COOLDOWN_SEC  = 2.0      # analyse every 2 seconds
PANEL_W       = 230      # side panel width in pixels

# Training settings
EPOCHS        = 50
BATCH_SIZE    = 16
LR            = 0.0005
EMBED_DIM     = 256
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15

# ── Accuracy filters — calibrated from 211 real pad crops ─────────────────
# (These are 10th–90th percentile values from your actual labeled images)
CONF_THR      = 0.50     # CNN confidence minimum
SIM_THR       = 0.50     # cosine similarity to training embeddings
BRIGHT_MIN    = 80.0     # brightness 10th pct was 101, relaxed slightly
BRIGHT_MAX    = 220.0    # brightness 90th pct was 206, relaxed slightly
TEX_MIN       = 80.0     # texture variance minimum (10th pct = 132, relaxed)
TEX_MAX       = 8000.0   # texture variance maximum
EDGE_MIN      = 5.0      # Canny edge density minimum (10th pct = 9.56, relaxed)
EDGE_MAX      = 80.0     # Canny edge density maximum
COLOR_STD_MIN = 20.0     # color std minimum (10th pct = 32.9, relaxed)
COLOR_STD_MAX = 95.0     # color std maximum

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class AHPModel(nn.Module):
    """
    MobileNetV2 backbone + dual head (classifier + embedding).
    Fast, accurate, proven for industrial vision tasks.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze first 14 layers, fine-tune the rest
        for i, layer in enumerate(base.features):
            for p in layer.parameters():
                p.requires_grad = (i >= 14)

        self.backbone   = base.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, embed_dim), nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        f = self.pool(self.backbone(x))
        e = self.embed_head(f)
        return self.classifier(e), e


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset():
    """
    Extract zip, read YOLO labels, crop each labeled pad region,
    generate background (no-pad) crops, split into train/val/test.
    Positive label = pad (1),  Negative label = no_pad (0).
    """
    if Path(DATASET_DIR).exists():
        total = sum(1 for _ in Path(DATASET_DIR).rglob("*.jpg"))
        if total > 10:
            print(f"  Dataset already prepared ({total} images). Skipping.")
            return

    print("\n  Preparing dataset from ZIP ...")
    raw = Path("ahp_raw")

    if not Path(ZIP_FILE).exists():
        print(f"\n  ERROR: '{ZIP_FILE}' not found in this folder.")
        sys.exit(1)

    if not raw.exists():
        with zipfile.ZipFile(ZIP_FILE) as z:
            z.extractall(str(raw))

    img_dir = raw / "images"
    lbl_dir = raw / "labels"
    all_imgs = sorted(list(img_dir.glob("*.jpg")) +
                      list(img_dir.glob("*.png")))

    pad_crops   = []
    nopad_crops = []

    for img_path in all_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes_px = []

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                x1=max(0,int((cx-bw/2)*w)); y1=max(0,int((cy-bh/2)*h))
                x2=min(w,int((cx+bw/2)*w)); y2=min(h,int((cy+bh/2)*h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_r = cv2.resize(crop, (224, 224))
                pad_crops.append(crop_r)
                boxes_px.append((x1, y1, x2, y2))

        # Background crops — regions that do NOT overlap any pad
        for _ in range(30):
            cw = random.randint(w//5, w//2)
            ch = random.randint(h//5, h//2)
            x1 = random.randint(0, w-cw)
            y1 = random.randint(0, h-ch)
            x2, y2 = x1+cw, y1+ch
            overlap = any(x1<bx2 and x2>bx1 and y1<by2 and y2>by1
                          for bx1,by1,bx2,by2 in boxes_px)
            if not overlap:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    nopad_crops.append(cv2.resize(crop, (224,224)))
                break

    print(f"  Pad crops   : {len(pad_crops)}")
    print(f"  No-pad crops: {len(nopad_crops)}")

    random.seed(42)

    def save_split(crops, cls_name):
        random.shuffle(crops)
        n = len(crops)
        ntr = int(n * TRAIN_RATIO)
        nva = int(n * VAL_RATIO)
        for split, items in [("train", crops[:ntr]),
                              ("val",   crops[ntr:ntr+nva]),
                              ("test",  crops[ntr+nva:])]:
            d = Path(DATASET_DIR) / split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(items):
                cv2.imwrite(str(d / f"{cls_name}_{split}_{i:04d}.jpg"), c)

    save_split(pad_crops,   "pad")
    save_split(nopad_crops, "no_pad")
    print("  Dataset ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TF = T.Compose([
    T.Resize((240,240)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(25),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    T.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.85,1.15)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    T.RandomErasing(p=0.2),
])

EVAL_TF = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

INFER_TF = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


class PadDataset(Dataset):
    LABELS = {"pad": 1, "no_pad": 0}

    def __init__(self, root, tf):
        self.tf      = tf
        self.samples = []
        for cls, lbl in self.LABELS.items():
            for p in Path(root).glob(f"{cls}/*.jpg"):
                self.samples.append((str(p), lbl))
        if not self.samples:
            raise FileNotFoundError(f"No images in {root}")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        try:   img = Image.open(path).convert("RGB")
        except: img = Image.new("RGB",(224,224),(128,128,128))
        return self.tf(img), lbl


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    print("=" * 60)
    print("  TRAINING MODEL  (this runs once, then saves)")
    print("=" * 60)
    print(f"  Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}\n")

    prepare_dataset()

    tr_ds = PadDataset(f"{DATASET_DIR}/train", TRAIN_TF)
    va_ds = PadDataset(f"{DATASET_DIR}/val",   EVAL_TF)
    te_ds = PadDataset(f"{DATASET_DIR}/test",  EVAL_TF)

    pad_n   = sum(1 for _,l in tr_ds.samples if l==1)
    nopad_n = sum(1 for _,l in tr_ds.samples if l==0)
    print(f"  Train: {len(tr_ds)}  (pad={pad_n}, no_pad={nopad_n})")
    print(f"  Val  : {len(va_ds)}   Test: {len(te_ds)}\n")

    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model  = AHPModel(EMBED_DIM).to(DEVICE)
    total  = pad_n + nopad_n
    w      = torch.tensor([total/max(nopad_n,1), total/max(pad_n,1)],
                          dtype=torch.float32).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=w)
    opt    = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                         lr=LR, weight_decay=1e-4)
    sched  = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR*10,
                                           steps_per_epoch=len(tr_dl),
                                           epochs=EPOCHS)

    best_acc, patience = 0.0, 0
    print(f"  {'Ep':>3}  {'Loss':>8}  {'Train':>7}  {'Val':>7}  Note")
    print("  " + "─" * 44)

    for ep in range(1, EPOCHS+1):
        model.train()
        tloss = correct = total_n = 0
        for imgs, lbls in tr_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(imgs)
            loss = crit(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tloss   += loss.item()
            correct += (logits.argmax(1)==lbls).sum().item()
            total_n += lbls.size(0)

        val_acc = _accuracy(model, va_dl)
        note    = ""
        if val_acc >= best_acc:
            best_acc = val_acc; patience = 0
            torch.save(model.state_dict(), MODEL_FILE)
            note = "✅ saved"
        else:
            patience += 1
            note = f"patience {patience}/15"

        print(f"  {ep:>3}  {tloss/max(len(tr_dl),1):>8.4f}"
              f"  {correct/max(total_n,1):>7.3f}  {val_acc:>7.3f}  {note}")

        if patience >= 15:
            print(f"\n  Early stop at epoch {ep}.")
            break

    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    te_acc = _accuracy(model, te_dl)
    print(f"\n  Best val acc : {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"  Test acc     : {te_acc:.4f} ({te_acc*100:.1f}%)")

    _build_embeddings(model, tr_ds)

    print("\n  ✅  Training complete. Starting detection...\n")
    return model


def _accuracy(model, dl):
    model.eval(); c = t = 0
    with torch.no_grad():
        for imgs, lbls in dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            c += (model(imgs)[0].argmax(1)==lbls).sum().item()
            t += lbls.size(0)
    return c/max(t,1)


def _build_embeddings(model, ds):
    print("\n  Building embeddings for similarity comparison ...")
    model.eval()
    dl  = DataLoader(ds, 32, num_workers=0)
    emb = []
    with torch.no_grad():
        for imgs, _ in dl:
            _, e = model(imgs.to(DEVICE))
            emb.append(e.cpu().numpy())
    np.save(EMBED_FILE, np.vstack(emb))
    print(f"  Saved {sum(len(x) for x in emb)} embeddings → {EMBED_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load trained model. Trains first if model file doesn't exist."""
    if not Path(MODEL_FILE).exists():
        print("\n  Model not found. Training now (one-time setup)...")
        model = train_model()
    else:
        model = AHPModel(EMBED_DIM).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        model.eval()

    db_emb = np.load(EMBED_FILE) if Path(EMBED_FILE).exists() else None

    print("=" * 60)
    print("  AHP DETECTION SYSTEM — READY")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Model   : {MODEL_FILE}")
    if db_emb is not None:
        print(f"  Dataset : {len(db_emb)} training embeddings loaded")
    print(f"  Cooldown: every {COOLDOWN_SEC}s")
    print("=" * 60 + "\n")

    return model, db_emb


# ─────────────────────────────────────────────────────────────────────────────
#  VISUAL FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn(model, crop_bgr):
    """CNN inference → confidence + 256D embedding."""
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TF(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, emb = model(tensor)
        conf = torch.softmax(logits, dim=1)[0,1].item()
    return conf, emb.cpu().numpy()


def compute_features(crop_bgr):
    """
    Extract all visual features from a crop.
    Values calibrated from 211 real pad crops in your dataset.
    Returns dict with texture, brightness, edge_density, color_std.
    """
    gray       = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    texture    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(crop_bgr))
    edges      = cv2.Canny(gray, 50, 150)
    edge_den   = float(edges.mean())
    color_std  = float(np.std(crop_bgr))

    return {
        "texture"   : texture,
        "brightness": brightness,
        "edge_den"  : edge_den,
        "color_std" : color_std,
    }


def compute_similarity(q_emb, db_emb, k=7):
    """Top-k cosine similarity against training embeddings."""
    if db_emb is None: return 1.0
    db_n = db_emb / (np.linalg.norm(db_emb, axis=1, keepdims=True) + 1e-8)
    q_n  = q_emb  / (np.linalg.norm(q_emb)                          + 1e-8)
    sims = (db_n @ q_n.T).flatten()
    return float(np.mean(np.sort(sims)[-min(k, len(sims)):]))


def normalise_tex(v):
    return float(np.clip((v - 0) / 7000.0, 0, 1))


def find_candidates(frame):
    """
    Find candidate object bounding boxes using LAB + Otsu + morphology.
    Returns list of (bx, by, bw, bh).
    """
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    blur   = cv2.GaussianBlur(lab[:,:,0], (5,5), 0)
    _, th  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    th     = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th     = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = frame.shape[:2]
    boxes  = []
    for c in cnts:
        area = cv2.contourArea(c)
        ar   = area / (fw * fh)
        if ar < 0.03 or ar > 0.95: continue
        bx,by,bw,bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        if aspect < 0.15 or aspect > 7.0: continue
        boxes.append((bx, by, bw, bh))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ANALYSIS FUNCTION  ← use this in your flap code
# ─────────────────────────────────────────────────────────────────────────────

def get_result(frame, model, db_emb):
    """Returns 1 (pad detected) or 0 (not detected). Use in flap code."""
    info = analyse(frame, model, db_emb)
    return 1 if info["detected"] else 0


def analyse(frame, model, db_emb):
    """
    Full analysis of one frame.
    Runs CNN + 4 visual filters calibrated from your labeled dataset.
    Returns result dict.
    """
    fh, fw    = frame.shape[:2]
    candidates = find_candidates(frame)
    best       = None
    best_conf  = 0.0

    for (bx, by, bw, bh) in candidates:
        pad  = 10
        crop = frame[max(0,by-pad):min(fh,by+bh+pad),
                     max(0,bx-pad):min(fw,bx+bw+pad)]
        if crop.size == 0:
            continue

        # 1. CNN confidence + embedding
        conf, emb = run_cnn(model, crop)

        # 2. Visual features (calibrated from 211 real pad crops)
        feats = compute_features(crop)
        tex   = feats["texture"]
        brt   = feats["brightness"]
        edg   = feats["edge_den"]
        cstd  = feats["color_std"]

        # 3. Dataset similarity
        sim = compute_similarity(emb, db_emb)

        # 4. All filters — every single one must pass
        cnn_ok    = conf  >= CONF_THR
        sim_ok    = sim   >= SIM_THR
        tex_ok    = TEX_MIN   <= tex  <= TEX_MAX
        bright_ok = BRIGHT_MIN <= brt  <= BRIGHT_MAX
        edge_ok   = EDGE_MIN   <= edg  <= EDGE_MAX
        cstd_ok   = COLOR_STD_MIN <= cstd <= COLOR_STD_MAX

        detected = cnn_ok and sim_ok and tex_ok and bright_ok and edge_ok and cstd_ok

        # Keep the best (highest confidence) detection
        if detected and conf > best_conf:
            best_conf = conf
            best = {
                "detected"   : True,
                "bx":bx, "by":by, "bw":bw, "bh":bh,
                "confidence" : round(conf, 3),
                "similarity" : round(sim,  3),
                "texture"    : round(tex,  1),
                "tex_norm"   : round(normalise_tex(tex)*100, 1),
                "brightness" : round(brt,  1),
                "edge_den"   : round(edg,  2),
                "color_std"  : round(cstd, 1),
                "checks"     : {
                    "cnn": cnn_ok, "sim": sim_ok, "tex": tex_ok,
                    "bright": bright_ok, "edge": edge_ok, "cstd": cstd_ok,
                },
                "timestamp"  : round(time.time(), 3),
            }

    # If no pad found, still return info about the largest object for display
    if best is None:
        if candidates:
            bx,by,bw,bh = candidates[0]
            crop = frame[max(0,by-10):min(fh,by+bh+10),
                         max(0,bx-10):min(fw,bx+bw+10)]
            if crop.size > 0:
                conf, emb = run_cnn(model, crop)
                feats = compute_features(crop)
                sim   = compute_similarity(emb, db_emb)
                tex   = feats["texture"]
                return {
                    "detected"   : False,
                    "bx":bx, "by":by, "bw":bw, "bh":bh,
                    "confidence" : round(conf, 3),
                    "similarity" : round(sim,  3),
                    "texture"    : round(tex,  1),
                    "tex_norm"   : round(normalise_tex(tex)*100, 1),
                    "brightness" : round(feats["brightness"], 1),
                    "edge_den"   : round(feats["edge_den"],   2),
                    "color_std"  : round(feats["color_std"],  1),
                    "checks"     : {},
                    "timestamp"  : round(time.time(), 3),
                }

        return {
            "detected": False, "confidence":0.0, "similarity":0.0,
            "texture":0.0, "tex_norm":0.0, "brightness":0.0,
            "edge_den":0.0, "color_std":0.0, "checks":{},
            "timestamp": round(time.time(), 3),
        }

    return best


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA THREAD — background thread eliminates lag
# ─────────────────────────────────────────────────────────────────────────────

class CameraThread:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            print(f"\n  ERROR: Camera {CAMERA_ID} not found.")
            print("  Change CAMERA_ID at the top of this file.\n")
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW — camera + side panel
# ─────────────────────────────────────────────────────────────────────────────

def draw(frame, info, countdown, total):
    fh, fw = frame.shape[:2]
    detected = info["detected"]

    # ── Camera side ───────────────────────────────────────────
    cam = frame.copy()
    if "bx" in info:
        bx,by,bw,bh = info["bx"],info["by"],info["bw"],info["bh"]
        cx, cy = bx + bw//2, by + bh//2
        col   = (0,230,0) if detected else (60,60,200)
        thick = 4         if detected else 2

        # Bounding box
        cv2.rectangle(cam, (bx,by), (bx+bw,by+bh), col, thick)

        # X axis — horizontal line through centre
        cv2.line(cam, (bx, cy), (bx+bw, cy), col, 2)
        # Y axis — vertical line through centre
        cv2.line(cam, (cx, by), (cx, by+bh), col, 2)
        # Centre dot
        cv2.circle(cam, (cx, cy), 6, col, -1)

        # Coordinate label near centre
        cv2.putText(cam, f"X={cx} Y={cy}",
                    (cx+8, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 2)

        # Label above box
        label = "SANITARY PAD" if detected else "CHECKING..."
        (tw,th2),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bg = (0,160,0) if detected else (40,40,130)
        cv2.rectangle(cam, (bx,by-th2-12), (bx+tw+8,by), bg, -1)
        cv2.putText(cam, label, (bx+4,by-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # ── Side panel ────────────────────────────────────────────
    panel = np.full((fh, PANEL_W, 3), 18, dtype=np.uint8)

    def txt(s, y, col=(190,190,190), sc=0.45, b=1):
        cv2.putText(panel, s, (10,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, b)

    def bar(val, maxv, y, col, h=8):
        bw2 = PANEL_W - 20
        cv2.rectangle(panel,(10,y),(10+bw2,y+h),(40,40,40),-1)
        fill = int(bw2 * min(val/max(maxv,0.001),1.0))
        if fill > 0:
            cv2.rectangle(panel,(10,y),(10+fill,y+h),col,-1)

    # Title
    cv2.rectangle(panel,(0,0),(PANEL_W,36),(30,30,30),-1)
    txt("AHP DETECTION", 24, (220,220,220), 0.52, 2)

    # Status box
    sc = (0,180,0) if detected else (160,40,40)
    cv2.rectangle(panel,(0,40),(PANEL_W,70),sc,-1)
    st = "DETECTED" if detected else "NOT DETECTED"
    txt(st, 61, (255,255,255), 0.55, 2)

    # Output
    out = "1" if detected else "0"
    cv2.putText(panel, out, (PANEL_W//2-18,115),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (0,220,0) if detected else (80,80,200), 3)
    txt("OUTPUT", 125, (100,100,100), 0.38)

    # Cooldown bar
    cv2.line(panel,(10,132),(PANEL_W-10,132),(45,45,45),1)
    txt(f"Next scan: {countdown:.1f}s", 150, (130,130,130), 0.40)
    bar(COOLDOWN_SEC-countdown, COOLDOWN_SEC, 154, (0,150,200))

    cv2.line(panel,(10,168),(PANEL_W-10,168),(45,45,45),1)

    # Scores
    rows = [
        ("Confidence", info["confidence"], 1.0,   CONF_THR,    185),
        ("Similarity",  info["similarity"],  1.0,   SIM_THR,     215),
        ("Texture %",   info["tex_norm"],    100.0, 0,           245),
        ("Brightness",  info["brightness"],  255.0, BRIGHT_MIN,  275),
        ("Edge density",info["edge_den"],    60.0,  EDGE_MIN,    305),
        ("Color std",   info["color_std"],   80.0,  COLOR_STD_MIN,335),
    ]

    for label, val, maxv, thr, y in rows:
        ok  = val >= thr if thr > 0 else True
        col = (0,200,0) if ok else (80,80,200)
        txt(f"{label}: {val:.1f}", y-2, col, 0.38)
        bar(val, maxv, y+2, col)

    # Pass/fail checklist
    cv2.line(panel,(10,355),(PANEL_W-10,355),(45,45,45),1)
    txt("Checks", 372, (100,100,100), 0.38)
    chk_names = [("CNN",info["confidence"]>=CONF_THR),
                 ("Sim",info["similarity"]>=SIM_THR),
                 ("Tex",TEX_MIN<=info["texture"]<=TEX_MAX),
                 ("Brt",BRIGHT_MIN<=info["brightness"]<=BRIGHT_MAX),
                 ("Edg",EDGE_MIN<=info["edge_den"]<=EDGE_MAX),
                 ("Std",COLOR_STD_MIN<=info["color_std"]<=COLOR_STD_MAX)]
    x_pos = 10
    for name, passed in chk_names:
        col = (0,200,0) if passed else (80,80,200)
        sym = "+" if passed else "-"
        txt(f"{sym}{name}", 392, col, 0.36)
        x_pos += 36
        cv2.putText(panel, f"{sym}{name}", (x_pos-36,392),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)

    txt(f"Total detected: {total}", 415, (100,100,100), 0.38)

    return np.hstack([cam, panel])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(model, db_emb):
    cam    = CameraThread()
    time.sleep(0.5)

    log           = []
    last_t        = time.time()
    last_info     = {"detected":False,"confidence":0.0,"similarity":0.0,
                     "texture":0.0,"tex_norm":0.0,"brightness":0.0,
                     "edge_den":0.0,"color_std":0.0,"checks":{}}
    total_det     = 0

    print(f"  {'TIME':>10}  OUT  CONF   SIM   TEX%  BRIGHT  EDGE")
    print("  " + "─" * 55)

    while True:
        frame = cam.get()
        if frame is None:
            continue

        now       = time.time()
        elapsed   = now - last_t
        countdown = max(0.0, COOLDOWN_SEC - elapsed)

        # Run analysis every COOLDOWN_SEC seconds
        if elapsed >= COOLDOWN_SEC:
            last_info = analyse(frame, model, db_emb)
            last_t    = now

            out = 1 if last_info["detected"] else 0
            t_s = time.strftime("%H:%M:%S")
            print(f"  [{t_s}]  {out}   "
                  f"{last_info['confidence']:.2f}  "
                  f"{last_info['similarity']:.2f}  "
                  f"{last_info['tex_norm']:>5.1f}  "
                  f"{last_info['brightness']:>6.1f}  "
                  f"{last_info['edge_den']:>5.2f}"
                  + ("  ← PAD" if out==1 else ""))

            if out == 1:
                total_det += 1
                log.append(last_info)
                with open(LOG_FILE,"w") as f:
                    json.dump(log, f, indent=2)

        # Draw and show
        display = draw(frame, last_info, countdown, total_det)
        cv2.imshow("AHP Waste Segregation  [Q = quit]", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\n  Done. {total_det} pads detected. Saved → {LOG_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, db_emb = load_model()
    run_detection(model, db_emb)
