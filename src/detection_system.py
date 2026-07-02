"""
================================================================================
AHP WASTE SEGREGATION — DETECTION SYSTEM
================================================================================
This file does everything:
  1. Loads your trained model
  2. Opens the camera
  3. Detects sanitary pads in real time
  4. Prints clean output to terminal every second:
        RESULT=1  X=312  Y=218   (pad detected)
        RESULT=0             (not a pad)

HOW TO INTEGRATE WITH YOUR FLAP CODE:
  Import the get_detection() function from this file:
      from detection_system import get_detection
      result = get_detection()
      if result["output"] == 1:
          # trigger your flap here
          print("Flap ON  at X=", result["x"], "Y=", result["y"])

HOW TO RUN STANDALONE:
  Just run this file directly:
      python detection_system.py

REQUIREMENTS (install once):
  pip install torch torchvision opencv-python Pillow numpy
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import time
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS  ← Edit only these values if needed
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE     = "ahp_model.pth"       # trained model (from module2_train.py)
EMBED_FILE     = "ahp_embeddings.npy"  # training embeddings (from module2_train.py)
LOG_FILE       = "detection_output.json"

CAMERA_ID      = 0      # 0 = default webcam. Change to 1 or 2 if needed.
CAM_WIDTH      = 640
CAM_HEIGHT     = 480

# Thresholds — all 3 must pass for RESULT=1
# Lower values = easier to detect (use if pad is being missed)
# Higher values = stricter (use if wrong objects are detected)
CONF_THRESHOLD = 0.55   # CNN confidence
SIM_THRESHOLD  = 0.60   # similarity to dataset images
TEX_THRESHOLD  = 0.30   # texture score

EMBED_DIM      = 256
OUTPUT_EVERY   = 1.0    # print result to terminal every N seconds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITION
#  Must match exactly what was used in module2_train.py
# ─────────────────────────────────────────────────────────────────────────────

class AHPClassifier(nn.Module):
    """
    MobileNetV2-based binary classifier.
    Output: logits for [no_pad=0, pad=1] + 256D embedding vector
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        base            = models.mobilenet_v2(weights=None)
        self.backbone   = base.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, embed_dim), nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        features  = self.pool(self.backbone(x))
        embedding = self.embed_head(features)
        logits    = self.classifier(embedding)
        return logits, embedding


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE TRANSFORM — applied to every crop before CNN inference
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """
    Load trained model weights and training embeddings from disk.
    Call this once at the start of your program.
    Returns: (model, db_embeddings)
    """
    if not Path(MODEL_FILE).exists():
        print("=" * 55)
        print("  ERROR: Model file not found!")
        print(f"  Expected: '{MODEL_FILE}'")
        print("  Run module2_train.py first to train the model.")
        print("=" * 55)
        exit(1)

    model = AHPClassifier(EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()  # set to inference mode (disables dropout)

    db_embeddings = None
    if Path(EMBED_FILE).exists():
        db_embeddings = np.load(EMBED_FILE)

    print("=" * 55)
    print("  AHP DETECTION SYSTEM — READY")
    print("=" * 55)
    print(f"  Model    : {MODEL_FILE}  [{DEVICE}]")
    if db_embeddings is not None:
        print(f"  Dataset  : {len(db_embeddings)} training embeddings loaded")
    print(f"  Camera   : ID {CAMERA_ID}")
    print(f"  Output   : every {OUTPUT_EVERY} second(s)")
    print("=" * 55)
    print()

    return model, db_embeddings


# ─────────────────────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn(model, crop_bgr):
    """
    Run the CNN on one image crop.

    Input : crop_bgr — a BGR image (numpy array from OpenCV)
    Output: confidence (0.0–1.0), embedding (256D numpy array)

    confidence > 0.5 means the model thinks it's a pad.
    The embedding is used for dataset similarity comparison.
    """
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, emb = model(tensor)
        confidence  = torch.softmax(logits, dim=1)[0, 1].item()
    return confidence, emb.cpu().numpy()


def compute_texture(crop_bgr):
    """
    Measure texture of the object.
    Sanitary pads have fibrous texture → higher texture score.

    Returns: float between 0.0 (smooth) and 1.0 (very textured)
    """
    gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(1.0 - np.exp(-variance / 400.0), 0.0, 1.0))


def compute_similarity(query_emb, db_embeddings, top_k=5):
    """
    Compare the query embedding against all training image embeddings.
    Uses cosine similarity — higher means object looks more like training data.

    Returns: float between 0.0 (nothing like dataset) and 1.0 (identical)
    """
    if db_embeddings is None:
        return 1.0  # skip check if no embeddings available

    db_norm = db_embeddings / (
        np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-8)
    q_norm  = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    sims    = (db_norm @ q_norm.T).flatten()
    top_k   = min(top_k, len(sims))
    return float(np.mean(np.sort(sims)[-top_k:]))


def find_objects(frame):
    """
    Detect candidate object regions in a camera frame.
    Uses LAB colour space + Otsu thresholding + morphological cleanup.

    Returns list of (bx, by, bw, bh, cx, cy):
        bx, by = top-left corner in pixels
        bw, bh = width and height in pixels
        cx, cy = centre of object in pixels
    """
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    blur   = cv2.GaussianBlur(lab[:, :, 0], (7, 7), 0)
    _, th  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    th     = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th     = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w    = frame.shape[:2]
    objects = []

    for c in cnts:
        area = cv2.contourArea(c)
        # Ignore too-small or too-large regions
        if area < w * h * 0.02 or area > w * h * 0.80:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        objects.append((bx, by, bw, bh, bx + bw // 2, by + bh // 2))

    return objects


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE FRAME ANALYSIS
#  ← USE THIS FUNCTION TO INTEGRATE WITH YOUR FLAP CODE
# ─────────────────────────────────────────────────────────────────────────────

def get_detection(frame, model, db_embeddings,
                  conf_t=CONF_THRESHOLD,
                  sim_t=SIM_THRESHOLD,
                  tex_t=TEX_THRESHOLD):
    """
    Analyse one camera frame and return the detection result.

    HOW TO USE IN YOUR FLAP CODE:
    ─────────────────────────────
        from detection_system import load_model, get_detection
        import cv2

        model, db_emb = load_model()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            result = get_detection(frame, model, db_emb)

            if result["output"] == 1:
                print("PAD DETECTED at X=", result["x"], "Y=", result["y"])
                # → trigger your flap here

    RETURN VALUE:
    ─────────────
        {
            "output"     : 1,        # 1 = sanitary pad,  0 = not a pad
            "x"          : 312,      # centre X pixel  (None if output=0)
            "y"          : 218,      # centre Y pixel  (None if output=0)
            "confidence" : 0.91,     # CNN score (0–1)
            "similarity" : 0.87,     # dataset match score (0–1)
            "texture"    : 0.74,     # texture score (0–1)
            "timestamp"  : 1712345678.1
        }
    """
    fh, fw  = frame.shape[:2]
    objects = find_objects(frame)

    best_output     = None
    best_confidence = 0.0

    for (bx, by, bw, bh, cx, cy) in objects:
        # Crop object with padding
        pad  = 12
        y1   = max(0,  by - pad);  y2 = min(fh, by + bh + pad)
        x1   = max(0,  bx - pad);  x2 = min(fw, bx + bw + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Run CNN
        confidence, embedding = run_cnn(model, crop)

        # Additional checks
        texture    = compute_texture(crop)
        similarity = compute_similarity(embedding, db_embeddings)
        area_ratio = (bw * bh) / (fw * fh)
        size_ok    = 0.02 <= area_ratio <= 0.80

        # Final decision — ALL must pass
        is_pad = (confidence >= conf_t and
                  texture    >= tex_t   and
                  similarity >= sim_t   and
                  size_ok)

        if is_pad and confidence > best_confidence:
            best_confidence = confidence
            best_output = {
                "output"     : 1,
                "x"          : cx,
                "y"          : cy,
                "width"      : bw,
                "height"     : bh,
                "confidence" : round(confidence, 4),
                "texture"    : round(texture,    4),
                "similarity" : round(similarity, 4),
                "timestamp"  : round(time.time(), 3),
            }

    # If no pad found, return output=0
    if best_output is None:
        best_output = {
            "output"     : 0,
            "x"          : None,
            "y"          : None,
            "confidence" : 0.0,
            "texture"    : 0.0,
            "similarity" : 0.0,
            "timestamp"  : round(time.time(), 3),
        }

    return best_output


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW RESULT ON FRAME
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(frame, result, objects, model, db_embeddings,
                conf_t, sim_t, tex_t):
    """
    Draw bounding boxes, axis lines, and scores on the frame.
    Returns annotated frame for display.
    """
    display = frame.copy()
    fh, fw  = frame.shape[:2]

    for (bx, by, bw, bh, cx, cy) in objects:
        pad  = 12
        y1   = max(0,  by - pad);  y2 = min(fh, by + bh + pad)
        x1   = max(0,  bx - pad);  x2 = min(fw, bx + bw + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        confidence, embedding = run_cnn(model, crop)
        texture               = compute_texture(crop)
        similarity            = compute_similarity(embedding, db_embeddings)
        area_ratio            = (bw * bh) / (fw * fh)
        size_ok               = 0.02 <= area_ratio <= 0.80

        is_pad = (confidence >= conf_t and texture >= tex_t and
                  similarity >= sim_t  and size_ok)

        colour    = (0, 230, 0)  if is_pad else (20, 20, 210)
        thickness = 3            if is_pad else 2

        # Bounding box
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), colour, thickness)

        # X axis line (horizontal through centre)
        cv2.line(display, (bx, cy), (bx + bw, cy), colour, 2)
        # Y axis line (vertical through centre)
        cv2.line(display, (cx, by), (cx, by + bh), colour, 2)
        # Centre dot
        cv2.circle(display, (cx, cy), 7, colour, -1)

        # Label
        label = "SANITARY PAD  OUTPUT=1" if is_pad else "NOT PAD  OUTPUT=0"
        (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        bg = (0, 150, 0) if is_pad else (150, 0, 0)
        cv2.rectangle(display, (bx, by - th2 - 12), (bx + tw + 8, by), bg, -1)
        cv2.putText(display, label, (bx + 4, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Scores to the right of box
        for i, (lbl, val, thr) in enumerate([
            ("Conf", confidence, conf_t),
            ("Tex",  texture,    tex_t),
            ("Sim",  similarity, sim_t),
        ]):
            col = (50, 220, 50) if val >= thr else (50, 50, 210)
            cv2.putText(display, f"{lbl}: {val:.2f}",
                        (bx + bw + 8, by + 22 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

        # X Y coordinates near centre
        cv2.putText(display, f"X={cx}  Y={cy}",
                    (cx + 8, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 2)

    # Bottom status bar
    bar_col = (0, 200, 0) if result["output"] == 1 else (0, 0, 200)
    if result["output"] == 1:
        status = (f"OUTPUT=1  SANITARY PAD  "
                  f"X={result['x']}  Y={result['y']}  "
                  f"Conf={result['confidence']:.2f}")
    else:
        status = "OUTPUT=0  No sanitary pad detected"

    cv2.rectangle(display, (0, fh - 36), (fw, fh), (15, 15, 15), -1)
    cv2.putText(display, status, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, bar_col, 2)

    return display


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — runs when you execute this file directly
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load model once
    model, db_embeddings = load_model()

    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera {CAMERA_ID}")
        print(f"  Change CAMERA_ID = 1 or 2 at the top of this file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prevents frame lag

    conf_t = CONF_THRESHOLD
    sim_t  = SIM_THRESHOLD
    tex_t  = TEX_THRESHOLD

    detection_log = []
    last_print    = time.time()
    fps_counter   = fps_timer = fps = 0

    print("  Camera open. Detection running.")
    print("  Press Q to quit  |  + to lower thresholds  |  - to raise\n")
    print(f"  {'TIME':>10}   OUTPUT   {'X':>5}   {'Y':>5}   CONF   SIM    TEX")
    print("  " + "─" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS
        fps_counter += 1
        if fps_counter >= 20:
            fps       = fps_counter / max(time.time() - fps_timer, 0.001)
            fps_timer = time.time()
            fps_counter = 0

        # Get objects in frame
        objects = find_objects(frame)

        # Run detection
        result = get_detection(frame, model, db_embeddings, conf_t, sim_t, tex_t)

        # Print to terminal every OUTPUT_EVERY seconds
        now = time.time()
        if now - last_print >= OUTPUT_EVERY:
            last_print = now
            t_str = time.strftime("%H:%M:%S")

            if result["output"] == 1:
                print(f"  {t_str}   OUTPUT=1"
                      f"   X={result['x']:>4}"
                      f"   Y={result['y']:>4}"
                      f"   {result['confidence']:.2f}"
                      f"   {result['similarity']:.2f}"
                      f"   {result['texture']:.2f}"
                      f"   ← SANITARY PAD")
                detection_log.append(result)
                with open(LOG_FILE, "w") as f:
                    json.dump(detection_log, f, indent=2)
            else:
                print(f"  {t_str}   OUTPUT=0"
                      f"   X= ---   Y= ---"
                      f"                     ← no pad")

        # Draw on frame
        display = draw_result(frame, result, objects, model,
                              db_embeddings, conf_t, sim_t, tex_t)

        # FPS on screen
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 0), 1)

        cv2.imshow("AHP Detection  [Q=quit  +=easier  -=harder]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('+'), ord('=')):
            conf_t = max(0.25, conf_t - 0.05)
            sim_t  = max(0.25, sim_t  - 0.05)
            tex_t  = max(0.15, tex_t  - 0.05)
            print(f"\n  Thresholds lowered →  Conf≥{conf_t:.2f}  "
                  f"Sim≥{sim_t:.2f}  Tex≥{tex_t:.2f}\n")
        elif key == ord('-'):
            conf_t = min(0.95, conf_t + 0.05)
            sim_t  = min(0.95, sim_t  + 0.05)
            tex_t  = min(0.95, tex_t  + 0.05)
            print(f"\n  Thresholds raised  →  Conf≥{conf_t:.2f}  "
                  f"Sim≥{sim_t:.2f}  Tex≥{tex_t:.2f}\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Stopped. {len(detection_log)} pad detections saved → '{LOG_FILE}'\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
