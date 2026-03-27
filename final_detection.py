"""
================================================================================
AHP WASTE SEGREGATION — FINAL DETECTION SYSTEM
================================================================================
Calibrated using your actual labeled dataset (190 images, 211 annotated pads):

  Dataset measurements used for accuracy:
    Object size  : 8.7% – 96.7% of frame area  (mean 57.7%)
    Object width : 23.3% – 100% of frame width  (mean 76.1%)
    Object height: 16.8% – 99.5% of frame height (mean 74.0%)
    Aspect ratio : 0.27 – 5.61               (mean 1.14 — roughly square/rect)
    Pad colour   : R=158 G=152 B=145         (light coloured, white/cream)
    Texture var  : 17 – 6509                 (mean 782, 25th=225, 75th=540)
    Brightness   : mean 152 out of 255       (light objects)

  These exact values are baked into the detection filters below.

HOW TO RUN:
    python final_detection.py

INTEGRATE WITH FLAP:
    from final_detection import load_model, get_result
    import cv2
    model, db_emb = load_model()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        result = get_result(frame, model, db_emb)
        if result == 1:
            pass  # trigger your flap here

OUTPUT:
    Terminal prints 1 or 0 every 2 seconds
    Camera window shows box + side panel with all scores
================================================================================
"""

import cv2
import time
import json
import threading
import queue
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE    = "ahp_model.pth"
EMBED_FILE    = "ahp_embeddings.npy"
LOG_FILE      = "detections.json"

CAMERA_ID     = 0         # change to 1 or 2 if camera not found
CAM_WIDTH     = 640
CAM_HEIGHT    = 480
COOLDOWN_SEC  = 2.0       # output every 2 seconds

# ── Thresholds calibrated from dataset analysis ────────────────────────────
CONF_THR      = 0.52      # CNN confidence
SIM_THR       = 0.55      # cosine similarity to training embeddings
TEX_MIN       = 17.0      # minimum texture variance (from dataset: min=17.3)
TEX_MAX       = 7000.0    # maximum texture variance (from dataset: max=6509)
BRIGHT_MIN    = 50.0      # minimum brightness (from dataset: min=61)
BRIGHT_MAX    = 240.0     # maximum brightness (from dataset: max=228)
AREA_MIN      = 0.05      # minimum object area ratio (from dataset: 5th pct=0.158 → relaxed)
AREA_MAX      = 0.98      # maximum object area ratio
ASPECT_MIN    = 0.20      # minimum w/h ratio (from dataset: min=0.266 → relaxed)
ASPECT_MAX    = 6.50      # maximum w/h ratio (from dataset: max=5.609 → relaxed)

EMBED_DIM     = 256
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  (must match module2_train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class AHPClassifier(nn.Module):
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
        f = self.pool(self.backbone(x))
        e = self.embed_head(f)
        return self.classifier(e), e


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
    if not Path(MODEL_FILE).exists():
        print("\n  ERROR: ahp_model.pth not found.")
        print("  Run module2_train.py first.\n")
        exit(1)

    model = AHPClassifier(EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    db_emb = np.load(EMBED_FILE) if Path(EMBED_FILE).exists() else None

    print("=" * 55)
    print("  AHP DETECTION SYSTEM — READY")
    print("=" * 55)
    print(f"  Device   : {DEVICE}")
    print(f"  Model    : {MODEL_FILE}")
    if db_emb is not None:
        print(f"  Dataset  : {len(db_emb)} training embeddings")
    print(f"  Cooldown : {COOLDOWN_SEC}s")
    print("=" * 55 + "\n")

    return model, db_emb


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA THREAD  — eliminates lag by reading frames on a separate thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraThread:
    """
    Reads camera frames in a background thread.
    Main thread always gets the LATEST frame — no buffer lag.
    """
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"\n  ERROR: Camera {camera_id} not found.")
            print("  Change CAMERA_ID at the top of this file.\n")
            exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame  = None
        self.lock   = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn(model, crop_bgr):
    """CNN inference. Returns confidence (0-1) and 256D embedding."""
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, emb = model(tensor)
        conf = torch.softmax(logits, dim=1)[0, 1].item()
    return conf, emb.cpu().numpy()


def compute_texture(crop_bgr):
    """
    Laplacian variance — measures surface texture/detail.
    From dataset analysis: pad textures range from 17 to 6509 (mean=782).
    Returns raw variance value.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(crop_bgr):
    """
    Mean pixel brightness.
    From dataset: pads are light coloured, brightness mean=152 (range 61–228).
    """
    return float(np.mean(crop_bgr))


def compute_similarity(query_emb, db_emb, top_k=7):
    """
    Cosine similarity between query and top-k closest training images.
    Higher = more similar to dataset.
    """
    if db_emb is None:
        return 1.0
    db_norm = db_emb       / (np.linalg.norm(db_emb, axis=1, keepdims=True) + 1e-8)
    q_norm  = query_emb    / (np.linalg.norm(query_emb)                      + 1e-8)
    sims    = (db_norm @ q_norm.T).flatten()
    return float(np.mean(np.sort(sims)[-min(top_k, len(sims)):]))


def find_objects(frame):
    """
    Find candidate object regions using LAB + Otsu + morphology.
    Filters by dataset-measured size constraints.
    Returns list of (bx, by, bw, bh, aspect_ratio).
    """
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    blur   = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)
    _, th  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw  = frame.shape[:2]
    objects = []

    for c in cnts:
        area = cv2.contourArea(c)
        ar   = area / (fw * fh)

        # Filter by dataset-calibrated area range
        if ar < AREA_MIN or ar > AREA_MAX:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)

        # Filter by dataset-calibrated aspect ratio range
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue

        objects.append((bx, by, bw, bh, aspect))

    return objects


def normalise_texture(raw_tex):
    """Convert raw Laplacian variance to 0-1 scale using dataset range."""
    return float(np.clip((raw_tex - TEX_MIN) / (TEX_MAX - TEX_MIN), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE FRAME ANALYSIS  ← use this in your flap code
# ─────────────────────────────────────────────────────────────────────────────

def get_result(frame, model, db_emb):
    """
    Analyse one frame. Returns 1 (pad detected) or 0 (not detected).

    Usage in flap code:
        from final_detection import load_model, get_result
        model, db_emb = load_model()
        result = get_result(frame, model, db_emb)
        if result == 1:
            trigger_flap()
    """
    info = analyse_frame(frame, model, db_emb)
    return 1 if info["detected"] else 0


def analyse_frame(frame, model, db_emb):
    """
    Full analysis of one frame.
    Returns dict with detection result and all scores.
    All filters are based on actual dataset measurements.
    """
    fh, fw   = frame.shape[:2]
    objects  = find_objects(frame)
    best     = None
    best_conf = 0.0

    for (bx, by, bw, bh, aspect) in objects:
        pad  = 10
        crop = frame[max(0, by-pad):min(fh, by+bh+pad),
                     max(0, bx-pad):min(fw, bx+bw+pad)]
        if crop.size == 0:
            continue

        # ── CNN ──────────────────────────────────────────────
        conf, emb = run_cnn(model, crop)

        # ── Texture ──────────────────────────────────────────
        raw_tex   = compute_texture(crop)
        tex_norm  = normalise_texture(raw_tex)

        # ── Brightness (pads are light coloured) ─────────────
        brightness = compute_brightness(crop)
        bright_ok  = BRIGHT_MIN <= brightness <= BRIGHT_MAX

        # ── Dataset similarity ────────────────────────────────
        sim = compute_similarity(emb, db_emb)

        # ── Texture range check (from dataset measurements) ───
        tex_ok = TEX_MIN <= raw_tex <= TEX_MAX

        # ── All filters must pass ─────────────────────────────
        detected = (
            conf       >= CONF_THR   and
            sim        >= SIM_THR    and
            tex_ok                   and
            bright_ok
        )

        if detected and conf > best_conf:
            best_conf = conf
            best = {
                "detected"   : True,
                "bx": bx, "by": by, "bw": bw, "bh": bh,
                "aspect"     : round(aspect,   2),
                "confidence" : round(conf,      3),
                "similarity" : round(sim,       3),
                "texture_raw": round(raw_tex,   1),
                "texture_pct": round(tex_norm * 100, 1),
                "brightness" : round(brightness, 1),
                "timestamp"  : round(time.time(), 3),
            }

    if best is None:
        # Return last object's scores even if not detected (for display)
        if objects:
            bx, by, bw, bh, aspect = objects[0]
            crop = frame[max(0,by-10):min(fh,by+bh+10),
                         max(0,bx-10):min(fw,bx+bw+10)]
            if crop.size > 0:
                conf, emb = run_cnn(model, crop)
                raw_tex   = compute_texture(crop)
                sim       = compute_similarity(emb, db_emb)
                brightness = compute_brightness(crop)
                return {
                    "detected"   : False,
                    "bx": bx, "by": by, "bw": bw, "bh": bh,
                    "aspect"     : round(aspect, 2),
                    "confidence" : round(conf,   3),
                    "similarity" : round(sim,    3),
                    "texture_raw": round(raw_tex, 1),
                    "texture_pct": round(normalise_texture(raw_tex)*100, 1),
                    "brightness" : round(brightness, 1),
                    "timestamp"  : round(time.time(), 3),
                }
        return {
            "detected": False,
            "confidence": 0.0, "similarity": 0.0,
            "texture_raw": 0.0, "texture_pct": 0.0,
            "brightness": 0.0, "aspect": 0.0,
            "timestamp": round(time.time(), 3),
        }

    return best


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW — camera window + side panel
# ─────────────────────────────────────────────────────────────────────────────

PANEL_W = 220   # width of side info panel

def draw_frame(frame, info, countdown, total_detected):
    """
    Draw bounding box on camera feed + side panel with all scores.
    Clean, minimal camera view. All details in the panel.
    """
    fh, fw = frame.shape[:2]

    # ── Side panel ────────────────────────────────────────────
    panel = np.zeros((fh, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)   # dark background

    detected = info["detected"]
    accent   = (0, 220, 0) if detected else (180, 180, 180)

    def put(text, y, colour=(200,200,200), scale=0.48, bold=1):
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, bold)

    def bar(label, value, max_val, y, colour):
        """Draw a small progress bar."""
        put(label, y, (160,160,160), 0.40)
        bar_x, bar_y = 10, y + 5
        bar_w = PANEL_W - 20
        bar_h = 10
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
        fill = int(bar_w * min(value/max(max_val,0.001), 1.0))
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), colour, -1)
        put(f"{value:.2f}", bar_y + 20, colour, 0.38)

    # Title
    cv2.rectangle(panel, (0,0), (PANEL_W, 38), (35,35,35), -1)
    put("AHP  DETECTION", 25, (220,220,220), 0.52, 2)

    # Status
    status_col = (0, 200, 0) if detected else (80, 80, 200)
    status_txt = "DETECTED" if detected else "NOT DETECTED"
    cv2.rectangle(panel, (0,42), (PANEL_W, 72), status_col, -1)
    put(status_txt, 63, (255,255,255), 0.55, 2)

    # Cooldown timer
    put(f"Next check: {countdown:.1f}s", 92, (160,160,160), 0.42)
    cd_bar_w = PANEL_W - 20
    cd_fill  = int(cd_bar_w * (1.0 - countdown / COOLDOWN_SEC))
    cv2.rectangle(panel, (10,96), (10+cd_bar_w, 104), (40,40,40), -1)
    cv2.rectangle(panel, (10,96), (10+cd_fill,  104), (0,160,200), -1)

    # Divider
    cv2.line(panel, (10,112), (PANEL_W-10,112), (50,50,50), 1)

    # ── Scores ────────────────────────────────────────────────
    conf_col = (0,220,0) if info["confidence"] >= CONF_THR else (80,80,200)
    sim_col  = (0,220,0) if info["similarity"]  >= SIM_THR  else (80,80,200)

    # Confidence
    put("Confidence", 132, (160,160,160), 0.42)
    bar("", info["confidence"], 1.0, 135, conf_col)

    # Similarity
    put("Dataset Similarity", 172, (160,160,160), 0.42)
    bar("", info["similarity"], 1.0, 175, sim_col)

    # Texture
    tex_pct = info["texture_pct"]
    tex_col = (0,220,0) if info["texture_raw"] >= TEX_MIN else (80,80,200)
    put(f"Texture  {tex_pct:.0f}%", 215, (160,160,160), 0.42)
    bar("", tex_pct, 100.0, 218, tex_col)

    # Brightness
    bright_col = (0,220,0) if BRIGHT_MIN <= info["brightness"] <= BRIGHT_MAX else (80,80,200)
    put(f"Brightness  {info['brightness']:.0f}", 258, (160,160,160), 0.42)
    bar("", info["brightness"], 255.0, 261, bright_col)

    # Aspect ratio
    put(f"Aspect ratio  {info['aspect']:.2f}", 305, (160,160,160), 0.42)

    # Divider
    cv2.line(panel, (10, 325), (PANEL_W-10, 325), (50,50,50), 1)

    # Output
    out_val  = "1" if detected else "0"
    out_col  = (0,220,0) if detected else (80,80,200)
    put("OUTPUT", 350, (160,160,160), 0.44)
    cv2.putText(panel, out_val, (10, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 1.80, out_col, 3)

    # Total detections
    put(f"Total detected: {total_detected}", 430, (140,140,140), 0.40)

    # Divider
    cv2.line(panel, (10, 445), (PANEL_W-10, 445), (50,50,50), 1)

    # Threshold reference
    put("Thresholds", 462, (100,100,100), 0.38)
    put(f"Conf  ≥ {CONF_THR:.2f}", 478, (80,80,80), 0.36)
    put(f"Sim   ≥ {SIM_THR:.2f}", 492, (80,80,80), 0.36)
    put(f"Tex    {TEX_MIN:.0f}–{TEX_MAX:.0f}", 506, (80,80,80), 0.36)
    put(f"Bright {BRIGHT_MIN:.0f}–{BRIGHT_MAX:.0f}", 520, (80,80,80), 0.36)

    # ── Camera view ───────────────────────────────────────────
    cam = frame.copy()

    # Draw bounding box only if object found
    if "bx" in info:
        bx, by, bw, bh = info["bx"], info["by"], info["bw"], info["bh"]
        colour    = (0, 230, 0)  if detected else (60, 60, 200)
        thickness = 4            if detected else 2

        cv2.rectangle(cam, (bx, by), (bx+bw, by+bh), colour, thickness)

        # Label above box
        label = "SANITARY PAD" if detected else "CHECKING..."
        (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bg = (0,160,0) if detected else (40,40,120)
        cv2.rectangle(cam, (bx, by-th2-12), (bx+tw+8, by), bg, -1)
        cv2.putText(cam, label, (bx+4, by-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # Stitch camera + panel side by side
    combined = np.hstack([cam, panel])
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, db_emb = load_model()
    cam           = CameraThread(CAMERA_ID)
    time.sleep(0.5)  # let camera warm up

    detection_log  = []
    last_output_t  = time.time()
    last_info      = {"detected": False, "confidence": 0.0, "similarity": 0.0,
                      "texture_raw": 0.0, "texture_pct": 0.0,
                      "brightness": 0.0, "aspect": 0.0,
                      "timestamp": time.time()}
    total_detected = 0

    print(f"  {'TIME':>10}   OUTPUT   CONF    SIM    TEX%   BRIGHT")
    print("  " + "─" * 56)

    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        # ── Run analysis every COOLDOWN_SEC seconds ───────────
        now      = time.time()
        elapsed  = now - last_output_t
        countdown = max(0.0, COOLDOWN_SEC - elapsed)

        if elapsed >= COOLDOWN_SEC:
            last_info     = analyse_frame(frame, model, db_emb)
            last_output_t = now
            countdown     = COOLDOWN_SEC

            if last_info["detected"]:
                total_detected += 1
                detection_log.append(last_info)
                with open(LOG_FILE, "w") as f:
                    json.dump(detection_log, f, indent=2)

            # Terminal output
            t_str  = time.strftime("%H:%M:%S")
            output = 1 if last_info["detected"] else 0
            print(f"  [{t_str}]   {output}      "
                  f"{last_info['confidence']:.2f}   "
                  f"{last_info['similarity']:.2f}   "
                  f"{last_info['texture_pct']:>5.1f}   "
                  f"{last_info['brightness']:>6.1f}"
                  + ("   ← SANITARY PAD" if output == 1 else ""))

        # ── Draw and display ──────────────────────────────────
        display = draw_frame(frame, last_info, countdown, total_detected)
        cv2.imshow("AHP Waste Segregation  [Q=quit]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\n  Stopped. {total_detected} pads detected. Log saved → '{LOG_FILE}'\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
