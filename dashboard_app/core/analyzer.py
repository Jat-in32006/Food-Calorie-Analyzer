import cv2
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Only add myenv if it exists AND we actually need it (avoid numpy conflicts)
myenv_path = str((ROOT / "myenv").resolve())
if myenv_path not in sys.path and (ROOT / "myenv").exists():
    # Append (not insert) so system numpy takes priority
    sys.path.append(myenv_path)

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

device = select_device('')

# Priority: best trained food model (needs enough epochs) → fallback COCO model
_CANDIDATE_WEIGHTS = [
    ROOT / "runs" / "train" / "food_demo"        / "weights" / "best.pt",
    ROOT / "runs" / "train" / "food_model_final" / "weights" / "best.pt",
    ROOT / "runs" / "train" / "food_model4"      / "weights" / "best.pt",
    ROOT / "runs" / "train" / "food_model3"      / "weights" / "best.pt",
    ROOT / "runs" / "train" / "food_model2"      / "weights" / "best.pt",
    ROOT / "runs" / "train" / "food_model"       / "weights" / "best.pt",
    ROOT / "yolov5s_v6.pt",
    ROOT / "yolov5s.pt",
]

weights_path = next((str(p) for p in _CANDIDATE_WEIGHTS if p.exists()), None)
if weights_path is None:
    raise FileNotFoundError("No YOLO weights found.")


_food_weights = [str(p) for p in _CANDIDATE_WEIGHTS[:6] if p.exists()]
_coco_weights  = next((str(p) for p in _CANDIDATE_WEIGHTS[6:] if p.exists()), None)

# Minimum mAP@0.5 required before switching to food model
_MIN_MAP_FOR_DEMO = 0.35

def _get_best_map(path: str) -> float:
    """Read the results.csv next to the weights to get best mAP@0.5 achieved."""
    try:
        import csv
        weights_dir = Path(path).parent        # .../weights/
        train_dir   = weights_dir.parent       # .../food_demo/
        csv_path    = train_dir / "results.csv"
        if not csv_path.exists():
            return 0.0
        best = 0.0
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # strip whitespace from keys (results.csv has padded headers)
                clean = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    v = float(clean.get('metrics/mAP_0.5', '0'))
                    if v > best:
                        best = v
                except ValueError:
                    pass
        return best
    except Exception:
        return 0.0

_using_food_model = False
weights_path = _coco_weights   # default

for fw in _food_weights:
    map_score = _get_best_map(fw)
    if map_score >= _MIN_MAP_FOR_DEMO:
        weights_path = fw
        _using_food_model = True
        print(f"[NutriScan] Food model selected (mAP={map_score:.3f}): {fw}")
        break

if not _using_food_model:
    best_map = _get_best_map(_food_weights[0]) if _food_weights else 0.0
    print(f"[NutriScan] Food model not ready yet (mAP={best_map:.3f} < {_MIN_MAP_FOR_DEMO}). Using COCO fallback.")

print(f"[NutriScan] Loading model: {weights_path}")
print(f"[NutriScan] Food model: {_using_food_model}")
model = DetectMultiBackend(weights_path, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
print(f"[NutriScan] Model ready — {len(names)} classes")

# Calorie database — matches both food-model class names and COCO fallback names
_CALORIE_MAP = {
    'Aloo Gobi':     150, 'Aloo Matar':     170, 'Aloo Methi':    160,
    'Aloo Tikki':    200, 'Apple':           95, 'Bhindi Masala': 120,
    'Biryani':       320, 'Boiled Egg':      70, 'Bread':          80,
    'Burger':        250, 'Butter Chicken': 350, 'Chai':           45,
    'Chicken Curry': 220, 'Chicken Tikka':  180, 'Chicken Wings': 280,
    'Chole':         200, 'Daal':           160, 'French Fries':  220,
    'French Toast':  200, 'Fried Egg':       90, 'Kadhi Pakora':  150,
    'Kheer':         220, 'Lobia Curry':    180, 'Omelette':      150,
    'Onion Pakora':  190, 'Onion Rings':    210, 'Palak Paneer':  230,
    'Pancakes':      220, 'Paratha':        180, 'Rice':          130,
    'Roti':          100, 'Samosa':         180, 'Sandwich':      280,
    'Spring Rolls':  160, 'Waffles':        250, 'White Rice':    150,
    # COCO fallback names (lowercase)
    'apple':          95, 'sandwich':       280, 'orange':         62,
    'banana':         89, 'broccoli':        55, 'carrot':         41,
    'hot dog':       290, 'pizza':          285, 'donut':         452,
    'cake':          350,
}

# Non-food COCO classes — skip silently
_SKIP_CLASSES = {
    'dining table', 'bowl', 'cup', 'fork', 'knife', 'spoon',
    'bottle', 'wine glass', 'chair', 'person', 'cell phone',
}

# COCO class → better food label remapping
# When the COCO model fires these classes on a food image, remap to correct food name
_COCO_REMAP = {
    'cake':         'Donut',
    'donut':        'Donut',
    'pizza':        'Pizza',
    'hot dog':      'Hot Dog',
    'sandwich':     'Sandwich',
    'apple':        'Apple',
    'banana':       'Banana',
    'orange':       'Orange',
    'broccoli':     'Broccoli',
    'carrot':       'Carrot',
}

# Extended calorie map covering remapped names
_CALORIE_MAP.update({
    'Cake/Dessert': 350,
    'Pizza':        285,
    'Donut':        452,
    'Hot Dog':      290,
    'Banana':        89,
    'Orange':        62,
    'Broccoli':      55,
    'Carrot':        41,
})


def analyze_food(image_bgr: np.ndarray, watershed_instances=None):
    """
    Pure YOLO detection — mirrors the original Gradio app.py exactly.
    image_bgr: BGR numpy array from cv2.imdecode (what the API provides).
    watershed_instances: ignored, kept for API compatibility.
    """
    im0 = image_bgr.copy()   # BGR, HWC — same as app.py's im0

    # ── Preprocessing — identical to app.py ──────────────────────────────
    imgsz = check_img_size((640, 640), s=stride)
    img_processed = letterbox(im0, imgsz, stride=stride, auto=pt)[0]

    # HWC → CHW  (letterbox returns BGR HWC, same as app.py)
    img_processed = img_processed.transpose((2, 0, 1))
    img_processed = np.ascontiguousarray(img_processed)

    im = torch.tensor(img_processed, dtype=torch.float32).to(device)
    im = im / 255.0
    if len(im.shape) == 3:
        im = im[None]

    # ── Inference ────────────────────────────────────────────────────────
    pred = model(im, augment=False, visualize=False)

    # ── NMS ──────────────────────────────────────────────────────────────
    # agnostic=False: NMS runs per-class so multiple donuts/apples survive
    # agnostic=True only used when COCO model fires cake+donut on same box
    pred = non_max_suppression(pred, 0.25, 0.45, None, not _using_food_model, max_det=100)

    # ── Parse detections ─────────────────────────────────────────────────
    detected_objects = []
    total_cals       = 0
    instance_id      = 1

    for det in pred:
        if not len(det):
            continue
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            name = names[int(cls)]
            if name in _SKIP_CLASSES:
                continue

            # Remap COCO generic labels to proper food names when using COCO model
            if not _using_food_model and name in _COCO_REMAP:
                name = _COCO_REMAP[name]

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            w, h = x2 - x1, y2 - y1
            cals  = _CALORIE_MAP.get(name, 0)
            total_cals += cals

            detected_objects.append({
                "id":         instance_id,
                "class":      name,
                "confidence": round(float(conf), 4),
                "calories":   cals,
                "bbox":       [x1, y1, w, h],
                "area":       w * h,
            })
            instance_id += 1

    return detected_objects, total_cals
