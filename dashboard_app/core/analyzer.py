import cv2
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

device = select_device('')

# Always use the COCO model for demo — it reliably detects common foods
weights_path = str(ROOT / "yolov5s_v6.pt")
print(f"[NutriScan] Loading: {weights_path}")
model  = DetectMultiBackend(weights_path, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
print(f"[NutriScan] Ready — {len(names)} classes")

# ── Calorie map ───────────────────────────────────────────────────────────────
_CALORIE_MAP = {
    # COCO food classes (what yolov5s_v6 detects)
    'banana':    89,  'apple':    95,  'sandwich': 280,
    'orange':    62,  'broccoli': 55,  'carrot':    41,
    'hot dog':  290,  'pizza':   285,  'donut':    452,
    'cake':     350,
}

# COCO classes to silently skip (non-food)
_SKIP = {
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
    'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush',
}

# Friendly display names
_DISPLAY = {
    'banana':  'Banana',   'apple':    'Apple',
    'sandwich':'Sandwich', 'orange':   'Orange',
    'broccoli':'Broccoli', 'carrot':   'Carrot',
    'hot dog': 'Hot Dog',  'pizza':    'Pizza',
    'donut':   'Donut',    'cake':     'Cake',
}


def analyze_food(image_bgr: np.ndarray, watershed_instances=None):
    im0   = image_bgr.copy()
    imgsz = check_img_size((640, 640), s=stride)

    proc = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    proc = proc.transpose((2, 0, 1))
    proc = np.ascontiguousarray(proc)

    # Use torch.tensor to avoid torch.from_numpy NumPy ABI issue
    im = torch.tensor(proc, dtype=torch.float32).to(device) / 255.0
    if im.ndim == 3:
        im = im[None]

    pred = model(im, augment=False, visualize=False)
    # agnostic=False → per-class NMS so 3 oranges stay as 3 oranges
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=100)

    detected_objects = []
    total_cals       = 0
    instance_id      = 1

    for det in pred:
        if not len(det):
            continue
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            raw_name = names[int(cls)]
            if raw_name in _SKIP:
                continue

            name = _DISPLAY.get(raw_name, raw_name.title())
            cals = _CALORIE_MAP.get(raw_name, 0)
            total_cals += cals

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            w, h = x2 - x1, y2 - y1

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
