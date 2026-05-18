import cv2
import numpy as np


def segment_and_count_instances(preprocess_results: dict) -> dict:
    """
    Marker-based Watershed segmentation with aggressive merging to prevent
    over-segmentation of single objects (e.g. one apple → many fragments).

    Key fixes vs original:
    - Distance-transform threshold raised from 0.5 → 0.65 so only strong
      peaks become seeds (fewer, more confident markers).
    - Minimum area filter raised from 150 → 2 % of image area so tiny
      texture/watermark fragments are discarded.
    - After watershed, adjacent regions that share >60 % of their smaller
      bounding-box area are merged into one instance.
    """
    gray = preprocess_results["processed_gray"]
    bgr  = preprocess_results["enhanced_bgr"].copy()

    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    # ── 1. Sobel gradient magnitude ──────────────────────────────────────
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad    = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── 2. Otsu threshold on gradient ────────────────────────────────────
    _, thresh = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 3. Morphological closing — larger kernel bridges gaps inside objects
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Sure background via dilation
    sure_bg = cv2.dilate(morphed, kernel, iterations=3)

    # ── 4. Distance transform — raise threshold to get fewer, stronger seeds
    dist = cv2.distanceTransform(morphed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.65 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown / boundary zone
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── 5. Connected components → watershed markers ───────────────────────
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(bgr, markers)

    # ── 6. Extract raw instances (strict area filter) ─────────────────────
    # Minimum area = 2 % of image — removes watermarks, leaves, tiny noise
    min_area = max(500, int(img_area * 0.02))

    raw_instances = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:          # background / boundary
            continue

        mask = np.uint8(markers == marker_id)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h_box = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = int(M["m10"] / (M["m00"] + 1e-5))
        cy = int(M["m01"] / (M["m00"] + 1e-5))

        if (M["mu20"] + M["mu02"]) > 0:
            sq = np.sqrt((M["mu20"] - M["mu02"]) ** 2 + 4 * M["mu11"] ** 2)
            ecc = (M["mu20"] + M["mu02"] - sq) / (M["mu20"] + M["mu02"] + sq)
        else:
            ecc = 0.0

        raw_instances.append({
            "bbox":        [x, y, w, h_box],
            "center":      [cx, cy],
            "area":        area,
            "eccentricity": float(ecc),
        })

    # ── 7. Merge over-segmented fragments ────────────────────────────────
    # Two regions are merged when one's bbox overlaps >60 % of the smaller one.
    merged = _merge_overlapping(raw_instances, iou_thresh=0.60)

    # ── 8. Assign final IDs ───────────────────────────────────────────────
    instance_metadata = []
    for i, inst in enumerate(merged, start=1):
        inst["instance_id"] = i
        instance_metadata.append(inst)

    return {
        "overall_count": len(instance_metadata),
        "instances":     instance_metadata,
    }


# ── Helper: greedy bbox-overlap merge ────────────────────────────────────────
def _merge_overlapping(instances: list, iou_thresh: float = 0.60) -> list:
    """
    Greedily merge any two instances whose bounding boxes overlap by more than
    iou_thresh (measured as intersection / area-of-smaller-box).
    Repeats until no more merges happen.
    """
    if not instances:
        return []

    changed = True
    while changed:
        changed = False
        merged_flags = [False] * len(instances)
        new_list = []

        for i in range(len(instances)):
            if merged_flags[i]:
                continue
            base = instances[i]
            for j in range(i + 1, len(instances)):
                if merged_flags[j]:
                    continue
                other = instances[j]
                if _box_overlap(base["bbox"], other["bbox"]) > iou_thresh:
                    # Merge: take union bbox, sum areas
                    base  = _union_instance(base, other)
                    merged_flags[j] = True
                    changed = True
            new_list.append(base)

        instances = new_list

    return instances


def _box_overlap(a, b):
    """Intersection area / area of the smaller box."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    smaller = min(aw * ah, bw * bh)
    if smaller == 0:
        return 0.0
    return inter / float(smaller)


def _union_instance(a, b):
    """Return a new instance dict that is the union of two instances."""
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    ux = min(ax, bx)
    uy = min(ay, by)
    ux2 = max(ax + aw, bx + bw)
    uy2 = max(ay + ah, by + bh)
    total_area = a["area"] + b["area"]
    cx = int((a["center"][0] * a["area"] + b["center"][0] * b["area"]) / total_area)
    cy = int((a["center"][1] * a["area"] + b["center"][1] * b["area"]) / total_area)
    return {
        "bbox":         [ux, uy, ux2 - ux, uy2 - uy],
        "center":       [cx, cy],
        "area":         total_area,
        "eccentricity": (a["eccentricity"] + b["eccentricity"]) / 2,
    }
