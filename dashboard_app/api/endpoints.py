from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from dashboard_app.core.analyzer import analyze_food

router = APIRouter()


@router.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data"})

    detected_objects, total_cals = analyze_food(image_bgr)

    return {
        "status":                   "success",
        "overall_count":            len(detected_objects),
        "estimated_calories_total": total_cals,
        "detected_objects":         detected_objects,
    }
