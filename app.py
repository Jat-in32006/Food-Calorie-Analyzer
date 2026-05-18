import sys
from pathlib import Path

# Add the custom environment path on E: drive where gradio is installed
sys.path.insert(0, str(Path(r"e:\Image Processing\myenv").resolve()))

import gradio as gr
import torch
import numpy as np
import cv2

# Set ROOT directory so we can import models and utils
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Default GPU device
device = select_device('')

# Use the customized trained model if it exists, otherwise use sanity check YOLO base model
food_model_path = ROOT / "runs" / "train" / "food_model" / "weights" / "best.pt"
fallback_model_path = ROOT / "yolov5s_v6.pt"

if food_model_path.exists():
    weights = str(food_model_path)
    model_status_text = "✨ Custom Food Model Loaded!"
else:
    weights = str(fallback_model_path)
    model_status_text = "⚠️ Custom Train Model not finished yet. Falling back to Generalized YOLO weights."

# Initialize detection model
model = DetectMultiBackend(weights, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt

# Food Calories Mapping from code
food_data = {
    'Name': ['Aloo Gobi', 'Aloo Matar', 'Aloo Methi', 'Aloo Tikki', 'Apple', 'Bhindi Masala', 
             'Biryani', 'Boiled Egg', 'Bread', 'Burger', 'Butter Chicken', 'Chai', 'Chicken Curry',
             'Chicken Tikka', 'Chicken Wings', 'Chole', 'Daal', 'French Fries', 'French Toast', 'Fried Egg', 
             'Kadhi Pakora', 'Kheer', 'Lobia Curry', 'Omelette', 'Onion Pakora', 'Onion Rings', 'Palak Paneer',
             'Pancakes', 'Paratha', 'Rice', 'Roti', 'Samosa', 'Sandwich', 'Spring Rolls', 'Waffles', 'White Rice', 'apple', 'sandwich', 'bowl', 'dining table'],
    'Calories': [150, 170, 160, 200, 95, 120, 320, 70, 80, 250, 350, 45, 220, 180, 280, 200, 160, 220, 200,
                 90, 150, 220, 180, 150, 190, 210, 230, 220, 180, 130, 100, 180, 280, 160, 250, 150, 95, 280, 0, 0]
}

def predict(img):
    if img is None:
        return None, "No image uploaded."
    
    im0 = img.copy()
    
    # 1. Image Preprocessing (Letterbox & color spaces)
    imgsz = check_img_size((640, 640), s=stride)
    img_processed = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    
    # Convert HWC -> CHW, and RGB -> BGR for processing 
    img_processed = img_processed.transpose((2, 0, 1))
    img_processed = np.ascontiguousarray(img_processed)
    
    im = torch.from_numpy(img_processed).to(device)
    im = im.float() / 255.0
    if len(im.shape) == 3:
        im = im[None]
        
    # 2. Run Inference
    pred = model(im, augment=False, visualize=False)
    
    # 3. Non Maximum Suppression
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=100)
    
    total_calories = 0
    detected_items = {}
    
    # 4. Box Drawing & Logic
    annotator = Annotator(im0, line_width=3, example=str(names))
    
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls_id = int(cls)
                name = names[cls_id]
                annotator.box_label(xyxy, f"{name} {conf:.2f}", color=colors(cls_id, True))
                
                # Fetch Calorie Info
                detected_items[name] = detected_items.get(name, 0) + 1
                
    im0 = annotator.result()
    
    # 5. Format Calorie Outputs
    lines = []
    
    for k, v in detected_items.items():
        if k in food_data['Name']:
            idx = food_data['Name'].index(k)
            cals = food_data['Calories'][idx] * v
            total_calories += cals
            if cals > 0:
                lines.append(f"✅ {v}x {k} : {cals} kcal")
        else:
            # Fallback for base YOLO
            lines.append(f"❓ {v}x {k} (Unknown Category)")

    # Construct Results Markdown
    response = f"### 📊 Total Calorie Estimate: {total_calories} kcal\n\n**{model_status_text}**\n\n"
    if len(lines) > 0:
         response += "**Detected Food Elements:**\n\n" + "\n\n".join(lines)
    else:
         response += "\nNo recognized food distinctors found in this image."
         
    return im0, response

# Custom Aesthetic Gradio Blocks
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🥘 AI Deep Learning Food Recognition System")
    gr.Markdown("Upload your dish or snap a photo. The neural network detects individual plates and components to accurately measure total dietary calories.", elem_classes="text-center")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Food Selection", height=400)
            btn = gr.Button("⚡ Analyze Calories")
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Recognition Bounds")
            output_text = gr.Markdown()
            
    btn.click(fn=predict, inputs=input_image, outputs=[output_image, output_text])
    
if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)
