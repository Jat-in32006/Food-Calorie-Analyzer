<a href="https://colab.research.google.com/github/Yogeshpvt/Deep-Learning-Based-Food-Recognition-and-Calorie-Estimation-for-Indian-Food-Images/blob/main/Model_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab_Notebooks/Calorie
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow
%pip install comet_ml

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
torch.cuda.is_available()
%pip install tim
%pip install timm
from roboflow import Roboflow
rf = Roboflow(api_key="4TFLqpycRN0FG5gHvY2z")
project = rf.workspace("object-detection-vpvcm").project("nutracal-food-detection")
dataset = project.version(5).download("yolov5")
os.environ["DATASET_DIRECTORY"] = '/content/NutraCal-Food-Detection-5'
#20
#!python train.py --img 640 --batch 16 --epochs 4 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
#!python train.py --img 640 --batch 16 --epochs 4 --data {dataset.location}/data.yaml --weights runs/train/exp23/weights/best.pt --cache
!python train.py --img 640 --batch 16 --epochs 4 --data {dataset.location}/data.yaml --weights runs/train/exp29/weights/best.pt --cache
!python detect.py --weights runs/train/exp29/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp29/weights/best.pt --task val
#50 epochs
!python train.py --img 640 --batch 16 --epochs 5 --data {dataset.location}/data.yaml --weights runs/train/exp29/weights/best.pt --cache
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp27/weights/best.pt --task val
!pip install --upgrade pip
!pip install --upgrade tensorflow ml-dtypes opencv-python-headless
%pip install unet
import torch
import torchvision.models as models

# Create a VGG19 model
model = models.vgg19(pretrained=False)  # We set pretrained to False since you are loading your own weights

# Load the pre-trained weights into the model
state_dict = torch.load("/content/drive/MyDrive/Colab_Notebooks/Calorie/yolov5/weights/vgg19-dcbb9e9d.pth")
model.load_state_dict(state_dict)

# Save the model as a script module
example = torch.rand(1, 3, 224, 224)  # Adjust the input size as needed
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/content/drive/MyDrive/Colab_Notebooks/Calorie/yolov5/weights/vgg19-dcbb9e9d.pt")

#!python train.py --img 640 --batch 16 --freeze 10 --epochs 50 --data {dataset.location}/data.yaml --cfg VGG.yaml --weights runs/train/exp26/weights/vgg19-dcbb9e9d.pt --cache
!python train.py --img 640 --batch 16  --epochs 50 --data {dataset.location}/data.yaml --weights runs/train/exp25/weights/yolo5s.pt --cache
#After weights are transfered
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp33/weights/best.pt --task val
!python detect.py --weights runs/train/exp33/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
!python detect3.py --weights runs/train/exp33/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images/
!python train.py --img 640 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --weights runs/train/exp33/weights/best.pt  --cache
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp41/weights/best.pt --task val

 # Hyperparameters for the existing low-augmentation training of YOLO v5

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
def fitness(x):
     # Model fitness as a weighted combination of metrics
     w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
     return (x[:, :4] * w).sum(1)
!python train.py --img 640 --batch 16 --epochs 50 --data {dataset.location}/data.yaml --weights runs/train/exp33/weights/best.pt  --cache --evolve
!python train.py --img 640 --batch 16 --epochs 20 --data {dataset.location}/data.yaml --weights runs/train/exp33/weights/best.pt  --cache --evolve
#Training after fine tuning the model

lr0: 0.01165          # Initial learning rate
lrf: 0.01             # Final learning rate
momentum: 0.97078     # Momentum for the optimizer
weight_decay: 0.00049 # L2 regularization strength
warmup_epochs: 3.0532 # Number of epochs for learning rate warm-up
warmup_momentum: 0.8  # Momentum during warm-up
warmup_bias_lr: 0.10211 # Bias learning rate during warm-up

box: 0.06189           # Weight for bounding box loss
cls: 0.53518           # Weight for class loss
cls_pw: 0.76182        # Power for class loss (class loss is raised to this power)
obj: 0.81225           # Weight for objectness loss
obj_pw: 0.9747         # Power for objectness loss (objectness loss is raised to this power)
iou_t: 0.2             # Threshold for intersection over union (IOU) for object detection
anchor_t: 4.0954       # Anchor threshold

fl_gamma: 0.0          # Focal loss gamma (used in classification loss)
hsv_h: 0.01161         # Hue augmentation
hsv_s: 0.606           # Saturation augmentation
hsv_v: 0.4             # Value (brightness) augmentation
degrees: 0.0           # Rotation augmentation
translate: 0.06866     # Translation augmentation
scale: 0.40252         # Scaling augmentation
shear: 0.0             # Shear augmentation
perspective: 0.0       # Perspective augmentation
flipud: 0.0            # Probability of flipping images vertically
fliplr: 0.5            # Probability of flipping images horizontally
mosaic: 1.0            # Probability of applying mosaic augmentation
mixup: 0.0             # Probability of applying mixup augmentation
copy_paste: 0.0        # Probability of applying copy-paste augmentation

anchors: 4.5694        # Anchor value (used for anchor boxes in object detection)
!python train.py --img 640 --batch 16 --hyp hyp_evolve.yaml --epochs 75 --data {dataset.location}/data.yaml --weights runs/train/exp33/weights/best.pt  --cache
!python train.py --img 640 --batch 16 --hyp hyp_evolve.yaml --epochs 30 --data {dataset.location}/data.yaml --weights runs/train/exp37/weights/best.pt  --cache
#Result of hyperparameter tuning with the transfer weights model for 100 epochs
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp38/weights/best.pt --task val
!python train.py --img 640 --batch 16 --hyp hyp_evolve.yaml --epochs 100 --data {dataset.location}/data.yaml --weights runs/train/exp38/weights/best.pt  --cache
!python train.py --img 640 --batch 16 --hyp hyp_evolve.yaml --epochs 50 --data {dataset.location}/data.yaml --weights runs/train/exp52/weights/best.pt  --cache
#fine tuning plus 230 epochs
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp53/weights/best.pt --task val
#fine tuned and weights transfered
!python detect.py --weights runs/train/exp38/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images/
#weights transfered and trained for additional 100 epochs but not fine tuned
!python detect3.py --weights runs/train/exp33/weights/best.pt --img 416 --conf 0.1 --source data/images/
#weights transfered but not fine tuned
#2 - burger and fries & #11 - samosa & #14 - aloo gobi
!python detect2.py --weights runs/train/exp33/weights/best.pt --img 416 --conf 0.1 --source data/images/26.jpg
!python detect2.py --weights runs/train/exp33/weights/best.pt --img 416 --conf 0.1 --source data/images/22.jpg
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp38/weights/best.pt --task val
!python detect2.py --weights runs/train/exp33/weights/best.pt --img 416 --conf 0.1 --source data/images/12.jpg
!python val.py --data {dataset.location}/data.yaml --weights runs/train/exp38/weights/best.pt runs/train/exp33/weights/best.pt --half
!python export.py --weights runs/train/exp33/weights/best.pt --include tflite --img 416
!python detect2.py --data {dataset.location}/data.yaml --weights runs/train/exp38/weights/best.pt runs/train/exp33/weights/best.pt --half
!pip3 install -q tf-models-nightly
!pip3 install -q opencv-python-headless==4.1.2.30
!pip3 install Cmake
!pip3 install keras==2.1.6
!pip3 install tensorflow==1.15.0
!pip3 install h5py==2.10.0
%pip install supervision
import supervision as sv
detections = sv.Detections.from_ultralytics()
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")