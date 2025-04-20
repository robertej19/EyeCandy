import os
import glob
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

# --- Step 1: Get earliest frame ---
frame_dir = 'data/frames/'
jpg_paths = glob.glob(os.path.join(frame_dir, '*.jpg'))
if not jpg_paths:
    raise ValueError("No JPG images found in data/frames/")

earliest_path = min(jpg_paths, key=lambda p: int(os.path.basename(p).split('.')[0]))

# --- Step 2: Load image and convert to tensor ---
img_bgr = cv2.imread(earliest_path)
if img_bgr is None:
    raise IOError(f"Failed to load image: {earliest_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
img_tensor = F.to_tensor(img_pil).unsqueeze(0)  # Add batch dimension

# --- Step 3: Load model ---
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
img_tensor = img_tensor.to(device)

# --- Step 4: Run inference ---
with torch.no_grad():
    outputs = model(img_tensor)[0]

# --- Step 5: Filter and draw vehicle classes ---
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


vehicle_classes = {'car', 'motorcycle', 'bus', 'truck'}
for box, label_idx, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
    if label_idx.item() >= len(COCO_INSTANCE_CATEGORY_NAMES):
        continue  # Skip invalid label indices

    class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx.item()]
    print(class_name)
    if class_name in vehicle_classes and score.item() > 0.5:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {score:.2f}"
        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

#convert im bgr to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# --- Step 6: Save output ---
output_path = 'detected_frame_frcnn.jpg'

cv2.imwrite(output_path, img_rgb)
print(f"[✓] Saved detection result to {output_path}")
