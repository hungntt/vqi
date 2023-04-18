import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, ToTensor
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
import json
from torchvision.transforms import ToTensor

# Define the COCO_CLASSES and COCO_LABEL_MAP
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # Get list of image and label files
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load image and label
        img = Image.open(img_path).convert('RGB')
        with open(label_path) as f:
            label = json.load(f)

        # Convert label to segmentation mask
        mask = torch.zeros(img.size[1], img.size[0], dtype=torch.int64)
        for shape in label['shapes']:
            if shape['label'] in COCO_CLASSES:
                class_id = COCO_LABEL_MAP[COCO_CLASSES.index(shape['label'])]  # Map COCO class name to ID
                points = shape['points']
                # Convert polygon points to binary mask
                polygon = [(p[0], p[1]) for p in points]
                img_draw = ImageDraw.Draw(img)
                img_draw.polygon(polygon, fill=1)
                mask = torch.maximum(mask, torch.tensor(img).permute(1, 0, 2).sum(dim=2).round().int())

        # Apply transformation if specified
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask


def custom_collate(batch):
    # Convert batch of PIL images and labels to batch of tensors
    img_batch, mask_batch = zip(*batch)
    img_batch = torch.stack([ToTensor()(img) for img in img_batch])
    mask_batch = torch.stack(mask_batch)

    return img_batch, mask_batch


# Initialization
num_epochs = 1
batch_size = 4

# Initialize custom dataset and data loader
dataset = CustomDataset(img_dir='data/images', label_dir='data/labels')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Initialize model and optimizer
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=len(COCO_CLASSES))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('Training started...')
for epoch in range(num_epochs):
    progress_bar = tqdm(data_loader)
    for i, (images, masks) in enumerate(data_loader):
        # Forward pass
        outputs = model(images)['out']

        # Compute loss
        loss = torch.nn.functional.cross_entropy(outputs, masks, ignore_index=255)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar = tqdm(data_loader)
        # Print status
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

print('Saving model...')
# Save the model checkpoint as pth file
torch.save(model.state_dict(), 'model.pth')
