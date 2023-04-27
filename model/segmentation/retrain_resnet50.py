import torch
import torchvision
from torchvision.models.detection import transform
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, ToTensor
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import numpy as np
import os
import json
from torchvision.transforms import ToTensor

# Define the COCO_CLASSES and COCO_LABEL_MAP
COCO_CLASSES = ('cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden')

COCO_LABEL_MAP = {0: 1, 1: 2, 2: 3, 3: 4}


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
                object_mask = torch.tensor(np.array(img)).permute(1, 0, 2).sum(dim=2).round().int()
                object_mask[object_mask != class_id] = 0
                object_mask[object_mask == class_id] = 1
                mask = torch.maximum(mask, object_mask)

        # Apply transformation if specified
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask


def custom_collate(batch):
    # Convert batch of PIL images and labels to batch of tensors
    img_batch, mask_batch = zip(*batch)
    img_batch = torch.stack([ToTensor()(img).to(device) for img in img_batch])
    mask_batch = torch.stack([torch.tensor(np.array(mask)).long().to(device) for mask in mask_batch])

    return img_batch, mask_batch


# Initialization
num_epochs = 1
batch_size = 4

# Initialize custom dataset and data loader
dataset = CustomDataset(img_dir='data/images', label_dir='data/labels')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Initialize model and optimizer
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
# Move model and data to GPU
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=len(COCO_CLASSES))
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('Training started...')
for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(data_loader):
        # Forward pass
        outputs = model(images)['out']

        # Compute loss
        loss = torch.nn.functional.cross_entropy(outputs, masks, ignore_index=255)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print status
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

print('Saving model...')
# Save the model checkpoint as pth file
torch.save(model.state_dict(), 'model.pth')
