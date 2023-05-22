import json
import os
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101

# Define the COCO_CLASSES and COCO_LABEL_MAP
COCO_CLASSES = ('__background__', 'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden')

COCO_LABEL_MAP = {cls: idx for (idx, cls) in enumerate(COCO_CLASSES)}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = os.listdir(self.img_dir)
        self.label_files = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load image
        orig_image = Image.open(img_path).convert('RGB')
        image_size = orig_image.size
        # Load label and extract segmentation mask
        with open(label_path, 'r') as f:
            data = json.load(f)

        mask = Image.new('L', orig_image.size, 0)
        draw = ImageDraw.Draw(mask)
        for shape in data['shapes']:
            if shape['label'] in COCO_CLASSES:
                class_id = COCO_LABEL_MAP[shape['label']]
                points = [(p[0], p[1]) for p in shape['points']]
                draw.polygon(points, fill=class_id)
        mask = np.array(mask)

        # Apply transforms to image and mask
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(orig_image)
        mask = torch.tensor(mask, dtype=torch.int64)
        # Display the original image and the segmentation mask for verification
        # visualize_mask(mask)
        return img, mask


def visualize_mask(mask):
    fig, ax = plt.subplots(1)
    ax.imshow(mask, cmap='gray')
    ax.set_title('Segmentation Mask')
    plt.show()


def custom_collate(batch):
    images, masks = [], []
    for sample in batch:
        img, mask = sample
        if isinstance(img, torch.Tensor):
            images.append(img)
        else:
            img = transforms.ToTensor()(img)
            images.append(img)
        masks.append(mask)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks


def dice_loss(pred, target):
    smooth = 1.
    num_classes = pred.size(1)
    loss = 0.
    for c in range(num_classes):
        p = pred[:, c]
        t = target == c
        intersection = (p * t).sum()
        loss += (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return 1. - loss / num_classes


def jaccard_index_loss(logits, targets):
    # apply softmax to logits along the channel dimension
    probs = F.softmax(logits, dim=1)
    # get the predicted class masks from the probabilities
    pred_masks = probs.argmax(dim=1)
    # compute intersection and union between predicted and ground truth masks
    intersection = (pred_masks & targets).float().sum(dim=(1, 2))
    union = (pred_masks | targets).float().sum(dim=(1, 2))
    # compute Jaccard index and return the loss
    jaccard = intersection / union
    return (1 - jaccard).mean()


# Initialization
num_epochs = 30
batch_size = 4

# Initialize custom dataset and data loader
dataset = CustomDataset(img_dir='data/images', label_dir='data/labels')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Initialize model and optimizer
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
# Move model and data to GPU
model = deeplabv3_resnet101(pretrained=False, num_classes=len(COCO_CLASSES))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print('Training started...')
for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(data_loader):
        # Move data to GPU
        images = images.to(device)
        masks = masks.to(device)
        # Forward pass
        outputs = model(images)['out']
        # Compute loss
        loss = torch.nn.functional.cross_entropy(outputs, masks)
        # loss = dice_loss(F.softmax(outputs, dim=1), masks)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print status
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

print('Saving model...')
# Save the model checkpoint as pth file
torch.save(model.state_dict(), 'model.pth')
