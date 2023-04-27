import json

import numpy as np
import requests
from PIL import Image, ImageDraw
import torch
from matplotlib import patches, pyplot as plt
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2

from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper
from model.segmentation.semantic_segmentation_target import SemanticSegmentationTarget


class GradCamSegmentation:
    def __init__(self):
        self.sem_classes = [
            'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden'
        ]

        self.label_map = {0: 1, 1: 2, 2: 3, 3: 4}

        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.sem_classes)}

        self.model = deeplabv3_resnet50(pretrained=False, num_classes=len(self.sem_classes))
        PATH = 'model/segmentation/model.pth'
        self.model.load_state_dict(torch.load(PATH))
        self.model = self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = SegmentationModelOutputWrapper(self.model)

    def process_image(self, image_path, label_path, category="cable", is_url=False, xai="GradCAM"):
        orig_image = Image.open(image_path)
        image = np.array(orig_image)

        rgb_img = np.float32(image) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        output = self.model(input_tensor)
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

        with open(label_path, 'r') as f:
            data = json.load(f)

        mask = Image.new('RGBA', orig_image.size, (0, 0, 0, 0))

        draw = ImageDraw.Draw(mask)
        for shape in data['shapes']:
            points = [(p[0], p[1]) for p in shape['points']]
            draw.polygon(points, fill=tuple(shape['fill_color']), outline=tuple(shape['line_color']))
            # Add the label name to the annotation
            draw.text(points[0], shape['label'], fill=tuple(shape['fill_color']), align='center')

        # Combine the image and the segmentation mask
        result = Image.alpha_composite(orig_image.convert('RGBA'), mask)

        # Convert the result to a PIL image object
        pil_seg_image = result.convert('RGB')

        category_idx = self.sem_class_to_idx[category]
        mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        mask_uint8 = 255 * np.uint8(mask == category_idx)
        mask_float = np.float32(mask == category_idx)

        # Resize mask to the same size as input image
        mask_uint8 = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert mask to color image
        mask_color = np.zeros_like(image)
        mask_color[mask_uint8 > 0] = (0, 0, 255)  # set the color of the segmentation mask to red (BGR format)
        # Overlay segmentation mask on input image
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.addWeighted(bgr_image, 0.5, mask_color, 0.5, 0)

        # Convert the overlayed image back to RGB for PIL
        rgb_overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

        # Convert the overlayed image to a PIL image object
        pil_overlayed_image = Image.fromarray(rgb_overlayed_image)

        target_layers = [self.model.model.backbone.layer4]
        targets = [SemanticSegmentationTarget(category_idx, mask_float)]

        with globals()[xai](model=self.model, target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return pil_overlayed_image, Image.fromarray(cam_image), pil_seg_image
