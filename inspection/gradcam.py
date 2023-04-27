import numpy as np
import requests
from PIL import Image
import torch
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
        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.sem_classes)}

        self.model = deeplabv3_resnet50(pretrained=True)
        # PATH = 'model/segmentation/model.pth'
        # self.model.load_state_dict(torch.load(PATH))
        self.model = self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = SegmentationModelOutputWrapper(self.model)

    def process_image(self, image_path, category="car", is_url=False, xai="GradCAM"):
        if is_url:
            image = np.array(Image.open(requests.get(image_path, stream=True).raw))
        else:
            image = np.array(Image.open(image_path))

        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        output = self.model(input_tensor)
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

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
        overlayed_image = cv2.addWeighted(image, 0.5, mask_color, 0.5, 0)

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
            grad_cam_image = Image.fromarray(np.uint8(cam_image * 255))

        return pil_overlayed_image, grad_cam_image
