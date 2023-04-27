import numpy as np
import requests
from PIL import Image
import torch
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet50

from model.segmentation.resnet50 import ResNet50
from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper
from model.segmentation.semantic_segmentation_target import SemanticSegmentationTarget


class GradCamSegmentation:
    def __init__(self):
        self.sem_classes = [
            'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden'
        ]
        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(self.sem_classes)}

        self.model = deeplabv3_resnet50(pretrained=False, num_classes=4)
        PATH = 'model/segmentation/model.pth'
        self.model.load_state_dict(torch.load(PATH))
        self.model = self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = SegmentationModelOutputWrapper(self.model)

    def process_image(self, image_path, category="cable", is_url=False, xai="GradCAM"):
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
        segmentation_image = Image.fromarray(np.uint8(mask_uint8))

        target_layers = [self.model.model.backbone.layer4]
        targets = [SemanticSegmentationTarget(category_idx, mask_float)]

        with globals()[xai](model=self.model, target_layers=target_layers,
                            use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            grad_cam_image = Image.fromarray(np.uint8(cam_image * 255))

        return segmentation_image, grad_cam_image


if __name__ == '__main__':
    grad_cam = GradCamSegmentation().process_image(image_path='data/images/1_00186.jpg', is_url=False, xai="EigenCAM")
