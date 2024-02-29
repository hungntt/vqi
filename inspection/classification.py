import torch
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision.models.segmentation import deeplabv3_resnet50
from pytorch_grad_cam import *
import numpy as np
from PIL import Image, ImageDraw
import torch


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class ClassificationExplainer:
    def __init__(self):
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model = self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def explain(self, image_path, label_path, xai):
        orig_image = Image.open(image_path)
        image = np.array(orig_image)

        rgb_img = np.float32(image) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        output = self.model(input_tensor)
        target_layers = [self.model.model.backbone.layer4[-1]]

        with globals()[xai]()(model=self.model,
                              target_layers=target_layers,
                              use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        explanation = Image.fromarray(cam_image)

        return explanation
