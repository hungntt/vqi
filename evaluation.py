import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange, DropInConfidence, IncreaseInConfidence
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image, deprocess_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget, ClassifierOutputSoftmaxTarget
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM, EigenCAM, ScoreCAM, \
    XGradCAM, GradCAMElementWise, FullGrad, HiResCAM

from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper

xai = "GradCAM"
COCO_CLASSES = ['__background__', 'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden']

COCO_LABEL_MAP = {cls: idx for (idx, cls) in enumerate(COCO_CLASSES)}
PATH = 'model/segmentation/model_ResNet50.pth'


class SemanticSegmentationSoftmaxTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        output_masked = torch.softmax(model_output, dim=-1)[self.category, :, :] * self.mask
        # Remove 0 from the output_masked
        output_masked = output_masked[output_masked != 0]
        return output_masked.mean()


# Showing the metrics on top of the CAM :
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "Remove and Debias", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return visualization


def benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, use_cuda=False):
    methods = [
        ("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("EigenCAM", EigenCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("ScoreCAM", ScoreCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("XGradCAM", XGradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("GradCAMElementWise", GradCAMElementWise(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("FullGrad", FullGrad(model=model, target_layers=target_layers, use_cuda=use_cuda)),
        ("HiResCAM", HiResCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)),
    ]
    # cam_mult_metric = CamMultImageConfidenceChange()
    drop_metric = DropInConfidence()
    increase_metric = IncreaseInConfidence()
    road_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    targets = [SemanticSegmentationTarget(category_idx, mask_float)]
    metric_targets = [SemanticSegmentationSoftmaxTarget(category_idx, mask_float)]

    # visualizations = []
    # percentiles = [10, 50, 90]
    for name, cam_method in methods:
        with cam_method:
            attributions = cam_method(input_tensor=input_tensor,
                                      targets=targets,
                                      use_cuda=torch.cuda.is_available())
        # attribution = attributions[0, :]
        print('Start calculating metrics')
        road_scores = road_metric(input_tensor, attributions, metric_targets, model)
        road_score = road_scores[0]

        # cam_mult_scores = cam_mult_metric(input_tensor, attributions, metric_targets, model)
        # cam_mult_score = cam_mult_scores[0]

        drop_scores = drop_metric(input_tensor, attributions, metric_targets, model)
        drop_score = drop_scores[0]

        increase_scores = increase_metric(input_tensor, attributions, metric_targets, model)
        increase_score = increase_scores[0]

        print(f"{name} - Road: {road_score:.5f} - "
              f"Drop: {drop_score:.5f} - Increase: {increase_score:.5f}")

        # visualization = show_cam_on_image(rgb_img, attribution, use_rgb=True)
        # visualization = visualize_score(visualization, score, name, percentiles)
        # visualizations.append(visualization)
    # return road_score, cam_mult_score, drop_score, increase_score


if __name__ == '__main__':
    np.random.seed(42)
    category = 'tower_wooden'

    model = deeplabv3_resnet50(pretrained=False, num_classes=len(COCO_CLASSES))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(PATH))
    else:
        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    model.eval()

    img = Image.open('data/images/3_00092.jpg')
    img = np.array(img)
    rgb_img = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model.cuda()

    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    category_idx = COCO_LABEL_MAP[category]
    mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    mask_uint8 = 255 * np.uint8(mask == category_idx)
    mask_float = np.float32(mask == category_idx)
    target_layers = [model.model.backbone.layer4]

    benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)
    # image.save(f"{xai}_{category}.png")

    # cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    # target_layers = [model.model.backbone.layer4]
    # targets = [SemanticSegmentationTarget(category_idx, mask_float)]
    #
    # with globals()[xai](model=model,
    #                     target_layers=target_layers,
    #                     use_cuda=torch.cuda.is_available()) as cam:
    #     grayscale_cams = cam(input_tensor=input_tensor, targets=targets),
    #     cam_image = show_cam_on_image(rgb_img, grayscale_cams[0, :], use_rgb=True)

    # cam = np.uint8(255 * grayscale_cams[0, :])
    # cam = cv2.merge([cam, cam, cam])
    # images = np.hstack((np.uint8(255 * img), cam, cam_image))
    # Image.fromarray(images)
    #
    # cam_metric = CamMultImageConfidenceChange()
    # scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
    # score = scores[0]
    # visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
    # visualization = deprocess_image(visualization)
    # print(f"The confidence increase percent: {100 * score}")
    # print("The visualization of the perturbed image for the metric:")
    # Image.fromarray(visualization)
    # print("Drop in confidence", DropInConfidence()(input_tensor, grayscale_cams, targets, model))
    # print("Increase in confidence", IncreaseInConfidence()(input_tensor, grayscale_cams, targets, model))
