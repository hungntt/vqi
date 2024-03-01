import urllib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import transforms
from tqdm import tqdm

from model.segmentation.segmentation_output_wrapper import SegmentationModelOutputWrapper

MOBILE = True
if torch.cuda.is_available() and not MOBILE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


sem_classes = ['__background__', 'cable', 'tower_lattice', 'tower_tucohy', 'tower_wooden']
model = torch.hub.load('pytorch/vision:v0.11.0', 'deeplabv3_resnet50', pretrained=False,
                       num_classes=len(sem_classes))

PATH = f'model/segmentation/model_ResNet50.pth'

if torch.cuda.is_available() and not MOBILE:
    model.load_state_dict(torch.load(PATH))
else:
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

if MOBILE:
    model = optimize_for_mobile(torch.jit.script(model)).eval()

model = SegmentationModelOutputWrapper(model)

filename = 'data/images/3_00092.jpg'

input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available() and not MOBILE:
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)[0]
output_predictions = output.argmax(0)


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def generate_masks(n_masks, input_size, p1=0.1, initial_mask_size=(7, 7), binary=True):
    # cell size in the upsampled mask
    Ch = np.ceil(input_size[0] / initial_mask_size[0])
    Cw = np.ceil(input_size[1] / initial_mask_size[1])

    resize_h = int((initial_mask_size[0] + 1) * Ch)
    resize_w = int((initial_mask_size[1] + 1) * Cw)

    masks = []

    for _ in range(n_masks):
        # generate binary mask
        binary_mask = torch.randn(
            1, 1, initial_mask_size[0], initial_mask_size[1])
        binary_mask = (binary_mask < p1).float()

        # upsampling mask
        if binary:
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='nearest')
        else:
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

        # random cropping
        i = np.random.randint(0, Ch)
        j = np.random.randint(0, Cw)
        mask = mask[:, :, i:i + input_size[0], j:j + input_size[1]]

        masks.append(mask)

    masks = torch.cat(masks, dim=0)  # (N_masks, 1, H, W)

    return masks


def rise_segmentation(masks, image, model, preprocess_transform, target=None, n_masks=None, box=None, DEVICE='cpu',
                      vis=True, vis_skip=1):
    if preprocess_transform is None:
        input_tensor = image.clone()
        image = image.permute(1, 2, 0).numpy()
    else:
        input_tensor = preprocess_transform(image)

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    coef = []

    if n_masks is None:
        n_masks = len(masks)

    output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
    output_1 = output.argmax(axis=0)
    output_a = output_1[y_start:y_end, x_start:x_end]

    if target is None:
        target = output_a.max().item()

    for index, mask in tqdm(enumerate(masks[:n_masks])):
        # input_tensor = preprocess_transform(image)
        # output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
        # output_1 = output.argmax(axis = 0)

        input_tensor_1 = input_tensor * mask
        output = model(input_tensor_1.unsqueeze(0).to(DEVICE))[0].detach().cpu()
        output_2 = output.argmax(axis=0)

        # output_a = output_1[y_start:y_end, x_start:x_end]
        #         if target is None:
        #             target = output_a.max().item()
        output_b = output_2[y_start:y_end, x_start:x_end]

        DICE = dice(output_a == target, output_b == target)
        coef.append(DICE)

        if vis:
            if index % vis_skip == 0:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(output_1)
                plt.subplot(1, 3, 2)
                plt.imshow(output_2)
                plt.subplot(1, 3, 3)
                plt.imshow(mask[0])
                plt.show()
    return coef


def rise_aggregated(image, masks, coef, fig_name=None, vis=True):
    aggregated_mask = np.zeros(masks[0][0].shape)

    for i, j in zip(masks[:len(coef)], coef):
        aggregated_mask += i[0].detach().cpu().numpy() * j.item()

    max_, min_ = aggregated_mask.max(), aggregated_mask.min()
    aggregated_mask = np.uint8(255 * (aggregated_mask - min_) / (max_ - min_))
    overlaid = show_cam_on_image(image / 255, aggregated_mask / 255, use_rgb=True)

    title = 'RISE'
    title += ' (Mobile)' if MOBILE else ''

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(aggregated_mask)
    plt.title(title)
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid)
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    if vis is True:
        plt.show()
    plt.close()

    return aggregated_mask, overlaid


def vis_predict(image, model, preprocess_transform, DEVICE='cpu', mask=None, box=None, fig_name=None, vis=True):
    if preprocess_transform is None:
        input_tensor = image.clone()
        image = image.permute(1, 2, 0).numpy()
    else:
        input_tensor = preprocess_transform(image)

    if mask is not None:
        input_tensor = input_tensor * mask
    output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
    output = output.argmax(axis=0)

    if box is None:
        box = (0, 0, 0, 0)
    rect_image = image.copy()
    rect_image = cv2.rectangle(rect_image, (box[2], box[0]), (box[3], box[1]), 255, 10)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title('To Explain')
    plt.subplot(1, 3, 3)
    plt.imshow(rect_image)
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches='tight')
    if vis is True:
        plt.show()
    plt.close()
    return image, output, rect_image


def utility_dilation(image, model, preprocess_transform, target=None, box=None, DEVICE='cpu', vis=True):
    if preprocess_transform is None:
        input_tensor = image.clone()
        image = image.permute(1, 2, 0).numpy()
        max_, min_ = image.max(), image.min()
        image = np.uint8(255 * (image - min_) / (max_ - min_))
    else:
        input_tensor = preprocess_transform(image)

    output = model(input_tensor.unsqueeze(0).to(DEVICE))

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    if target is None:
        target = output[0].argmax(0).max().item()

    mask = output[0].argmax(0).detach().cpu().numpy()
    mask_uint8 = 255 * np.uint8(mask == target)
    mask_float = np.float32(mask == target)
    mask_mask = np.zeros(mask_float.shape)
    mask_mask[y_start:y_end, x_start:x_end] = 1
    mask_float = mask_float * mask_mask
    mask_uint8 = np.uint8(mask_uint8 * mask_mask)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(np.repeat(mask_uint8[:, :, None], 3, axis=-1))
    if vis is True:
        plt.show()
    plt.close()

    return image, mask_float


def dc(result, reference, label):
    result = result == label
    reference = reference == label
    # result = np.atleast_1d(result.astype(np.bool))
    # reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    dc = 2. * intersection / float(size_i1 + size_i2)
    return dc


def dilation(image, model, preprocess_transform, target=None, box=None, DEVICE='cpu',
             mask=None, kernel_size=5, threshold=0.2, iterations=100, original_prediction=None, skip_vis=10):
    dil_mask = mask.copy()
    dil_mask[dil_mask < threshold] = 0
    dil_mask[dil_mask > threshold] = 1
    images, masks = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(0, iterations):
        dil_mask = cv2.dilate(dil_mask.astype(np.uint8), kernel, iterations=i)
        if image.shape[0] <= 3:
            im = image * dil_mask[None, :, :]
        else:
            im = image * dil_mask[:, :, None]
        if i % skip_vis == 0:
            vis = True
        else:
            vis = False
        im_iter, mask_iter = utility_dilation(image=im, model=model, preprocess_transform=
        preprocess_transform, target=target, box=box,
                                              DEVICE=DEVICE, vis=vis)
        images.append(im_iter)
        masks.append(mask_iter)

    if box is None:
        if image.shape[0] <= 3:
            y_start, y_end, x_start, x_end = 0, image.shape[1], 0, image.shape[2]
        else:
            y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    mask_float = np.float32(original_prediction == target)
    mask_mask = np.zeros(mask_float.shape)
    print(y_start, y_end, x_start, x_end)
    mask_mask[y_start:y_end, x_start:x_end] = 1
    mask_float = mask_float * mask_mask

    a = []
    for i in masks:
        print(dc(mask_float, i, 1))
        a.append(dc(mask_float, i, 1))

    plt.figure()
    plt.scatter(range(len(a)), a)

    return images, masks, a


preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# np.array image
np_input_image = np.array(input_image)

n_masks, p1, window_size = 500, 0.1, (7, 7)
input_size = np_input_image.shape
pool_sizes, pool_modes, reshape_transformer = [0, 1, 2], [None, np.mean, np.mean], False
fig_size = (30, 50)
vis, vis_base, vis_rise, grid = False, False, True, True
preprocess_transform = preprocess_transform
initial_mask_size = (7, 7)
vis_skip = 500
target = 3
y_start, y_end, x_start, x_end = 150, 320, 450, 630

box = (y_start, y_end, x_start, x_end)

im, pred, rect = vis_predict(np_input_image, model, preprocess_transform=preprocess_transform,
                             DEVICE=DEVICE, mask=None, box=box, vis=vis)

results = [rect, pred]
results_masks = []

masks = generate_masks(n_masks=n_masks, input_size=input_size, p1=p1, initial_mask_size=initial_mask_size)
coef = rise_segmentation(masks, np_input_image, model, preprocess_transform=preprocess_transform,
                         target=target, box=box, DEVICE=DEVICE, vis=vis_rise, vis_skip=vis_skip)
mask, overlay = rise_aggregated(np_input_image, masks, coef, vis=vis_rise)
results.append(overlay)
results_masks.append(mask)

for i in range(len(results)):
    plt.subplot(1, len(results), i + 1)
    plt.imshow(results[i])
    if not grid:
        plt.axis('off')

ims_med, masks_med, dice_med = dilation(np_input_image, model, preprocess_transform, target=target, box=box,
                                        DEVICE=DEVICE,
                                        mask=results_masks[-2], kernel_size=5, threshold=0.2, iterations=10,
                                        original_prediction=results[1],
                                        skip_vis=2)
