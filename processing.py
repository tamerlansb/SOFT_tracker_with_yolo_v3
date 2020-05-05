import cv2
import numpy as np
import torch
from skimage.io import imread


def image_to_net_inp(image, resolution):
    image_to_detection, trans = letterbox_image(image, (resolution, resolution))
    inp = (torch.from_numpy(image_to_detection).float().permute(2, 0, 1) / 255).unsqueeze(0)
    return inp, trans


def letterbox_image(img, dim):
    # Create the background
    image = np.full(dim + (3,), 128)

    img_dim = (img.shape[1], img.shape[0])
    box_w, box_h, box_x, box_y, ratio = letterbox_transforms(img_dim, dim)
    box_image = cv2.resize(img, (box_w, box_h), interpolation=cv2.INTER_CUBIC)

    # Put the box image on top of the blank image
    image[box_y:box_y + box_h, box_x:box_x + box_w] = box_image

    return image, (box_w, box_h, box_x, box_y, ratio)


def letterbox_transforms(inner_dim, outer_dim):
    outer_w, outer_h = outer_dim
    inner_w, inner_h = inner_dim
    ratio = min(outer_w / inner_w, outer_h / inner_h)
    box_w = int(inner_w * ratio)
    box_h = int(inner_h * ratio)
    box_x_offset = (outer_w // 2) - (box_w // 2)
    box_y_offset = (outer_h // 2) - (box_h // 2)
    return box_w, box_h, box_x_offset, box_y_offset, ratio


# Mode - letterbox, resize
def load_image(img_path, mode=None, dim=None):
    img = imread(img_path)
    trans = None
    if mode is not None and dim is not None:
        if mode == 'letterbox':
            img, trans = letterbox_image(img, dim)
        elif mode == 'resize':
            img = cv2.resize(img, dim)

    img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
    return img, trans


def bbox_transform(box, x_max, y_max, x_offset, y_offset, ratio):
    """
    :param box:
    :param x_max:
    :param y_max:
    :param x_offset:
    :param y_offset:
    :param ratio:
    :return:
    """
    box[:,[0,2]] = torch.clamp((box[:,[0,2]] - x_offset) / ratio, 0, x_max)
    box[:,[1,3]] = torch.clamp((box[:,[1,3]] - y_offset) / ratio, 0, y_max)
    return box
