from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch.functional import F

import torch
import random

from .metrics import iou


def expand(
    image: torch.TensorType, boxes: torch.TensorType, filler: list
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    Helps to learn detect small object.

    Parameters
    ----------
    image : torch.TensorType
        Input image (3, original_h, original_w)
    boxes : torch.TensorType
        Bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    filler : list
        RGB values of the filler material, a list like [R, G, B]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Expanded image, updated bounding box coordinates.
    """
    max_scale = 4
    scale = random.uniform(1, max_scale)

    original_h = image.size(1)
    original_w = image.size(2)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler.
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    # Place to orignial image at random coordinates in this new image (origin at the top-left of iamge)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes


def random_crop(
    image: torch.TensorType,
    boxes: torch.TensorType,
    labels: torch.TensorType,
    difficulties: torch.TensorType,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Randomly cropout a part of the image

    Parameters
    ----------
    image : torch.TensorType
        Image of the shape (3, h, w)
    boxes : torch.TensorType
        Bounding boxes in boundary coordinates (Pascal VOC Format.)
    labels : torch.TensorType
        Lables of objects
    difficulties : torch.TensorType
        Difficulties of objects

    Returns
    -------
    Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
        Return the cropped image and respective bboxes, lables and difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)

    # Keep choosing minimum overlap until successfull crop is made.
    while True:
        min_overlap = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None])  # None is no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try upto 50 times
        max_trials = 50
        for _ in range(max_trials):
            # crop(scale) dimensions must be in [0.3, 1]
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])

            # Calculate IoU of crop and bounding boxes
            iou_score = iou(crop.unsqueeze(0), boxes)
            iou_score = iou_score.squeeze(0)

            if iou_score.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]

            # Find center
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (
                (bb_centers[:, 0] > left)
                * (bb_centers[:, 0] < right)
                * (bb_centers[:, 1] > top)
                * (bb_centers[:, 1] < bottom)
            )

            # Continue if centers
            if not centers_in_crop.any():
                continue

            # Discard boxes that do not meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image: np.array, bboxes: np.array) -> Tuple[torch.tensor, torch.tensor]:
    """
    Perform a Horizontal Flip operation on the image.

    Parameters
    ----------
    image : torch.TensorType
        Image in the shape (h, w, 3)

    bboxes : torch.TensorType
        Bounding Boxes of objects.

    Returns
    -------
    Tuple[torch.tensor, torch.tensor]
        Returns flipped image and bounding boxes.
    """
    flip = A.HorizontalFlip(p=1.0)
    new_image = flip(image=image)["image"]

    new_boxes = bboxes
    new_boxes[:, 0] = image.shape[1] - bboxes[:, 0] - 1
    new_boxes[:, 2] = image.shape[1] - bboxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(
    image: np.array,
    bboxes: np.array,
    dims: Tuple[int, int] = (300, 300),
    return_percent_coords: bool = True,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Resize the image and bounding boxes.

    Parameters
    ----------
    image : np.array
        Image of the shape (h, w, 3)
    bboxes : np.array
        Bounding boxes of objects in the image
    dims : Tuple[int, int], optional
        Dimensions of resized image, by default (300, 300)
    return_percent_coords : bool, optional
        Return bounding box coordinates as percentage of previous dimensions, by default True

    Returns
    -------
    Tuple[torch.tensor, torch.tensor]
        Returns flipped image and bounding boxes.
    """
    new_image = A.Resize(300, 300)(image=image)["image"]

    old_dims = torch.FloatTensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    new_boxes = bboxes / old_dims

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image: np.array) -> np.array:
    """
    Randomly perform photometric distortion on the images.

    Distortions -
        * Brightness
        * Contrast
        * Saturation
        * Hue

    Parameters
    ----------
    image : np.array
        Original Image

    Returns
    -------
    [np.array]
        Distorted Image
    """
    new_image = Image.fromarray(image)

    distortions = [
        F.adjust_brightness,
        F.adjust_contrast,
        F.adjust_saturation,
        F.adjust_hue,
    ]
    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == "adjust_hue":
                adjust_factor = random.uniform(-19 / 255.0, 18 / 255.0)
            else:
                adjust_factor = random.uniform(0.5, 1.5)

            new_image = d(new_image, adjust_factor)

    return np.array(new_image)


def transform(
    image: np.array,
    boxes: torch.TensorType,
    labels: torch.TensorType,
    difficulties: torch.TensorType,
    split: torch.TensorType,
) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType, torch.TensorType]:
    """
    Randomly applies transformation to the image.

    Parameters
    ----------
    image : np.array
        Input image
    boxes : torch.TensorType
        Bounding Boxes of image
    labels : torch.TensorType
        Labels of objects in image
    difficulties : torch.TensorType
        Difficulties of objects in image
    split : torch.TensorType
        {"TRAIN", "TEST"}

    Returns
    -------
    Tuple[ torch.TensorType, torch.TensorType, torch.TensorType, torch.TensorType ]
        Return updated image, boxes, labels and difficulties.
    """
    assert split.upper() in ("TRAIN", "TEST")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    if split == "TRAIN":
        new_image = photometric_distort(new_image)

        new_image = torch.tensor(new_image.transpose(2, 0, 1))
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, new_boxes, filler=mean)

        new_image, new_boxes, new_labels, new_difficulties = random_crop(
            new_image,
            new_boxes.clone().float(),
            new_labels,
            new_difficulties,
        )

        new_image = np.array(new_image)
        new_image = new_image.transpose(1, 2, 0)

        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))
    # new_image = A.Normalize(mean, std, 1)(image=new_image / 255.0)["image"]
    new_image = torch.tensor(new_image.transpose(2, 0, 1))

    return new_image, new_boxes, new_labels, new_difficulties


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from .utils import parse_annotation, LABEL_MAP_R, LABEL_CMAP
    from .vizualise import draw_bboxes

    image_path = "/home/raghhuveerj/code/data/pascal_VOC/VOCdevkit/VOC2007/JPEGImages/"
    annot_path = "/home/raghhuveerj/code/data/pascal_VOC/VOCdevkit/VOC2007/Annotations/"

    images = sorted(os.listdir(image_path))
    annots = sorted(os.listdir(annot_path))

    index = random.randint(0, len(images))
    image = images[index]
    annot = annots[index]

    image = os.path.join(image_path, image)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annot = os.path.join(annot_path, annot)
    annot = parse_annotation(annot)

    new_image, new_boxes, new_labels, _ = transform(
        image,
        torch.tensor(annot["boxes"]),
        torch.tensor(annot["labels"]),
        torch.tensor(annot["difficulties"]),
        "TRAIN",
    )

    draw_bboxes(
        np.array(new_image).transpose(1, 2, 0).astype(np.uint8),
        np.array(new_boxes),
        np.array(new_labels),
    )
