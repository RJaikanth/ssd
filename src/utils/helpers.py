import os
import json
import torch

from xml.etree import ElementTree as ET

from .defaults import LABEL_MAP


def parse_annotation(annotation_path: str) -> dict:
    """
    Parse an XML annotation file and return the boxes, labels and difficulties

    Parameters
    ----------
    annotation_path : str
        Path to annotation XML file.

    Returns
    -------
    dict:
        Dictionary containing boxes, labels and difficulties as the keys.

    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes, labels, difficulties = [], [], []

    for object_ in root.iter("object"):
        # Parse Difficulty of object
        difficult = int(object_.find("difficult").text == "1")

        # Parse label of object
        label = object_.find("name").text.lower().strip()
        if label not in LABEL_MAP:
            continue

        bbox = object_.find("bndbox")
        xmin = int(bbox.find("xmin").text) - 1
        ymin = int(bbox.find("ymin").text) - 1
        xmax = int(bbox.find("xmax").text) - 1
        ymax = int(bbox.find("ymax").text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(LABEL_MAP[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}


def decimate(tensor: torch.tensor, m: list) -> torch.tensor:
    """n
    Function to decimate tensor every m steps

    Parameters
    ----------
    tensor : torch.tensor
        Tensor to be decimated.
    m : list
        List of factor for each axis in the tensor.

    Returns
    -------
    torch.tensor
        Decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d,
                index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long(),
            )

    return tensor


def check_config(args, flag):
    from argparse import ArgumentError

    args_dict = vars(args)
    if args_dict[flag] is None:
        raise ArgumentError(args_dict[flag], f"--{flag} must be passed.")


def read_config(base_path, filename):
    config_path = os.path.join(base_path, filename)
    with open(config_path, "r") as f:
        return json.load(f)


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * decay_rate

    print("Decaying Learning Rate.\n The new LR is {}".format(optimizer.param_groups[1]["lr"]))


def clip_gradient(optimzer, grad_clip):
    for group in optimzer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_checkpoint(epoch, model, optimizer):
	state = {
		"epoch": epoch,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict()
	}
	filename = "checkpoint_ssd300_run1.pth.tar"
	torch.save(state, filename)
