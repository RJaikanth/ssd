from typing import Tuple

import os
import json

import cv2
import torch
import torch.utils.data as D

from .utils.transforms import transform

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class PascalVOCDataset(D.Dataset):
    """
    Dataset for the Pascal VOC Dataset.

    Parameters
    ----------
    data_folder : str
        Path to data folder.
    split : str
        {train, test}
        Test/Train Split
    keep_difficult : bool, optional
        Keep Difficult objects if True, by default False
    """

    def __init__(self, data_folder: str, split: str, keep_difficult: bool = False):
        self.split = split.upper()
        assert self.split in {"TRAIN", "TEST"}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, self.split + "_images.json")) as f:
            self.images = json.load(f)
        with open(os.path.join(data_folder, self.split + "_objects.json")) as f:
            self.objects = json.load(f)
        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Read Image
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[idx]
        boxes = torch.FloatTensor(objects["boxes"])
        labels = torch.LongTensor(objects["labels"])
        difficulties = torch.ByteTensor(objects["difficulties"])

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformation
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def collate_fn(self, batch) -> Tuple[torch.tensor, list, list, list]:
        """
        Function to collate objects, labels and difficulties

        We will store all objects, labels and difficulties in separate lists

        Parameters
        ----------
        batch :
            An iterable of n sets from __getitem__()

        Returns
        -------
        [torch.tensor, list, list, list]:
            A tensor of images, losts of varying size tensors of bounding boxes, labels and difficulties
        """
        images, boxes, labels, difficulties = [], [], [], []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties


if __name__ == "__main__":
    ds = PascalVOCDataset("./data/", "TRAIN")

    train_dl = D.DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        collate_fn=ds.collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    for i, (images, _, _, _) in enumerate(train_dl):
        if images.type == torch.uint8:
            print(i, images.shape)
