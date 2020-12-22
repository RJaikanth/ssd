import time
import torch

from src.dataset import PascalVOCDataset
from .utils.helpers import AverageMeter, clip_gradient


def train(train_dl, model, criterion, optimizer, epoch, exp_config):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_dl):
        data_time.update(time.time() - start)

        # Move to device
        images = images.float()
        images = images.to(exp_config["device"])
        boxes = [b.to(exp_config["device"]) for b in boxes]
        labels = [l.to(exp_config["device"]) for l in labels]

        # Forward prop
        predicted_locs, predicted_scores = model(images)

        # Loss
        loss = criterion(
            predicted_locs.add_(torch.ones(predicted_locs.size()).to(exp_config["device"])),
            predicted_scores,
            boxes,
            labels,
        )

        # Backward prop
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if exp_config["grad_clip"] is not None:
            clip_gradient(optimizer, exp_config["grad_clip"])

        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % exp_config["print_freq"] == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time - {batch_time.avg:.3f}  "
                "Data Time - {data_time.avg:.3f}  "
                "Loss - {loss.avg:.4f}".format(
                    epoch, i, len(train_dl), batch_time=batch_time, data_time=data_time, loss=losses
                )
            )
    # Free memory
    del predicted_locs, predicted_scores, images, boxes, labels
