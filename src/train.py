import time
import json
import argparse

import torch.optim
import torch.utils.data as D
import torch.backends.cudnn as cudnn

from .engine import train
from .dataset import PascalVOCDataset
from .model import SSD300, MultiBoxLoss
from .utils.helpers import adjust_learning_rate, save_checkpoint


cudnn.benchmark = True


def trainAndEval(data_config, model_config, exp_config):
    """
    Training
    """

    if model_config["checkpoint"] is None:
        start_epoch = 0
        model = SSD300(n_classes=model_config["n_classes"])

        # Initialize the optimizer with twice the lr for biases
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(
            params=[{"params": biases, "lr": 2 * exp_config["lr"]}, {"params": not_biases}],
            lr=exp_config["lr"],
            momentum=exp_config["momentum"],
            weight_decay=exp_config["weight_decay"],
        )
    else:
        checkpoint = torch.load(model_config["checkpoint"])
        start_epoch = checkpoint["epoch"] + 1
        print("\n Loaded checkpoint from epoch {}.\n".format(start_epoch))
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

    # Move to device
    model = model.to(exp_config["device"])
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(exp_config["device"])

    # Custom DataLoaders
    train_ds = PascalVOCDataset(
        data_config["lists_folder"], split="train", keep_difficult=data_config["keep_difficult"]
    )
    train_dl = D.DataLoader(
        train_ds,
        batch_size=exp_config["batch_size"],
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=exp_config["workers"],
        pin_memory=True,
    )

    # Calculate total number of epochs to trainand the epochs to decay learning rate at.
    epochs = exp_config["iterations"] // (len(train_ds) // exp_config["batch_size"])
    decay_lr_at = [it // (len(train_ds) // exp_config["batch_size"]) for it in exp_config["decay_lr_at"]]

    for epoch in range(start_epoch, epochs):
        # Decay lr at particular epochs.
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, exp_config["decay_lr_to"])

        train(
            train_dl=train_dl,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            exp_config=exp_config,
        )

        save_checkpoint(epoch, model, optimizer)
        print("Saved Model after epoch - {}".format(epoch))
        print("="*30)
