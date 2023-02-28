import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import json
from models.unet import UNet
from models.focalresconvnet import FocalResConvNet
from models.focalconvunet import FocalConvUNet
from models.focalunet import FocalUNet
from utils.dataset import AneurysmDataset, train_valid_test_split
from utils.utils import Augmentation, EarlyStopping
from utils.trainer import train_one_epoch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch
from collections import Counter
from monai.networks.nets import SwinUNETR
from timm.models.layers import to_3tuple
from utils.utils import *
from inference.inference import validation
import matplotlib.pyplot as plt
from utils.losses import *


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # extract args from config file
    epochs = config_dict["epochs"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    lr = config_dict["lr"]
    do_augmentation = str_to_bool(config_dict["do_augmentation"])
    do_sampling = str_to_bool(config_dict["do_sampling"])
    use_validation = str_to_bool(config_dict["use_validation"])
    validation_ratio = config_dict["validation_ratio"]

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    augmentation = None
    if do_augmentation:
        augmentation = Augmentation()

    if use_validation:
        train_sub_ses, valid_sub_ses, test_sub_ses = train_valid_test_split(data_path, os.path.join(extra_path, 'data_split'), validation_ratio, override=True)
    else:
        train_sub_ses, _, valid_sub_ses = train_valid_test_split(data_path, os.path.join(extra_path, 'data_split'), 0, override=True)

    train_ds = AneurysmDataset(data_path, train_sub_ses, transform=augmentation)
    valid_ds = AneurysmDataset(data_path, valid_sub_ses)

    train_sampler = None
    if do_sampling:
        labels_counts = Counter(train_ds.labels)
        target_list = torch.tensor(train_ds.labels)
        weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
        class_weights = weights[target_list]
        train_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

    model = build_model(config_dict)
    checkpoint_name = model.__class__.__name__
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("model: ", checkpoint_name, " num-params:", num_params)
    loss_fn = CEDiceBoundaryLoss()

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.9)
    early_stopping = EarlyStopping(model, 5, os.path.join(extra_path, f"weights/{checkpoint_name}.pt"))
    epoch_number = 1
    train_losses = []
    valid_losses = []
    while not early_stopping.early_stop and epochs <= epoch_number:
        _metrics = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader)
        val_loss = _metrics['valid_cfm'].get_mean_loss()

        train_losses.extend(_metrics['train_cfm'].losses)
        valid_losses.extend(_metrics['valid_cfm'].losses)
        visualize_losses(train_losses, valid_losses)

        print(f"\nepoch {epoch_number}: train-loss:{_metrics['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}\n"
              f"train-dice={_metrics['train_cfm'].get_mean_dice()}, valid-dice={_metrics['valid_cfm'].get_mean_dice()}\n"
              f"train-iou={_metrics['train_cfm'].get_mean_iou()}, valid-iou={_metrics['valid_cfm'].get_mean_iou()}")

        early_stopping(val_loss)
        scheduler.step(val_loss)
        epoch_number += 1

    if use_validation:
        test_ds = AneurysmDataset(data_path, test_sub_ses, return_dist_map=False)
        test_cfm = ConfusionMatrix()
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
        validation(model, test_loader, test_cfm)
        print_test_result(test_cfm)


if __name__ == '__main__':
    main()
