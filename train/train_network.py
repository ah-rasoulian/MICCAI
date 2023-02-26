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
    model_name = config_dict["model"]
    assert model_name in ["unet", "focalconvunet", "focalunet", "swinunetr"]
    multitask = str_to_bool(config_dict["multitask"])
    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    epochs = config_dict["epochs"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    lr = config_dict["lr"]
    do_augmentation = str_to_bool(config_dict["do_augmentation"])
    do_sampling = str_to_bool(config_dict["do_sampling"])
    use_validation = str_to_bool(config_dict["use_validation"])
    validation_ratio = config_dict["validation_ratio"]

    unet_embed_dims = list(config_dict["unet_embed_dims"])

    focal_patch_size = config_dict["focal_patch_size"]
    focal_embed_dims = config_dict["focal_embed_dims"]
    focal_depths = list(config_dict["focal_depths"])
    focal_levels = list(config_dict["focal_levels"])
    focal_windows = list(config_dict["focal_windows"])

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    augmentation = None
    if do_augmentation:
        augmentation = Augmentation()

    if use_validation:
        train_sub_ses, valid_sub_ses, test_sub_ses = train_valid_test_split(data_path, os.path.join(extra_path, 'data_split'), validation_ratio, override=True)
    else:
        train_sub_ses, _, valid_sub_ses = train_valid_test_split(data_path, os.path.join(extra_path, 'data_split'), 0, override=True)

    train_ds = AneurysmDataset(data_path, train_sub_ses, transform=augmentation, shrink_masks=False)
    valid_ds = AneurysmDataset(data_path, valid_sub_ses, shrink_masks=False)

    train_sampler = None
    labels_counts = Counter(train_ds.labels)
    if do_sampling:
        target_list = torch.tensor(train_ds.labels)
        weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
        class_weights = weights[target_list]
        train_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

    if model_name == "unet":
        model = UNet(in_ch, num_classes, unet_embed_dims, multitask=multitask)
    elif model_name == 'swinunetr':
        model = SwinUNETR(img_size=to_3tuple(img_size), in_channels=in_ch, out_channels=num_classes, feature_size=48)
    elif model_name == "focalconvunet":
        model = FocalConvUNet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes,
                              embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=True, multitask=multitask)
    else:
        model = FocalUNet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes,
                          embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=True, multitask=multitask)

    loss_fn = MultitaskDiceBoundaryLoss()

    opt = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', patience=5)
    print(model)
    checkpoint_name = model.__class__.__name__
    early_stopping = EarlyStopping(model, 5, os.path.join(extra_path, f"weights/{checkpoint_name}.pt"))
    epoch_number = 1
    train_losses = []
    valid_losses = []
    while not early_stopping.early_stop and epochs <= epoch_number:
        _metrics = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader, multitask=False)
        val_loss = _metrics['valid_cfm'].get_mean_loss()

        train_losses.extend(_metrics['train_cfm'].losses)
        valid_losses.extend(_metrics['valid_cfm'].losses)
        visualize_losses(train_losses, valid_losses)

        print(f"\nepoch {epoch_number}: train-loss:{_metrics['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}\n"
              f"train-dice={_metrics['train_cfm'].get_mean_dice()}, valid-dice={_metrics['valid_cfm'].get_mean_dice()}\n"
              f"train-iou={_metrics['train_cfm'].get_mean_iou()}, valid-iou={_metrics['valid_cfm'].get_mean_iou()}")
        if model.multitask:
            _metrics["train_cfm"].compute_confusion_matrix()
            _metrics["valid_cfm"].compute_confusion_matrix()
            print(f"train-acc={_metrics['train_cfm'].get_accuracy().item()}, valid-acc={_metrics['valid_cfm'].get_accuracy().item()}\n"
                  f"train-recall:{_metrics['train_cfm'].get_recall_sensitivity().item()}, valid-recall:{_metrics['valid_cfm'].get_recall_sensitivity().item()}\n"
                  f"train-precision:{_metrics['train_cfm'].get_precision().item()}, valid-precision:{_metrics['valid_cfm'].get_precision().item()}\n"
                  f"train-F1:{_metrics['train_cfm'].get_f1_score().item()}, valid-F1:{_metrics['valid_cfm'].get_f1_score().item()}\n"
                  f"train-specificity:{_metrics['train_cfm'].get_specificity().item()}, valid-specificity:{_metrics['valid_cfm'].get_specificity().item()}")

        early_stopping(val_loss)
        scheduler.step(val_loss)
        epoch_number += 1

    if use_validation:
        test_ds = AneurysmDataset(data_path, test_sub_ses)
        test_cfm = ConfusionMatrix()
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
        validation(model, test_loader, test_cfm)

        print(f"\ntest result:"
              f"dice={test_cfm.get_mean_dice()}, iou={test_cfm.get_mean_iou()}\n")
        if model.multitask:
            test_cfm.compute_confusion_matrix()
            print(f"acc={test_cfm.get_accuracy().item()} specificity:{test_cfm.get_specificity().item()}\n"
                  f"precision:{test_cfm.get_precision().item()} recall:{test_cfm.get_recall_sensitivity().item()} F1:{test_cfm.get_f1_score().item()}\n")


if __name__ == '__main__':
    main()
