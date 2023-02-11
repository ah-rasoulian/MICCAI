import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import json
from models.unet import *
from utils.dataset import *
from utils.utils import *
from utils.trainer import *
from torch.optim import Adam, AdamW
import torch.nn as nn
from models.focalnet import FocalNet
import torch
from monai.losses import FocalLoss
from monai.transforms import RandGaussianNoise, RandRotate, RandFlip, RandZoom
from torchvision.transforms import transforms
from numpy import deg2rad


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # extract args from config file
    epochs = config_dict["epochs"]
    lambda_loss = config_dict["lambda_loss"]
    batch_size = config_dict["batch_size"]
    lr = config_dict["lr"]
    embed_dims_unet = list(config_dict["embed_dims_unet"])
    embed_dims_focal = config_dict["embed_dims_focal"]
    fold_to_do = config_dict["fold_to_do"]

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    augmentation = transforms.Compose([RandGaussianNoise(prob=0.5, std=0.4),
                                       RandRotate(prob=0.5, range_x=deg2rad(45), range_y=deg2rad(45), range_z=deg2rad(45)),
                                       RandFlip(prob=0.5),
                                       RandZoom(prob=0.5)])

    train_sub_ses, valid_sub_ses, test_sub_ses = get_train_valid_test_sub_ses(data_path, fold_to_do, os.path.join(extra_path, "cross_validation_folds"))
    train_ds = AneurysmDataset(data_path, train_sub_ses, transform=augmentation)
    valid_ds = AneurysmDataset(data_path, valid_sub_ses)
    test_ds = AneurysmDataset(data_path, test_sub_ses)

    labels_counts = Counter(train_ds.labels)
    target_list = torch.tensor(train_ds.labels)
    class_weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]])
    class_weights = class_weights[target_list]
    train_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=image_label_collate, sampler=train_sampler, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, collate_fn=image_label_collate, pin_memory=True, num_workers=4)

    model = FocalNet(img_size=64, patch_size=1, in_chans=1, num_classes=1, embed_dim=embed_dims_focal, depths=[2, 2, 2, 2], focal_levels=[3, 3, 3, 3], focal_windows=[3, 3, 3, 3])
    print(model)
    early_stopping = EarlyStopping(model, 2, os.path.join(extra_path, "focal_checkpoint.pth"))
    loss_fn = FocalLoss()
    opt = AdamW(model.parameters(), lr=lr)
    epoch_number = 1
    while not early_stopping.early_stop and epochs <= epoch_number:
        m = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader)
        print(f"\nepoch {epoch_number}: train-loss:{m['train_cfm'].get_mean_loss()}, valid_loss:{m['valid_cfm'].get_mean_loss()}\n"
              f"train-acc={m['train_cfm'].get_accuracy().item()}, valid-acc={m['valid_cfm'].get_accuracy().item()}\n"
              f"train-recall:{m['train_cfm'].get_recall_sensitivity().item()}, valid-recall:{m['valid_cfm'].get_recall_sensitivity().item()}\n"
              f"train-precision:{m['train_cfm'].get_precision().item()}, valid-precision:{m['valid_cfm'].get_precision().item()}\n"
              f"train-specificity:{m['train_cfm'].get_specificity().item()}, valid-specificity:{m['valid_cfm'].get_specificity().item()}")
        early_stopping(m['valid_cfm'].get_mean_loss())
        epoch_number += 1


if __name__ == '__main__':
    main()
