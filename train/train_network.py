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
from monai.transforms import RandGaussianNoise, RandRotate, RandFlip, RandZoom, Compose
from numpy import deg2rad
import cv2
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # extract args from config file
    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    epochs = config_dict["epochs"]
    lambda_loss = config_dict["lambda_loss"]
    batch_size = config_dict["batch_size"]
    lr = config_dict["lr"]
    num_workers = config_dict["num_workers"]
    do_augmentation = bool(config_dict["do_augmentation"])

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
        augmentation = Compose([RandGaussianNoise(prob=0.5, std=0.4),
                                RandRotate(prob=0.5, range_x=deg2rad(90), range_y=deg2rad(90), range_z=deg2rad(90)),
                                RandFlip(prob=0.5),
                                RandZoom(prob=0.5)])

    train_sub_ses, valid_sub_ses, test_sub_ses = get_train_valid_test_sub_ses(data_path)
    train_ds = AneurysmDataset(data_path, train_sub_ses, transform=augmentation)
    # for i in range(len(train_ds)):
    #     x, m, y = train_ds[i]
    #     print(x.min(), x.max())
    #     if y == 0:
    #         continue
    #     for j in range(m.shape[0]):
    #         print(j)
    #         cv2.imshow('w', m[j].numpy())
    #         cv2.imshow('im', x[0, j].numpy())
    #         cv2.waitKey()
    # return

    valid_ds = AneurysmDataset(data_path, valid_sub_ses)
    # test_ds = AneurysmDataset(data_path, test_sub_ses)

    labels_counts = Counter(train_ds.labels)
    target_list = torch.tensor(train_ds.labels)
    class_weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]])
    class_weights = class_weights[target_list]
    train_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=image_label_collate, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, collate_fn=image_label_collate, num_workers=num_workers)

    model = FocalNet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes,
                     embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows)

    loss_fn = FocalLoss()
    opt = AdamW(model.parameters(), lr=lr)
    print(model)
    checkpoint_name = model.__class__.__name__ + "_" + loss_fn.__class__.__name__
    early_stopping = EarlyStopping(model, 5, os.path.join(extra_path, f"weights/{checkpoint_name}.pth"))
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
