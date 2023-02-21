import os
import pickle
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import json
from models.unet import *
from models.weakfocal import WeakFocalNet3D
from models.multitaskfocalunet import MultiTaskFocalUnet
from utils.dataset import *
from utils.utils import *
from utils.trainer import *
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from models.focalnet import FocalNet
import torch
from monai.losses import FocalLoss
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from inference.inference import test
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # extract args from config file
    model_name = config_dict["model"]
    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    epochs = config_dict["epochs"]
    batch_size = config_dict["batch_size"]
    lr = config_dict["lr"]
    num_workers = config_dict["num_workers"]
    do_augmentation = bool(config_dict["do_augmentation"])
    do_sampling = bool(config_dict["do_sampling"])
    use_validation = bool(config_dict["use_validation"])
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

    train_ds = AneurysmDataset(data_path, train_sub_ses, transform=augmentation)
    valid_ds = AneurysmDataset(data_path, valid_sub_ses)

    train_sampler = None
    labels_counts = Counter(train_ds.labels)
    if do_sampling:
        target_list = torch.tensor(train_ds.labels)
        weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])
        class_weights = weights[target_list]
        train_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

    if model_name == "weakfocalnet3d":
        model = WeakFocalNet3D(img_size, focal_patch_size, in_ch, num_classes, focal_embed_dims, focal_depths, focal_levels, focal_windows)
        setup = "classification"
    elif model_name == "unet3d":
        model = UNet3D(in_ch, num_classes, unet_embed_dims)
        setup = "segmentation"
    elif model_name == "multitaskunet3d":
        model = MultiTaskUNet3D(in_ch, num_classes, unet_embed_dims)
        setup = "multitask"
    else:
        model = MultiTaskFocalUnet(img_size, focal_patch_size, in_ch, num_classes, focal_embed_dims, focal_depths, focal_levels, focal_windows)
        setup = "multitask"

    if setup == "classification":
        loss_fn = nn.BCEWithLogitsLoss()
    elif setup == "segmentation":
        loss_fn = DiceBCELoss()
    else:
        loss_fn = (0.5, nn.BCEWithLogitsLoss(), DiceBCELoss())

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(opt, mode='min', patience=5)
    print(model)
    checkpoint_name = model.__class__.__name__
    early_stopping = EarlyStopping(model, 5, os.path.join(extra_path, f"weights/{checkpoint_name}.pth"))
    epoch_number = 1
    while not early_stopping.early_stop and epochs <= epoch_number:
        m = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader, train_type=setup)
        val_loss = m['valid_cfm'].get_mean_loss()
        print(f"\nepoch {epoch_number}: train-loss:{m['train_cfm'].get_mean_loss()}, valid_loss:{val_loss}")
        if setup in ["classification", "multitask"]:
            print(f"train-acc={m['train_cfm'].get_accuracy().item()}, valid-acc={m['valid_cfm'].get_accuracy().item()}\n"
                  f"train-recall:{m['train_cfm'].get_recall_sensitivity().item()}, valid-recall:{m['valid_cfm'].get_recall_sensitivity().item()}\n"
                  f"train-precision:{m['train_cfm'].get_precision().item()}, valid-precision:{m['valid_cfm'].get_precision().item()}\n"
                  f"train-F1:{m['train_cfm'].get_f1_score().item()}, valid-F1:{m['valid_cfm'].get_f1_score().item()}\n"
                  f"train-specificity:{m['train_cfm'].get_specificity().item()}, valid-specificity:{m['valid_cfm'].get_specificity().item()}")
        if setup in ["segmentation", "multitask"]:
            print(f"train-dice={m['train_cfm'].get_mean_dice()}, valid-dice={m['valid_cfm'].get_mean_dice()}")

        early_stopping(val_loss)
        scheduler.step(val_loss)
        epoch_number += 1

    # if use_validation:
    #     checkpoint_path = os.path.join(extra_path, f"weights/{checkpoint_name}.pth")
    #     test_ds = AneurysmDataset(data_path, test_sub_ses)
    #     test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=image_label_collate)
    #     test(model, checkpoint_path, test_loader)


if __name__ == '__main__':
    main()
