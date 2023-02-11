import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import json
from models.unet import *
from utils.dataset import *
from utils.utils import *
from utils.trainer import *
from torch.optim import Adam
import torch.nn as nn
from models.focalnet import FocalNet
import torch


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
    embed_dims = list(config_dict["embed_dims"])
    fold_to_do = config_dict["fold_to_do"]

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    train_sub_ses, valid_sub_ses, test_sub_ses = get_train_valid_test_sub_ses(data_path, fold_to_do, os.path.join(extra_path, "cross_validation_folds"))
    train_ds = AneurysmDataset(data_path, train_sub_ses)
    valid_ds = AneurysmDataset(data_path, valid_sub_ses)
    test_ds = AneurysmDataset(data_path, test_sub_ses)

    labels_counts = Counter(train_ds.labels)
    target_list = torch.tensor(train_ds.labels)
    class_weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]])
    class_weights = class_weights[target_list]
    train_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=False)

    train_loader = DataLoader(train_ds, batch_size=16, collate_fn=image_label_collate, sampler=train_sampler)
    valid_loader = DataLoader(valid_ds, batch_size=16, collate_fn=image_label_collate)

    model = FocalNet(img_size=64, patch_size=2, in_chans=1, num_classes=1, embed_dim=96, depths=[2, 2, 6, 2], focal_levels=[3, 3, 3, 3], focal_windows=[3, 3, 3, 3])
    print(model)
    early_stopping = EarlyStopping(model, 3, "focal_checkpoint.pth")
    loss_fn = FocalBCELoss(gamma=5)
    opt = Adam(model.parameters(), lr=1e-4)
    epoch = 1
    while not early_stopping.early_stop:
        m = train_one_epoch(model, opt, loss_fn, train_loader, valid_loader)
        print(f"\nepoch {epoch}: train-loss:{m['train_cfm'].get_mean_loss()}, valid_loss:{m['valid_cfm'].get_mean_loss()}\n"
              f"train-acc={m['train_cfm'].get_accuracy()}, valid-acc={m['valid_cfm'].get_accuracy()}, train-F1:{m['train_cfm'].get_f1_score()}, valid-F1:{m['valid_cfm'].get_f1_score()}")
        early_stopping(m['valid_cfm'].get_mean_loss())
        epoch += 1


if __name__ == '__main__':
    main()
