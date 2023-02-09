import os
from collections import Counter
from dataset import *
from torch.utils.data import DataLoader
import argparse
import json
from models.unet import *


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default='config.json')
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

    train_sub_ses, valid_sub_ses, test_sub_ses = get_train_valid_test_sub_ses(fold_to_do, os.path.join(extra_path, "cross_validation_folds"))
    train_ds = AneurysmDataset(data_path, train_sub_ses)
    valid_ds = AneurysmDataset(data_path, valid_sub_ses)
    test_ds = AneurysmDataset(data_path, test_sub_ses)

    _all = find_sub_ses_pairs(os.path.join(data_path, "Negative_Patches"))
    ds = AneurysmDataset(data_path, _all)
    dl = DataLoader(ds)
    model = UNet3D(1, 1, embed_dims)
    print(model)
    for x, m, y in dl:
        p = model(x)
        print(x.shape, m.shape, y.shape, p.shape)
        break


if __name__ == '__main__':
    main()
