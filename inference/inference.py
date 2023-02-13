import pickle
from utils.utils import *
from models.unet import *
from models.focalnet import *
from utils.dataset import *
import os
from tqdm import tqdm
import torch
import argparse
import json
from torch.utils.data import DataLoader


def test(model, checkpoint_path, test_loader, device='cuda'):
    print("test result")
    load_model(model, checkpoint_path)

    model.to(device)
    cfm = ConfusionMatrix()
    pbar = tqdm(test_loader, total=len(test_loader))
    pbar.set_description("testing")
    with torch.no_grad():
        for sample, label in pbar:
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            cfm.add_prediction(torch.sigmoid(pred), label)

        cfm.compute_confusion_matrix()
        print(f"acc={cfm.get_accuracy().item()}\n"
              f"recall:{cfm.get_recall_sensitivity().item()}\n"
              f"precision:{cfm.get_precision().item()}\n"
              f"F1:{cfm.get_f1_score().item()}\n"
              f"specificity:{cfm.get_specificity().item()}")


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="inference/inference_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # extract args from config file
    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    num_workers = config_dict["num_workers"]
    model_name = config_dict["model_name"]

    unet_embed_dims = list(config_dict["unet_embed_dims"])

    focal_patch_size = config_dict["focal_patch_size"]
    focal_embed_dims = config_dict["focal_embed_dims"]
    focal_depths = list(config_dict["focal_depths"])
    focal_levels = list(config_dict["focal_levels"])
    focal_windows = list(config_dict["focal_windows"])

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    checkpoint_path = os.path.join(extra_path, f"/weights/{model_name}.pth")
    with open(os.path.join(extra_path, "data_split/test_sub_ses.pth"), "rb") as f:
        test_sub_ses = pickle.load(f)

    if model_name == "focalnet":
        model = FocalNet(img_size, focal_patch_size, in_ch, num_classes, focal_embed_dims, focal_depths, focal_levels=focal_levels, focal_windows=focal_windows)

    test_ds = AneurysmDataset(data_path, test_sub_ses)
    test_loader = DataLoader(test_ds, num_workers=num_workers)

    test(model, checkpoint_path, test_loader)


if __name__ == "main":
    main()
