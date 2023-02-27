import pickle
from utils.utils import *
from models.unet import *
from models.focalresconvnet import FocalResConvNet
from models.focalconvunet import FocalConvUNet
from models.focalunet import FocalUNet
from utils.dataset import *
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from timm.models.layers import to_3tuple


def validation(model, data_loader, cfm: ConfusionMatrix, loss_fn=None, device='cuda'):
    model.to(device)
    model.eval()
    pbar_valid = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    pbar_valid.set_description('validating')
    with torch.no_grad():
        for i, (sample, target_mask, dist_mask, target) in pbar_valid:
            sample, target_mask, dist_mask, target = sample.to(device), target_mask.to(device), dist_mask.to(device), target.to(device)
            pred_mask = model(sample)
            if loss_fn:
                loss = loss_fn(pred_mask, target_mask, dist_mask)
                cfm.add_loss(loss)
                cfm.add_number_of_samples(len(target))

            cfm.add_dice(dice_metric(pred_mask, target_mask))
            cfm.add_iou(intersection_over_union_metric(pred_mask, target_mask))


def monte_carlo_sampling(model, data_loader, cfm: ConfusionMatrix, device='cuda', n=10):
    model.to(device)
    model.eval()
    enable_dropout(model)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    pbar.set_description('validating monte-carlo')
    with torch.no_grad():
        for i, (sample, target_mask, dist_mask, target) in pbar:
            sample, target_mask, dist_mask, target = sample.to(device), target_mask.to(device), dist_mask.to(device), target.to(device)
            predictions = torch.empty(n, *target_mask.shape)
            for forward_passes in range(n):
                pred_mask = model(sample)
                predictions[forward_passes] = pred_mask

            pred_mask = torch.mean(predictions, dim=0)
            cfm.add_dice(dice_metric(pred_mask, target_mask))
            cfm.add_iou(intersection_over_union_metric(pred_mask, target_mask))


def test_segmentation(model, test_loader, device='cuda'):
    global image_affine, mask_affine

    model.to(device)
    pbar = tqdm(test_loader, total=len(test_loader))
    pbar.set_description("testing")
    i = 1
    _type = 'unet'
    with torch.no_grad():
        for sample, target_mask, dist_mask, target in pbar:
            sample, target_mask, target = sample.to(device), target_mask.to(device), target.to(device)
            if target == 0:
                pred_mask = model(sample)

                print(i, dice_metric(pred_mask, target_mask), intersection_over_union_metric(pred_mask, target_mask))
                pred_mask = torch.argmax(pred_mask, dim=1).type(torch.int8)
                target_mask = torch.argmax(target_mask, dim=1).type(torch.int8)
                save_nifti_image(os.path.join(f'samples/{_type}/image/image_{i:03d}.nii'), sample, image_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/mask/mask_{i:03d}.nii'), target_mask, mask_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_{i:03d}.nii'), pred_mask, mask_affine)
                i += 1
                if i > 50:
                    return


def main():
    global image_affine, mask_affine

    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="inference/inference_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    # extract args from config file
    setup = config_dict["setup"]
    checkpoint_name = config_dict["checkpoint_name"]

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    with open(os.path.join(extra_path, "data_split/test_sub_ses.pth"), "rb") as f:
        test_sub_ses = pickle.load(f)

    model = build_model(config_dict)

    checkpoint_path = os.path.join(extra_path, f"weights/{checkpoint_name}")
    load_model(model, checkpoint_path)
    test_ds = AneurysmDataset(data_path, test_sub_ses, shuffle=False, return_dist_map=False)
    image_affine = test_ds.image_affine
    mask_affine = test_ds.mask_affine

    if setup == "classification":
        test_loader = DataLoader(test_ds, batch_size=16)
        test_cfm = ConfusionMatrix()
        validation(model, test_loader, test_cfm)
        print_test_result(model.multitask, test_cfm)
    else:
        test_loader = DataLoader(test_ds, batch_size=1)
        test_segmentation(model, test_loader)


image_affine = None
mask_affine = None

if __name__ == "__main__":
    main()
