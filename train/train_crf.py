import argparse
import json

import numpy as np
import torch

from utils.utils import *
import os
import pickle
from utils.dataset import AneurysmDataset
from models.crf import CRF
from torch.utils.data import DataLoader
from timm.models.layers import to_3tuple
from monai.metrics import compute_dice
from scipy.ndimage import binary_closing, binary_opening
import torch


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/crf_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    img_size = config_dict["img_size"]
    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]
    checkpoint_name = config_dict["checkpoint_name"]
    checkpoint_path = os.path.join(extra_path, f"weights/{checkpoint_name}")

    model = build_model(config_dict)
    load_model(model, checkpoint_path)

    with open(os.path.join(extra_path, "data_split/train-sub-ses.pth"), "rb") as f:
        test_sub_ses = pickle.load(f)

    test_ds = AneurysmDataset(data_path, test_sub_ses, shuffle=False, return_dist_map=False)
    image_affine = test_ds.image_affine
    mask_affine = test_ds.mask_affine
    test_loader = DataLoader(test_ds)

    crf_model = CRF(num_iters=1, num_classes=2, bilateral_weight=2, gaussian_weight=1, gaussian_spatial_sigma=10,
                    bilateral_spatial_sigma=64, bilateral_intensity_sigma=3)

    model.cuda()
    model.eval()
    i = 0
    j = 0
    for image, target_mask, dist_mask, target in test_loader:
        image, target_mask, dist_mask, target = image.to('cuda'), target_mask.to('cuda'), dist_mask.to('cuda'), target.to('cuda')
        if target == 1:
            # pred_mask = model(image)
            # pred_mask = F.softmax(pred_mask, dim=1)
            # crf_mask = crf_model(target_mask.cpu(), image)
            # print(target_mask.dtype, target_mask.min(), target_mask.max(), crf_mask.dtype, crf_mask.shape, crf_mask.min(), crf_mask.max())
            print(target_mask.shape)
            save_nifti_image(os.path.join(f"samples/unet/image_{i}.nii"), image.cpu(), image_affine)
            save_nifti_image(os.path.join(f"samples/unet/mask_{i}.nii"), target_mask.cpu(), image_affine)
            # save_nifti_image(os.path.join(f"samples/unet/crf_{i}.nii"), crf_mask.cpu(), mask_affine)
            # crf_mask = torch.argmax(crf_mask, dim=1)
            # print(crf_mask.min(), crf_mask.max())
            # image = nib.Nifti1Image(crf_mask.squeeze().cpu().numpy().astype("float32"), mask_affine)
            # nib.save(image, os.path.join(f"samples/unet/crf_{i}.nii"))
            i += 1
            if i > 5:
                break

            # crf_mask = crf_model(pred_mask, image).cuda()
            # crf_mask_onehot = pred_mask_to_onehot(crf_mask, False)
            # d1 = compute_dice(pred_mask_to_onehot(pred_mask), target_mask, False).item()
            # d2 = compute_dice(crf_mask_onehot, target_mask, False).item()
            # if d1 > d2:
            #     save_nifti_image(os.path.join(f"samples/unet/pred_mask_fg_{i}.nii"), pred_mask[0, 1].detach().cpu(), mask_affine)
            #     save_nifti_image(os.path.join(f"samples/unet/image_{i}.nii"), image.detach().cpu(), image_affine)
            #     save_nifti_image(os.path.join(f"samples/unet/mask_{i}.nii"), target_mask.detach().cpu(), image_affine)
            #     x = crf_mask_onehot[0, 1].detach().cpu()
            #     x = binary_closing(x, iterations=3)
            #     x = torch.Tensor(x)
            #     y = torch.zeros_like(crf_mask_onehot)
            #     y[0, 1] = x
            #     y[0, 0] = 1 - x
            #     d3 = compute_dice(y, target_mask, False).item()
            #     save_nifti_image(os.path.join(f"samples/unet/crf_mask_1_{i}.nii"), x, mask_affine)
            #     save_nifti_image(os.path.join(f"samples/unet/crf_mask_{i}.nii"), pred_mask_to_onehot(crf_mask), mask_affine)
            #     # save_nifti_image(os.path.join("samples/unet/crf_mask_2.nii"), crf_mask[0, 2].detach().cpu(), image_affine)
            #     # save_nifti_image(os.path.join("samples/unet/crf_mask_3.nii"), crf_mask[0, 3].detach().cpu(), image_affine)
            #     print(j, i, d1, d2, d3)
            #     if i > 10:
            #         break
            #     i += 1
            # j += 1


if __name__ == '__main__':
    main()
