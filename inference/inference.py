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
import argparse
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F


def validation(model, data_loader, cfm: ConfusionMatrix, loss_fn=None, device='cuda'):
    model.to(device)
    model.eval()
    pbar_valid = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    pbar_valid.set_description('validating')
    with torch.no_grad():
        for i, (sample, target_mask, dist_mask, target) in pbar_valid:
            sample, target_mask, dist_mask, target = sample.to(device), target_mask.to(device), dist_mask.to(device), target.to(device)
            prediction = model(sample)
            if loss_fn:
                loss = loss_fn(prediction, (target, target_mask), dist_mask)
                cfm.add_dice(loss)
                cfm.add_number_of_samples(len(target))

            if model.multitask:
                pred, pred_mask = prediction
                cfm.add_prediction(pred, target)
            else:
                pred_mask = prediction

            cfm.add_dice(dice_metric(pred_mask, target_mask))
            cfm.add_iou(intersection_over_union_metric(pred_mask, target_mask))


def test_segmentation(model, checkpoint_path, test_loader, device='cuda'):
    global image_affine, mask_affine
    load_model(model, checkpoint_path)

    model.to(device)
    pbar = tqdm(test_loader, total=len(test_loader))
    pbar.set_description("testing")
    i = 1
    _type = 'unet'
    with torch.no_grad():
        for sample, mask, label in pbar:
            sample, mask, label = sample.to(device), mask.to(device), label.to(device)

            # pred, pred_mask = model(sample)

            # print()
            # pred_class = torch.round(torch.sigmoid(pred))
            # print(pred_class, label)
            # print()
            if label == 1:
                pred, pred_mask = model(sample)
                print(i, dice_metric(pred_mask, mask), intersection_over_union_metric(pred_mask, mask))
                save_nifti_image(os.path.join(f'samples/{_type}/image/image_{i:03d}.nii'), sample, image_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/mask/mask_{i:03d}.nii'), mask, mask_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_{i:03d}.nii'), torch.sigmoid(pred_mask), mask_affine)
                i += 1
                if i > 10:
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
    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]

    num_workers = config_dict["num_workers"]
    model_name = config_dict["model_name"]
    checkpoint_name = config_dict["checkpoint_name"]

    unet_embed_dims = list(config_dict["unet_embed_dims"])

    focal_patch_size = config_dict["focal_patch_size"]
    focal_embed_dims = config_dict["focal_embed_dims"]
    focal_depths = list(config_dict["focal_depths"])
    focal_levels = list(config_dict["focal_levels"])
    focal_windows = list(config_dict["focal_windows"])

    data_path = config_dict["data_path"]
    extra_path = config_dict["extra_path"]

    with open(os.path.join(extra_path, "data_split/test_sub_ses.pth"), "rb") as f:
        test_sub_ses = pickle.load(f)

    # if model_name == "weakfocal":
        # model = WeakFocalNet3D(img_size, focal_patch_size, in_ch, num_classes, focal_embed_dims, focal_depths, focal_levels, focal_windows)
    # model = MultiTaskUNet3D(in_ch, num_classes, unet_embed_dims)
    # model = MultitaskFocalUnet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes,
    #                             embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=True)
    model = UNet(in_ch, num_classes, unet_embed_dims)
    # model = MultitaskFocalUnet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes,
    #                            embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=True)

    checkpoint_path = os.path.join(extra_path, f"weights/{checkpoint_name}")
    test_ds = AneurysmDataset(data_path, test_sub_ses, shuffle=False)
    image_affine = test_ds.image_affine
    mask_affine = test_ds.mask_affine

    if setup == "classification":
        test_loader = DataLoader(test_ds, batch_size=16)
        test(model, checkpoint_path, test_loader)
    else:
        test_loader = DataLoader(test_ds, batch_size=1)
        test_segmentation(model, checkpoint_path, test_loader)


image_affine = None
mask_affine = None

if __name__ == "__main__":
    main()
