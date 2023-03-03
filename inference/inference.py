import os
from utils.utils import *
from utils.dataset import *
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from torch.utils.data import DataLoader
# from monai.networks.blocks import CRF
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.transforms.post.array import one_hot
from models.crf import CRF
import cc3d


def validation(model, data_loader, cfm: ConfusionMatrix, loss_fn=None, device='cuda'):
    model.to(device)
    model.eval()
    # model = model[0]

    # crf_layer = CRF(iterations=5, bilateral_weight=3, gaussian_weight=1, bilateral_spatial_sigma=5,
    #                 bilateral_color_sigma=0.5, gaussian_spatial_sigma=5, update_factor=3)
    # cfm2 = ConfusionMatrix()

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

            pred_mask_onehot = pred_mask_to_onehot(pred_mask)
            cfm.add_dice(pred_mask_onehot, target_mask)
            cfm.add_iou(pred_mask_onehot, target_mask)
            cfm.add_hausdorff_distance(pred_mask_onehot, target_mask)

    #         sample = (sample - sample.min()) / (sample.max() - sample.min())
    #         crf_mask = crf_layer(pred_mask.cpu(), sample.cpu()).to(device)
    #         crf_mask_onehot = pred_mask_to_onehot(crf_mask)
    #         cfm.add_dice(crf_mask_onehot, target_mask)
    #         cfm.add_iou(crf_mask_onehot, target_mask)
    #         cfm.add_hausdorff_distance(crf_mask_onehot, target_mask)
    # print("crf:")
    # print_test_result(cfm2)


def monte_carlo_sampling(model, data_loader, cfm: ConfusionMatrix, device='cuda', n=10):
    model.to(device)
    model.eval()
    enable_dropout(model)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    pbar.set_description('validating monte-carlo')
    _type = 'unet'
    with torch.no_grad():
        for i, (sample, target_mask, dist_mask, target) in pbar:
            sample, target_mask, dist_mask, target = sample.to(device), target_mask.to(device), dist_mask.to(device), target.to(device)
            predictions = torch.empty(n, *target_mask.shape, device=device)
            for forward_passes in range(n):
                pred_mask = F.softmax(model(sample), dim=1)
                predictions[forward_passes] = pred_mask

            pred_mask = torch.mean(predictions, dim=0)
            pred_mask_onehot = pred_mask_to_onehot(pred_mask, softmax=False)

            cfm.add_dice(pred_mask_onehot, target_mask)
            cfm.add_iou(pred_mask_onehot, target_mask)
            cfm.add_hausdorff_distance(pred_mask_onehot, target_mask)
            variance_mask = torch.var(predictions, dim=0)

            save_nifti_image(os.path.join(f'samples/{_type}/image/image_{i:03d}.nii'), sample, image_affine)
            save_nifti_image(os.path.join(f'samples/{_type}/mask/mask_{i:03d}.nii'), target_mask, mask_affine)
            save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_{i:03d}.nii'), pred_mask_onehot, mask_affine)
            save_nifti_image(os.path.join(f'samples/{_type}/var/var_mask_fg_{i:03d}.nii'), variance_mask, mask_affine)
            for j in range(n):
                save_nifti_image(os.path.join(f'samples/{_type}/var/pred_fg_{i:03d}_{j}.nii'), predictions[j], mask_affine)
            if i > 1:
                return


def test_segmentation(model, test_loader, device='cuda'):
    global image_affine, mask_affine
    model.to(device)
    model.eval()
    model = model[0]

    pbar = tqdm(test_loader, total=len(test_loader))
    pbar.set_description("testing")
    i = 1
    _type = 'unet'
    crf_layer = CRF(iterations=5, bilateral_weight=3, gaussian_weight=1, bilateral_spatial_sigma=5,
                    bilateral_color_sigma=0.5, gaussian_spatial_sigma=5, update_factor=3)
    crf_layer.eval()
    cfm = ConfusionMatrix()
    # cfm2 = ConfusionMatrix()
    with torch.no_grad():
        for sample, target_mask, dist_mask, target in pbar:
            sample, target_mask, target = sample.to(device), target_mask.to(device), target.to(device)
            if target == 1:
                pred_mask = model(sample)
                sample = (sample - sample.min()) / (sample.max() - sample.min())
                # crf_mask, b = crf_layer(pred_mask.cpu(), sample.cpu())
                pred_mask_onehot = pred_mask_to_onehot(pred_mask)
                labels_out, N = cc3d.connected_components(pred_mask_onehot[0, 1].cpu().numpy(), return_N=True)

                # crf_mask_onehot = pred_mask_to_onehot(crf_mask)
                cfm.add_dice(pred_mask_onehot, target_mask)
                # cfm2.add_dice(crf_mask_onehot, target_mask)
                # print(len(b), b[0].shape)
                if N > 1:
                    save_nifti_image(os.path.join(f'samples/{_type}/image/image_{i:03d}.nii'), sample, image_affine)
                    save_nifti_image(os.path.join(f'samples/{_type}/mask/mask_{i:03d}.nii'), target_mask, mask_affine)
                    save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_{i:03d}.nii'), pred_mask_onehot, mask_affine)
                    crf_mask = crf_layer(pred_mask.cpu(), sample.cpu())
                    crf_mask_onehot = pred_mask_to_onehot(crf_mask)
                    labels_out2, N2, = cc3d.connected_components(crf_mask_onehot[0, 1].cpu().numpy(), return_N=True)
                    print(N, N2)
                    # for segid in range(1, N + 1):
                    #     extracted = labels_out * (labels_out == segid)
                    #     import numpy as np
                    #     print(i, segid, np.count_nonzero(extracted))
                    #
                    #     image = nib.Nifti1Image(extracted, mask_affine)
                    #     nib.save(image, os.path.join(f'samples/{_type}/pred/cc_{i:03d}_{segid}.nii'))

                # save_nifti_image(os.path.join(f'samples/{_type}/pred/crf_mask_{i:03d}.nii'), crf_mask_onehot, mask_affine)
                # save_nifti_image(os.path.join(f'samples/{_type}/pred/crf_mask_{i:03d}.nii'), crf_mask_onehot, mask_affine)
                i += 1
                if i > 100:
                    break
    print(cfm.get_mean_dice())
    # print(cfm2.dice.get_buffer(), cfm2.get_mean_dice())


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
    model = nn.Sequential(model,
                          nn.Softmax(dim=1))

    checkpoint_path = os.path.join(extra_path, f"weights/{checkpoint_name}")
    load_model(model, checkpoint_path)
    test_ds = AneurysmDataset(data_path, test_sub_ses, shuffle=False, return_dist_map=False)
    image_affine = test_ds.image_affine
    mask_affine = test_ds.mask_affine

    if setup == "classification":
        test_loader = DataLoader(test_ds, batch_size=1)
        test_cfm = ConfusionMatrix()
        validation(model, test_loader, test_cfm)
        print_test_result(test_cfm)
    elif setup == "segmentation":
        test_loader = DataLoader(test_ds, batch_size=1)
        test_segmentation(model, test_loader)
    else:
        test_loader = DataLoader(test_ds, batch_size=1)
        test_cfm = ConfusionMatrix()
        monte_carlo_sampling(model, test_loader, test_cfm)
        print_test_result(test_cfm)


image_affine = None
mask_affine = None

if __name__ == "__main__":
    # main()
    from torch.utils.cpp_extension import CUDA_HOME
    import os

    print(os.environ.get('CUDA_PATH'))
    print(CUDA_HOME)
    # x = CRF()
    # for m in x.modules():
    #     print(m)
    # from monai.networks.blocks import crf
    # from monai.networks.layers.filtering import PHLFilter
    # from monai.networks.blocks import CRF
    #
    # a = CRF()
    # x = torch.randn(1, 2, 64, 64, 64)
    # y = torch.randn(1, 1, 64, 64, 64)
    # # y = binary_to_onehot(y).unsqueeze(0)
    # z = a(x, y)
    # print(z.shape)
    #
    # c = SpatialFilter(y, 2)
    # out = c.apply(x)
    # print(out.shape)


