import os

import numpy as np

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
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.transforms.post.array import one_hot
import cc3d
from models.crf import CRF
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing, binary_fill_holes
import time


def validation(model, data_loader, cfm: ConfusionMatrix, loss_fn=None, device='cuda'):
    model.to(device)
    model.eval()

    # crf_layer = CRF(num_iters=-1, num_classes=2, bilateral_weight=10, gaussian_weight=2, gaussian_spatial_sigma=16,
    #                 bilateral_spatial_sigma=64, bilateral_intensity_sigma=4)
    #
    # cfm2 = ConfusionMatrix()

    pbar_valid = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    pbar_valid.set_description('validating')
    with torch.no_grad():
        for i, (sample, target_mask, dist_mask, target) in pbar_valid:
            # if target == 0:
            #     continue
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
    #         print()
    #
    # sample = (sample - sample.min()) / (sample.max() - sample.min())
    # pred_mask = F.softmax(pred_mask, dim=1)
    # pred_mask = pred_mask.cpu().numpy()
    # pred_mask[0, 1] = gaussian_filter(pred_mask[0, 1], 5)
    # crf_mask = crf_layer(pred_mask, sample).to(device)
    # # print(torch.norm(torch.log(pred_mask)), torch.norm(torch.log(crf_mask)))
    # crf_mask_onehot = pred_mask_to_onehot(crf_mask, softmax=False)
    # cfm2.add_dice(crf_mask_onehot, target_mask)
    # cfm2.add_iou(crf_mask_onehot, target_mask)
    # cfm2.add_hausdorff_distance(crf_mask_onehot, target_mask)
    # if i > 20:
    #     break
    #         print(compute_dice(pred_mask_onehot, target_mask, include_background=False), compute_dice(crf_mask_onehot, target_mask, include_background=False))
    #         print()
    # print("crf:")
    # print_test_result(cfm2)


def monte_carlo_sampling(model, data_loader, cfm: ConfusionMatrix, device='cuda', n=1):
    model.to(device)
    model.eval()
    # enable_dropout(model)
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    pbar.set_description('validating monte-carlo')
    _type = 'unet'
    #
    crf_layer = CRF(num_iters=1, num_classes=2, bilateral_weight=9, gaussian_weight=3, gaussian_spatial_sigma=18,
                    bilateral_spatial_sigma=80, bilateral_intensity_sigma=3)
    cfm2 = ConfusionMatrix()
    start = time.time()
    with torch.no_grad():
        for i, (sample, target_mask, dist_mask, target) in pbar:
            # if target == 0:
            #     continue
            sample, target_mask, dist_mask, target = sample.to(device), target_mask.to(device), dist_mask.to(device), target.to(device)
            predictions = []
            weights = []
            for forward_passes in range(n):
                pred_mask = F.softmax(model(sample), dim=1)
                # pred_mask = model(sample)
                weights.append(torch.ones_like(pred_mask) * torch.mean(torch.argmax(pred_mask, dim=1, keepdim=True) * sample))
                predictions.append(pred_mask)
            predictions = torch.stack(predictions, dim=0)
            weights = torch.stack(weights, dim=0)
            # predictions *= weights
            pred_mask = torch.mean(predictions, dim=0)
            variance_mask = torch.var(predictions, dim=0)
            # pred_mask = crf_layer(pred_mask, sample)
            # pred_mask /= variance_mask
            cfm.add_prediction(pred_mask, target_mask)
            # if i > 50:
            #     break
    print("time: ", time.time() - start)
    print(cfm.get_false_positives_rate())
    print(cfm.get_sensitivity_rate())
    print(torch.std_mean(cfm.dice.get_buffer()))
    print(torch.std_mean(cfm.iou.get_buffer()))
    print(torch.std_mean(cfm.hausdorff_distances.get_buffer()))

    print(np.nanmean(cfm.sensitivity_rate), np.nanstd(cfm.sensitivity_rate))
    print(np.nanmean(cfm.false_positive_rate), np.nanstd(cfm.false_positive_rate))


def test_segmentation(model, test_loader, device='cuda'):
    global image_affine, mask_affine
    model.to(device)
    model.eval()

    pbar = tqdm(test_loader, total=len(test_loader))
    pbar.set_description("testing")
    i = 1
    _type = 'unet'
    # crf_layer = CRF(iterations=5, bilateral_weight=3, gaussian_weight=1, bilateral_spatial_sigma=5,
    #                 bilateral_color_sigma=0.5, gaussian_spatial_sigma=5, update_factor=3)
    # crf_layer.eval()
    cfm = ConfusionMatrix()
    # cfm2 = ConfusionMatrix()
    with torch.no_grad():
        for sample, target_mask, dist_mask, target in pbar:
            sample, target_mask, target = sample.to(device), target_mask.to(device), target.to(device)
            if target == 1:
                pred_mask = model(sample)
                sample = (sample - sample.min()) / (sample.max() - sample.min())
                pred_mask_onehot = pred_mask_to_onehot(pred_mask)

                pred_mask = pred_mask.cpu()
                gaussian = torch.Tensor(gaussian_filter(pred_mask_onehot[0, 1].cpu(), sigma=10))
                print(gaussian.shape)
                save_nifti_image(os.path.join(f'samples/{_type}/image/image_{i:03d}.nii'), sample, image_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/mask/mask_{i:03d}.nii'), target_mask, mask_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_bin_{i:03d}.nii'), pred_mask_onehot, mask_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_{i:03d}.nii'), pred_mask, mask_affine)
                save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_gaussian_{i:03d}.nii'), gaussian, mask_affine)
                # labels_out, N = cc3d.connected_components(pred_mask_onehot[0, 1].cpu().numpy(), return_N=True)

                # crf_mask_onehot = pred_mask_to_onehot(crf_mask)
                # cfm.add_dice(pred_mask_onehot, target_mask)
                # cfm2.add_dice(crf_mask_onehot, target_mask)
                # print(len(b), b[0].shape)
                # if N > 1:
                #     save_nifti_image(os.path.join(f'samples/{_type}/image/image_{i:03d}.nii'), sample, image_affine)
                #     save_nifti_image(os.path.join(f'samples/{_type}/mask/mask_{i:03d}.nii'), target_mask, mask_affine)
                #     save_nifti_image(os.path.join(f'samples/{_type}/pred/pred_mask_{i:03d}.nii'), pred_mask_onehot, mask_affine)
                # crf_mask = crf_layer(pred_mask.cpu(), sample.cpu())
                # crf_mask_onehot = pred_mask_to_onehot(crf_mask)
                # labels_out2, N2, = cc3d.connected_components(crf_mask_onehot[0, 1].cpu().numpy(), return_N=True)
                # print(N, N2)
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
                if i > 10:
                    break
    # print(cfm.get_mean_dice())
    # print(cfm2.dice.get_buffer(), cfm2.get_mean_dice())


def visualization():
    unet_model = UNet(1, 2, [24, 48, 96, 192])
    swin_unetr_model = SwinUNETR(img_size=to_3tuple(64), in_channels=1, out_channels=2, feature_size=24)
    focal_model = FocalConvUNet(img_size=64, patch_size=2, in_chans=1, num_classes=2, embed_dim=24, depths=[2, 2, 2, 2], focal_levels=[3, 3, 3, 3], focal_windows=[3, 3, 3, 3], use_conv_embed=True)
    load_model(unet_model, 'extra/weights/backup/UNet.pt')
    load_model(swin_unetr_model, 'extra/weights/backup/SwinUNETR.pt')
    load_model(focal_model, 'extra/weights/backup/FocalConvUNet.pt')

    with open("extra/data_split/test_sub_ses.pth", "rb") as f:
        test_sub_ses = pickle.load(f)
    test_ds = AneurysmDataset("C:\\lausanne-aneurysym-patches\\Data_Set_Feb_05_2023_v1-all", test_sub_ses, shuffle=False, return_dist_map=False)
    test_loader = DataLoader(test_ds, batch_size=1)

    cfm_unet, cfm_unet_crf = ConfusionMatrix(), ConfusionMatrix()
    cfm_swin, cfm_swin_crf = ConfusionMatrix(), ConfusionMatrix()
    cfm_focal, cfm_focal_crf = ConfusionMatrix(), ConfusionMatrix()
    crf_layer = CRF(num_iters=1, num_classes=2, bilateral_weight=9, gaussian_weight=3, gaussian_spatial_sigma=18,
                    bilateral_spatial_sigma=80, bilateral_intensity_sigma=3)
    unet_model.cuda()
    swin_unetr_model.cuda()
    focal_model.cuda()
    for i, (sample, target_mask, dist_mask, target) in enumerate(test_loader):
        sample, target_mask, target = sample.to('cuda'), target_mask.to('cuda'), target.to('cuda')
        unet_out = unet_model(sample)
        cfm_unet.add_prediction(unet_out, target_mask)
        swin_out = swin_unetr_model(sample)
        cfm_swin.add_prediction(swin_out, target_mask)
        focal_out = focal_model(sample)
        cfm_focal.add_prediction(focal_out, target_mask)

        unet_crf = crf_layer(unet_out.cpu(), sample.cpu()).cuda()
        cfm_unet_crf.add_prediction(unet_crf, target_mask)
        swin_crf = crf_layer(swin_out.cpu(), sample.cpu()).cuda()
        cfm_swin_crf.add_prediction(swin_crf, target_mask)
        focal_crf = crf_layer(focal_out.cpu(), sample.cpu()).cuda()
        cfm_focal_crf.add_prediction(focal_crf, target_mask)

        save_nifti_image(f"samples/final/{i:03d}-image.nii", sample.cpu().detach(), test_ds.image_affine)
        save_nifti_image(f"samples/final/{i:03d}-mask.nii", target_mask.cpu().detach(), test_ds.mask_affine)
        save_nifti_image(f"samples/final/{i:03d}-unet.nii", unet_out.cpu().detach(), test_ds.mask_affine)
        save_nifti_image(f"samples/final/{i:03d}-swin.nii", swin_out.cpu().detach(), test_ds.mask_affine)
        save_nifti_image(f"samples/final/{i:03d}-focal.nii", focal_out.cpu().detach(), test_ds.mask_affine)
        save_nifti_image(f"samples/final/{i:03d}-unetcrf.nii", unet_crf.cpu().detach(), test_ds.mask_affine)
        save_nifti_image(f"samples/final/{i:03d}-swincrf.nii", swin_crf.cpu().detach(), test_ds.mask_affine)
        save_nifti_image(f"samples/final/{i:03d}-focalcrf.nii", focal_crf.cpu().detach(), test_ds.mask_affine)


        # print(i)
        # print(compute_dice(pred_mask_to_onehot(unet_out), target_mask, include_background=False))
        # print(compute_dice(pred_mask_to_onehot(swin_out), target_mask, include_background=False))
        # print(compute_dice(pred_mask_to_onehot(focal_out), target_mask, include_background=False))
        # print(compute_dice(pred_mask_to_onehot(unet_crf), target_mask, include_background=False))
        # print(compute_dice(pred_mask_to_onehot(swin_crf), target_mask, include_background=False))
        # print(compute_dice(pred_mask_to_onehot(focal_crf), target_mask, include_background=False))
        # print()
        # if i > 1:
        #     break

    np.savetxt("samples/seg_metrics/dice_unet.csv", cfm_unet.dice.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/iou_unet.csv", cfm_unet.iou.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/hd_unet.csv", cfm_unet.hausdorff_distances.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/fp_unet.csv", np.asarray(cfm_unet.false_positive_rate), delimiter=",")
    np.savetxt("samples/seg_metrics/sens_unet.csv", np.asarray(cfm_unet.sensitivity_rate), delimiter=",")

    np.savetxt("samples/seg_metrics/dice_unet_crf.csv", cfm_unet_crf.dice.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/iou_unet_crf.csv", cfm_unet_crf.iou.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/hd_unet_crf.csv", cfm_unet_crf.hausdorff_distances.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/fp_unet_crf.csv", np.asarray(cfm_unet_crf.false_positive_rate), delimiter=",")
    np.savetxt("samples/seg_metrics/sens_unet_crf.csv", np.asarray(cfm_unet_crf.sensitivity_rate), delimiter=",")

    np.savetxt("samples/seg_metrics/dice_swin.csv", cfm_swin.dice.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/iou_swin.csv", cfm_swin.iou.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/hd_swin.csv", cfm_swin.hausdorff_distances.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/fp_swin.csv", np.asarray(cfm_swin.false_positive_rate), delimiter=",")
    np.savetxt("samples/seg_metrics/sens_swin.csv", np.asarray(cfm_swin.sensitivity_rate), delimiter=",")

    np.savetxt("samples/seg_metrics/dice_swin_crf.csv", cfm_swin_crf.dice.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/iou_swin_crf.csv", cfm_swin_crf.iou.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/hd_swin_crf.csv", cfm_swin_crf.hausdorff_distances.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/fp_swin_crf.csv", np.asarray(cfm_swin_crf.false_positive_rate), delimiter=",")
    np.savetxt("samples/seg_metrics/sens_swin_crf.csv", np.asarray(cfm_swin_crf.sensitivity_rate), delimiter=",")

    np.savetxt("samples/seg_metrics/dice_focal.csv", cfm_focal.dice.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/iou_focal.csv", cfm_focal.iou.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/hd_focal.csv", cfm_focal.hausdorff_distances.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/fp_focal.csv", np.asarray(cfm_focal.false_positive_rate), delimiter=",")
    np.savetxt("samples/seg_metrics/sens_focal.csv", np.asarray(cfm_focal.sensitivity_rate), delimiter=",")

    np.savetxt("samples/seg_metrics/dice_focal_crf.csv", cfm_focal_crf.dice.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/iou_focal_crf.csv", cfm_focal_crf.iou.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/hd_focal_crf.csv", cfm_focal_crf.hausdorff_distances.get_buffer().numpy(), delimiter=",")
    np.savetxt("samples/seg_metrics/fp_focal_crf.csv", np.asarray(cfm_focal_crf.false_positive_rate), delimiter=",")
    np.savetxt("samples/seg_metrics/sens_focal_crf.csv", np.asarray(cfm_focal_crf.sensitivity_rate), delimiter=",")


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
    main()
    # visualization()
    # from utils.filters import _spatial_features
    # x = torch.randn(1, 2, 2)
    # y = _spatial_features(x, 1)
    # print(x)
    # print(y)
    # from torch.utils.cpp_extension import CUDA_HOME

    # print(CUDA_HOME)
    # x = CRF()
    # for m in x.modules():
    #     print(m)
    # from monai.networks.blocks import crf
    # from monai.networks.layers.filtering import PHLFilter
    # from monai.utils.module import optional_import
    #
    # _C, _ = optional_import("monai._C")
    # print(_)
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
