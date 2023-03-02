import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from monai.transforms import Compose, RandAffineD, RandFlipD, RandGaussianNoiseD
from numpy import deg2rad
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from utils.losses import *
import matplotlib.pyplot as plt
from models.unet import UNet
from models.focalunet import FocalUNet
from models.focalconvunet import FocalConvUNet
from monai.networks.nets import SwinUNETR
from timm.models.layers import to_3tuple

from monai.transforms.post.array import one_hot
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric


class ConfusionMatrix:
    def __init__(self, device='cuda'):
        self.device = device
        self.predictions = []
        self.ground_truth = []

        self.losses = []
        self.number_of_samples = 0

        self.true_positives = None
        self.true_negatives = None
        self.false_positives = None
        self.false_negatives = None

        self.dice = DiceMetric(include_background=False, reduction="none")
        self.iou = MeanIoU(include_background=False, reduction="none")
        self.hausdorff_distances = HausdorffDistanceMetric(include_background=False, percentile=95, directed=True, reduction="none")

    def add_prediction(self, pred_mask, target_mask):
        pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True).detach
        target_mask = target_mask.detach()

        self.predictions.extend(list(pred_mask.detach()))
        self.ground_truth.extend(list(target_mask.detach()))

    def add_number_of_samples(self, new_samples):
        self.number_of_samples += new_samples

    def add_loss(self, loss):
        self.losses.append(loss.item())

    def get_mean_loss(self):
        return sum(self.losses) / self.number_of_samples

    def add_dice(self, pred_mask, target_mask):
        self.dice(pred_mask, target_mask)

    def add_iou(self, pred_mask, target_mask):
        self.iou(pred_mask, target_mask)

    def add_hausdorff_distance(self, pred_mask, target_mask):
        self.hausdorff_distances(pred_mask, target_mask)

    def get_mean_dice(self):
        return self.dice.aggregate("mean").item()

    def get_mean_iou(self):
        return self.iou.aggregate("mean").item()

    def get_mean_hausdorff_distance(self):
        return self.hausdorff_distances.aggregate("mean").item()

    def compute_confusion_matrix(self):
        self.predictions = torch.stack(self.predictions).to(self.device)
        self.ground_truth = torch.stack(self.ground_truth).to(self.device)

        self.true_positives = torch.sum(self.predictions * self.ground_truth, dim=0)
        self.true_negatives = torch.sum((1 - self.predictions) * (1 - self.ground_truth), dim=0)
        self.false_positives = torch.sum(self.predictions * (1 - self.ground_truth), dim=0)
        self.false_negatives = torch.sum((1 - self.predictions) * self.ground_truth, dim=0)

    def get_accuracy(self):
        numerator = self.true_positives + self.true_negatives
        denominator = numerator + self.false_positives + self.false_negatives

        return torch.divide(numerator, denominator)

    def get_precision(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives

        return torch.divide(numerator, denominator)

    def get_recall_sensitivity(self):
        numerator = self.true_positives
        denominator = self.true_positives + self.false_negatives

        return torch.divide(numerator, denominator)

    def get_specificity(self):
        numerator = self.true_negatives
        denominator = self.true_negatives + self.false_positives

        return torch.divide(numerator, denominator)

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall_sensitivity()
        numerator = 2 * precision * recall
        denominator = precision + recall

        return torch.divide(numerator, denominator)

    # todo: fix it
    def get_auc_score(self):
        scores = []
        for i in range(self.ground_truth.shape[-1]):
            scores.append(metrics.roc_auc_score(self.ground_truth[:, 0, i].cpu().numpy(),
                                                self.predictions[:, 0, i].cpu().numpy()))
        return np.array(scores)


class EarlyStopping:
    def __init__(self, model: nn.Module, patience: int, path_to_save: str, gamma=0):
        assert patience > 0, 'patience must be positive'
        self.model = model
        self.patience = patience
        self.path_to_save = path_to_save
        self.gamma = gamma

        self.min_loss = np.Inf
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss + self.gamma < self.min_loss:
            print("val loss decreased from {} to {}".format(self.min_loss, val_loss))
            self.min_loss = val_loss
            self.counter = 0
            save_model(self.model, self.path_to_save)
        else:
            self.counter += 1
            if self.counter == self.patience:
                print("early stop")
                self.early_stop = True
            else:
                print("early stopping counter: {} of {}".format(self.counter, self.patience))


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path))


class Augmentation:
    def __init__(self):
        self.augmentation = Compose([
            RandAffineD(prob=0.5, rotate_range=(deg2rad(15), deg2rad(15), deg2rad(15)), scale_range=(0.1, 0.1, 0.1), keys=['image', 'mask']),
            RandFlipD(prob=0.5, keys=['image', 'mask']),
            RandGaussianNoiseD(prob=0.5, std=1, keys=['image']),
        ])

    def __call__(self, image, mask):
        x = {'image': image, 'mask': mask}
        x = self.augmentation(x)
        return x['image'], x['mask']


def save_nifti_image(file_path, image, affine):
    assert image.dim() == 5 and image.shape[0] == 1, "expected 1 in batch format"  # 1, c, d, h, w
    if image.shape[1] > 1:
        image = image[:, 1].type(torch.int8)  # getting the mask for foreground
    image = nib.Nifti1Image(image.squeeze().cpu().numpy(), affine)
    nib.save(image, file_path)


def pred_mask_to_onehot(pred_mask):
    pred_mask = F.softmax(pred_mask, dim=1)
    pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    return one_hot(pred_mask, num_classes=2, dim=1)


def onehot_to_dist(onehot: torch.Tensor, dtype=torch.int32):
    # inspired by https://github.com/LIVIAETS/boundary-loss/
    # all rights reserved to HKervadec
    number_of_classes = onehot.shape[0]
    result = torch.zeros_like(onehot, dtype=dtype)
    for k in range(number_of_classes):
        pos_mask = onehot[k]
        if pos_mask.any():
            neg_mask = 1 - pos_mask
            result[k] = torch.from_numpy(distance_transform_edt(neg_mask)) * neg_mask - (torch.from_numpy(distance_transform_edt(pos_mask)) - 1) * pos_mask
    return result.type(torch.FloatTensor)


class CEDiceBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.ce_loss = CrossEntropy(idc=[0, 1])
        self.dice_loss = GeneralizedDice(idc=[0, 1])
        self.boundary_loss = BoundaryLoss(idc=[1])
        self.alpha = alpha

    def forward(self, pred_mask, target_mask, dist_mask):
        pred_mask = F.softmax(pred_mask, dim=1)
        return self.ce_loss(pred_mask, target_mask) + self.dice_loss(pred_mask, target_mask) + self.alpha * self.boundary_loss(pred_mask, dist_mask)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def visualize_losses(train_losses, valid_losses):
    fig, axes = plt.subplots(2)
    axes[0].plot(train_losses)
    axes[1].plot(valid_losses)
    plt.show()


def str_to_bool(string):
    return True if string == "True" else False


def print_test_result(test_cfm: ConfusionMatrix):
    print(f"\ntest result:"
          f"dice={test_cfm.get_mean_dice()}, iou={test_cfm.get_mean_iou()}, hd={test_cfm.get_mean_hausdorff_distance()}\n")


def build_model(config_dict):
    model_name = config_dict["model"]
    assert model_name in ["unet", "focalconvunet", "focalunet", "swinunetr"]
    img_size = config_dict["img_size"]
    in_ch = config_dict["in_ch"]
    num_classes = config_dict["num_classes"]
    unet_embed_dims = list(config_dict["unet_embed_dims"])

    focal_patch_size = config_dict["focal_patch_size"]
    focal_embed_dims = config_dict["focal_embed_dims"]
    focal_depths = list(config_dict["focal_depths"])
    focal_levels = list(config_dict["focal_levels"])
    focal_windows = list(config_dict["focal_windows"])
    if model_name == "unet":
        model = UNet(in_ch, num_classes, unet_embed_dims, drop_rate=0.5)
    elif model_name == 'swinunetr':
        model = SwinUNETR(img_size=to_3tuple(img_size), in_channels=in_ch, out_channels=num_classes, feature_size=24, drop_rate=0.5)
    elif model_name == "focalconvunet":
        model = FocalConvUNet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes, drop_rate=0.5,
                              embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=True)
    else:
        model = FocalUNet(img_size=img_size, patch_size=focal_patch_size, in_chans=in_ch, num_classes=num_classes, drop_rate=0.5,
                          embed_dim=focal_embed_dims, depths=focal_depths, focal_levels=focal_levels, focal_windows=focal_windows, use_conv_embed=True)

    return model
