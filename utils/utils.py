import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn import metrics
from skimage.filters import threshold_otsu
from monai.metrics.utils import get_mask_edges, get_surface_distance
from torchvision.transforms import transforms
from monai.transforms import Compose, RandAffineD, RandFlipD, RandGaussianNoiseD
from numpy import deg2rad
import random
import nibabel as nib


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

        self.dices = []
        self.IoUs = []
        self.hausdorff_distances = []

    def add_prediction(self, pred, gt):
        pred = torch.round(torch.sigmoid(pred))
        self.predictions.extend(list(pred.detach()))
        self.ground_truth.extend(list(gt.detach()))

    def add_number_of_samples(self, new_samples):
        self.number_of_samples += new_samples

    def add_loss(self, loss):
        self.losses.append(loss.item())

    def get_mean_loss(self):
        return sum(self.losses) / self.number_of_samples

    def add_dice(self, dice):
        self.dices.extend(list(dice))

    def add_iou(self, iou):
        self.IoUs.extend(list(iou))

    def add_hausdorff_distance(self, hd):
        self.hausdorff_distances.extend(list(hd))

    def get_mean_dice(self):
        return torch.nanmean(torch.tensor(self.dices))

    def get_mean_iou(self):
        return torch.nanmean(torch.tensor(self.IoUs))

    def get_mean_hausdorff_distance(self):
        return torch.nanmean(torch.tensor(self.hausdorff_distances))

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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, gt):
        sigmoid = torch.sigmoid(pred)

        bce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        p_t = sigmoid * gt + (1 - sigmoid) * (1 - gt)
        focal_loss = bce * (1 - p_t) ** self.gamma

        if self.alpha > 0:
            alpha_t = self.alpha * gt + (1 - self.alpha) * (1 - gt)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'sum':
            reduced_loss = focal_loss.sum()
        else:
            reduced_loss = focal_loss.mean()

        return reduced_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class DiceSquareLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceSquareLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / ((inputs * inputs).sum() + targets.sum() + self.smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        return bce_loss + dice_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceFocalLoss, self).__init__()
        self.dice = DiceLoss(smooth)
        self.focal = FocalLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        focal_bce = self.focal(inputs, targets)
        return dice_loss + focal_bce


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
            RandAffineD(prob=0.5, rotate_range=(deg2rad(10), deg2rad(10), deg2rad(10)), scale_range=(0.1, 0.1, 0.1), keys=['image', 'mask']),
            RandFlipD(prob=0.5, keys=['image', 'mask']),
            RandGaussianNoiseD(prob=1, keys=['image']),
        ])

    def __call__(self, image, mask):
        x = {'image': image, 'mask': mask}
        x = self.augmentation(x)
        return x['image'], x['mask']


def save_nifti_image(file_path, image, affine):
    image = nib.Nifti1Image(image.squeeze().cpu().numpy(), affine)
    nib.save(image, file_path)


def dice_metric(predicted_mask, gt_mask):
    predicted_mask = torch.round(torch.sigmoid(predicted_mask)).detach()
    gt_mask = gt_mask.detach()

    predicted_mask = predicted_mask.flatten(1)
    gt_mask = gt_mask.flatten(1)

    intersection = torch.sum(predicted_mask * gt_mask, dim=1)
    gt_sum = torch.sum(gt_mask, dim=1)
    pred_sum = torch.sum(predicted_mask, dim=1)
    summation = pred_sum + gt_sum
    dice = torch.divide(2 * intersection, summation)
    dice[gt_sum == 0] = torch.nan
    return dice


def intersection_over_union_metric(predicted_mask, gt_mask):
    predicted_mask = torch.round(torch.sigmoid(predicted_mask)).detach()
    gt_mask = gt_mask.detach()

    predicted_mask = predicted_mask.flatten(1)
    gt_mask = gt_mask.flatten(1)

    intersection = torch.sum(predicted_mask * gt_mask, dim=1)
    gt_sum = torch.sum(gt_mask, dim=1)
    pred_sum = torch.sum(predicted_mask, dim=1)
    union = pred_sum + gt_sum - intersection
    iou = torch.divide(intersection, union)
    iou[gt_sum == 0] = torch.nan
    return iou
