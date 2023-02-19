import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn import metrics
from skimage.filters import threshold_otsu
from monai.metrics.utils import get_mask_edges, get_surface_distance
from torchvision.transforms import transforms
from monai.transforms import Compose, RandAffine, RandFlip, RandGaussianNoise
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
        self.predictions.extend(list(pred))
        self.ground_truth.extend(list(gt))
        self.add_number_of_samples(len(gt))

    def add_number_of_samples(self, new_samples):
        self.number_of_samples += new_samples

    def add_loss(self, loss):
        self.losses.append(loss)

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

        pred_out = torch.round(self.predictions)
        self.true_positives = torch.sum(pred_out * self.ground_truth, dim=0)
        self.true_negatives = torch.sum((1 - pred_out) * (1 - self.ground_truth), dim=0)
        self.false_positives = torch.sum(pred_out * (1 - self.ground_truth), dim=0)
        self.false_negatives = torch.sum((1 - pred_out) * self.ground_truth, dim=0)

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

    def get_auc_score(self):
        scores = []
        for i in range(self.ground_truth.shape[-1]):
            scores.append(metrics.roc_auc_score(self.ground_truth[:, 0, i].cpu().numpy(),
                                                self.predictions[:, 0, i].cpu().numpy()))
        return np.array(scores)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
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


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, inputs, targets, smooth=1):
        dice_loss = self.dice(inputs, targets, smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


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


class FocalBCELoss:
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def __call__(self, pred, gt):
        sigmoid = torch.sigmoid(pred)

        bce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        p_t = sigmoid * gt + (1 - sigmoid) * (1 - gt)
        focal_loss = bce * (1 - p_t) ** self.gamma

        alpha_t = self.alpha * gt + (1 - self.alpha) * (1 - gt)
        focal_loss = alpha_t * focal_loss

        if self.reduction == 'sum':
            reduced_loss = focal_loss.sum()
        else:
            reduced_loss = focal_loss.mean()

        return reduced_loss


class Augmentation:
    def __init__(self):
        self.displacement = Compose([RandAffine(prob=0.5, rotate_range=(deg2rad(10), deg2rad(10), deg2rad(10)), scale_range=(0.1, 0.1, 0.1)),
                                     RandFlip(prob=0.5, spatial_axis=0)])

        self.color_change = Compose([RandGaussianNoise(prob=0.5, std=30)])

        self.augmentation = Compose([
            self.displacement,
            self.color_change
        ])

    def __call__(self, image, mask=None):
        seed = self.augmentation.R.randint(0, 123456789)  # make a seed with numpy generator
        self.augmentation.set_random_state(seed=seed)
        image = self.augmentation(image)

        if mask is not None:
            self.displacement.set_random_state(seed=seed)
            mask = self.displacement(mask.unsqueeze(0)).squeeze(0)  # to apply a channel and remove it we use unsqueeze and squeeze
            return image, mask
        else:
            return image


def save_nifti_image(file_path, image, affine):
    image = nib.Nifti1Image(image.squeeze().cpu().numpy(), affine)
    nib.save(image, file_path)
