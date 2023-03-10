import torch.optim
from utils.utils import *
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from utils.losses import *
from inference.inference import validation


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda'):
    model.to(device)
    model.train()
    scheduler = StepLR(optimizer, 200, 0.98)
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}
    for i, (sample, target_mask, dist_mask, target) in pbar_train:
        optimizer.zero_grad()
        sample, target_mask, dist_mask, target = sample.to(device), target_mask.to(device), dist_mask.to(device), target.to(device)
        pred_mask = model(sample)
        loss = loss_fn(pred_mask, target_mask, dist_mask)
        _metrics["train_cfm"].add_loss(loss)
        _metrics["train_cfm"].add_number_of_samples(len(target))

        pred_mask_onehot = pred_mask_to_onehot(pred_mask)
        _metrics["train_cfm"].add_dice(pred_mask_onehot, target_mask)
        _metrics["train_cfm"].add_iou(pred_mask_onehot, target_mask)
        _metrics["train_cfm"].add_hausdorff_distance(pred_mask_onehot, target_mask)

        loss.backward()
        optimizer.step()
        scheduler.step()

    validation(model, valid_loader, _metrics["valid_cfm"], loss_fn, device=device)

    return _metrics
