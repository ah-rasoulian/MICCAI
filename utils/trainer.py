import torch.optim
from utils.utils import *
from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', train_type='classification'):
    assert train_type in ['classification', 'segmentation', 'multitask'], "train type not suppoerted"
    if train_type == 'multitask':
        assert type(loss_fn) is tuple
        alpha, classification_loss, segmentation_loss = loss_fn
        assert 0 <= alpha <= 1
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}
    for i, (sample, target_mask, target) in pbar_train:
        optimizer.zero_grad()
        sample, target_mask, target = sample.to(device), target_mask.to(device), target.to(device)

        pred = model(sample)
        if train_type == 'multitask':
            pred, pred_mask = pred
            loss = alpha * classification_loss(pred, target) + (1 - alpha) * segmentation_loss(pred_mask, target_mask)
        elif train_type == 'segmentation':
            loss = loss_fn(pred, target_mask)
        else:
            loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        if train_type == 'multitask':
            _metrics["train_cfm"].add_prediction(pred, target)
            _metrics["train_cfm"].add_dice(dice_metric(pred_mask, target_mask))
        elif train_type == 'segmentation':
            _metrics["train_cfm"].add_dice(dice_metric(pred_mask, target_mask))
        else:
            _metrics["train_cfm"].add_prediction(pred, target)

    if train_type in ['multitask', 'classification']:
        _metrics["train_cfm"].compute_confusion_matrix()

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')
    with torch.no_grad():
        for i, (sample, target) in pbar_valid:
            sample, target_mask, target = sample.to(device), target_mask.to(device), target.to(device)

            pred = model(sample)
            if train_type == 'multitask':
                pred, pred_mask = pred
                loss = alpha * classification_loss(pred, target) + (1 - alpha) * segmentation_loss(pred_mask, target_mask)
            elif train_type == 'segmentation':
                loss = loss_fn(pred, target_mask)
            else:
                loss = loss_fn(pred, target)

            _metrics["valid_cfm"].add_loss(loss.item())
            if train_type == 'multitask':
                _metrics["valid_cfm"].add_prediction(pred, target)
                _metrics["valid_cfm"].add_dice(dice_metric(pred_mask, target_mask))
            elif train_type == 'segmentation':
                _metrics["valid_cfm"].add_dice(dice_metric(pred_mask, target_mask))
            else:
                _metrics["valid_cfm"].add_prediction(pred, target)

        if train_type in ['multitask', 'classification']:
            _metrics["valid_cfm"].compute_confusion_matrix()

    return _metrics
