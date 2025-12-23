# utils/metrics.py
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from config import NUM_CLASSES, IGNORE_INDEX



@torch.no_grad()
def evaluate_loss_only(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=255,
        )
        total_loss += loss.item()

    return total_loss / max(1, num_batches)



def _compute_confusion_matrix(pred: torch.Tensor,
                              label: torch.Tensor,
                              num_classes: int,
                              ignore_index: int) -> torch.Tensor:
    """
    pred, label: [N, H, W]
    """
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]

    k = (label >= 0) & (label < num_classes)
    pred = pred[k]
    label = label[k]

    conf_mat = torch.bincount(
        num_classes * label + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return conf_mat


def _compute_miou_from_confmat(conf_mat: torch.Tensor) -> Tuple[float, torch.Tensor]:
    tp = torch.diag(conf_mat)
    pos_gt = conf_mat.sum(dim=1)
    pos_pred = conf_mat.sum(dim=0)
    union = pos_gt + pos_pred - tp
    iou = tp.float() / (union.float() + 1e-6)
    miou = iou.mean().item()
    return miou, iou


@torch.no_grad()
def evaluate_miou(model, dataloader, device):
    num_classes = NUM_CLASSES
    """
    在验证集上计算 mIoU & per-class IoU
    """
    model.eval()
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # for batch in tqdm(data_loader, desc="[Val mIoU]", leave=False):
    #     images = batch["pixel_values"].to(device)
    #     labels = batch["labels"].to(device)
    #
    #     # outputs = model(pixel_values=images)
    #     # logits = outputs.logits        # [B,C,H,W]
    #     outputs = model(pixel_values=images)
    #     logits = outputs.logits
    #     logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    #
    #     loss = F.cross_entropy(logits, labels, ignore_index=255)
    #     preds = torch.argmax(logits, dim=1)  # [B,H,W]
    #
    #     conf_mat += _compute_confusion_matrix(preds.cpu(), labels.cpu(),
    #                                           num_classes, ignore_index)
    for batch in dataloader:
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        preds = logits.argmax(dim=1)  # [B,H,W]

        conf_mat += _compute_confusion_matrix(
            preds.cpu(), labels.cpu(), num_classes=num_classes, ignore_index=IGNORE_INDEX
        )

    miou, per_class_iou = _compute_miou_from_confmat(conf_mat)
    return miou, per_class_iou

