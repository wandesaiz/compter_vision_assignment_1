# utils/metrics.py
import torch
import numpy as np
from torchvision.ops import box_iou
from tqdm import tqdm


@torch.no_grad()
def evaluate_loss_only(model, data_loader, device):
    """
    只计算验证集平均 loss，用来画训练 / 验证 loss 曲线。
    注意：Faster R-CNN 只有在 train 模式下才会返回 loss dict。
    """
    # 记录原来的模式（train / eval）
    was_training = model.training

    # 为了拿到 loss dict，这里强制切到 train 模式
    model.train()

    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # 理论上这里一定是 dict，如果是 list 说明又哪里写错了
        if isinstance(loss_dict, list):
            raise TypeError(
                f"Expected loss dict during validation, but got list (len={len(loss_dict)}). "
                "检查一下模型 forward 和模式切换。"
            )

        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    # 恢复原来的模式
    model.train(was_training)

    return {"val_loss": avg_loss}


@torch.no_grad()
def evaluate_map(model, data_loader, device, iou_thresh: float = 0.5):
    """
    VOC2007 风格 mAP 计算 —— 修复版
    不再使用 DataLoader 打开文件，从而解决 Too many open files 问题。
    """
    was_training = model.training
    model.eval()

    dataset = data_loader.dataset  # 直接使用 dataset
    num_items = len(dataset)

    gt_by_image = {}
    preds = []

    # ★ 不再依赖 DataLoader，避免文件句柄泄漏
    for img_id in tqdm(range(num_items), desc="Evaluating mAP"):
        image, target = dataset[img_id]

        # 保证 image 是 tensor
        image = image.to(device)
        outputs = model([image])[0]

        # GT
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        gt_by_image[img_id] = {
            "boxes": gt_boxes,
            "labels": gt_labels,
        }

        # Pred
        for b, l, s in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            preds.append({
                "image_id": img_id,
                "box": b.cpu(),
                "label": int(l),
                "score": float(s)
            })

        # ★ 主动清理，避免句柄累积
        del image, target, outputs
        torch.cuda.empty_cache()

    # 类别信息
    if hasattr(dataset, "CLASSES"):
        class_names = ["__background__"] + list(dataset.CLASSES)
        num_classes = len(class_names)
    else:
        num_classes = int(max((p["label"] for p in preds), default=0)) + 1
        class_names = [str(i) for i in range(num_classes)]

    ap_per_class = {}
    mAP_sum, valid_cls = 0.0, 0

    # 从 1 开始
    for cls_id in range(1, num_classes):
        cls_name = class_names[cls_id]

        # GT
        gt_for_cls = {}
        npos = 0
        for img_id, gt in gt_by_image.items():
            mask = gt["labels"] == cls_id
            boxes = gt["boxes"][mask]
            if boxes.numel() == 0:
                continue
            gt_for_cls[img_id] = {
                "boxes": boxes,
                "detected": [False] * boxes.size(0),
            }
            npos += boxes.size(0)

        if npos == 0:
            ap_per_class[cls_name] = 0.0
            continue

        # Pred
        cls_preds = [p for p in preds if p["label"] == cls_id]
        if not cls_preds:
            ap_per_class[cls_name] = 0.0
            continue

        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        tps, fps = [], []
        for p in cls_preds:
            img_id = p["image_id"]
            box = p["box"].unsqueeze(0)

            if img_id not in gt_for_cls:
                tps.append(0); fps.append(1)
                continue

            gt_info = gt_for_cls[img_id]
            ious = box_iou(box, gt_info["boxes"])[0]
            max_iou, max_idx = ious.max(dim=0)

            if max_iou >= iou_thresh and not gt_info["detected"][max_idx]:
                tps.append(1); fps.append(0)
                gt_info["detected"][max_idx] = True
            else:
                tps.append(0); fps.append(1)

        tps, fps = np.array(tps), np.array(fps)
        tp_cum, fp_cum = np.cumsum(tps), np.cumsum(fps)

        recalls = tp_cum / (npos + 1e-6)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recalls >= t):
                ap += np.max(precisions[recalls >= t]) / 11.0

        ap_per_class[cls_name] = ap
        mAP_sum += ap
        valid_cls += 1

    mAP = mAP_sum / max(1, valid_cls)

    model.train(was_training)

    return {"mAP": float(mAP), "AP_per_class": ap_per_class}

