# utils/visualize.py
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image

from config import FIG_DIR


# 反归一化：把 Normalize 过的 tensor 变回正常图片
def _denormalize(image: torch.Tensor) -> torch.Tensor:
    """
    image: Tensor[C, H, W], 已经经过
        F.normalize(image, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(-1, 1, 1)
    img = image * std + mean            # 反归一化
    img = img.clamp(0.0, 1.0)           # 保证在 [0,1] 范围
    return img


def visualize_gt(image, target, class_names, save_name="gt_example.png"):
    """
    image: Tensor[C,H,W]  (已 Normalize)
    target: dict (boxes, labels)
    """
    # 先反归一化，再转成 PIL
    img_denorm = _denormalize(image)
    img = to_pil_image(img_denorm.cpu())

    boxes = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
        cls_name = class_names[int(label)]
        ax.text(
            xmin,
            ymin - 2,
            cls_name,
            fontsize=10,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5)
        )

    ax.set_axis_off()
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved GT image to {save_path}")


def visualize_predictions(image, preds, class_names,
                          score_thresh=0.5, save_name="pred_example.png"):
    """
    image: Tensor[C,H,W]  (已 Normalize)
    preds: dict (boxes, labels, scores)
    """
    # 同样先反归一化再显示
    img_denorm = _denormalize(image)
    img = to_pil_image(img_denorm.cpu())

    boxes = preds["boxes"].cpu()
    labels = preds["labels"].cpu()
    scores = preds["scores"].cpu()

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue

        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        cls_name = class_names[int(label)]
        ax.text(
            xmin,
            ymin - 2,
            f"{cls_name}:{score:.2f}",
            fontsize=10,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5)
        )

    ax.set_axis_off()
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved prediction image to {save_path}")
