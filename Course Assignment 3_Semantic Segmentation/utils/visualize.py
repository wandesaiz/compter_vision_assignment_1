# utils/visualize.py
import os
from typing import Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import FIG_DIR


def _denormalize(image: torch.Tensor) -> np.ndarray:
    """
    image: Tensor[C,H,W]，已 normalize
    return: np.ndarray[H,W,3]，范围 0~1
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = image.cpu().numpy()
    img = (img * std + mean).clip(0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img


def voc_palette() -> np.ndarray:
    """VOC 官方 21 类调色板"""
    return np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ], dtype=np.uint8)


def decode_segmap(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    mask: [H,W] int (0~20 或 255)
    return: [H,W,3] uint8
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(palette)):
        color[mask == i] = palette[i]
    return color


def visualize_segmentation(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_name: str,
    class_names: Optional[List[str]] = None,
):
    """
    保存 Image / GT / Pred 三联图
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    img = _denormalize(image)
    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()

    # palette = voc_palette()
    palette = cityscapes_palette()
    gt_color = decode_segmap(gt, palette)
    pred_color = decode_segmap(pred, palette)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(gt_color)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_color)
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved to {save_path}")

def cityscapes_palette() -> np.ndarray:
    """Cityscapes 19 类调色板（trainId 0-18）"""
    return np.array([
        [128, 64,128],  # road
        [244, 35,232],  # sidewalk
        [ 70, 70, 70],  # building
        [102,102,156],  # wall
        [190,153,153],  # fence
        [153,153,153],  # pole
        [250,170, 30],  # traffic light
        [220,220,  0],  # traffic sign
        [107,142, 35],  # vegetation
        [152,251,152],  # terrain
        [ 70,130,180],  # sky
        [220, 20, 60],  # person
        [255,  0,  0],  # rider
        [  0,  0,142],  # car
        [  0,  0, 70],  # truck
        [  0, 60,100],  # bus
        [  0, 80,100],  # train
        [  0,  0,230],  # motorcycle
        [119, 11, 32],  # bicycle
    ], dtype=np.uint8)