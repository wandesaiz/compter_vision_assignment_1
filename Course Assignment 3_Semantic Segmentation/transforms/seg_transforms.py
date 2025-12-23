# transforms/seg_transforms.py
import random
from typing import Tuple

import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

from config import IMAGE_SIZE


def _get_hw(size) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"IMAGE_SIZE must be int or (H, W), got {size}")


class SegmentationTransform:
    """
    稳定版：
    - train: random hflip + resize + (可选) 轻微 color jitter
    - val:   仅 resize
    """

    def __init__(self, image_size=IMAGE_SIZE, is_train: bool = True):
        self.image_size = _get_hw(image_size)   # (H, W)
        self.is_train = is_train

        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    def _random_flip(self, img: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

    def _resize(self, img: Image.Image, mask: Image.Image):
        Ht, Wt = self.image_size
        # PIL 的 size 是 (W, H)
        img  = img.resize((Wt, Ht), Image.BILINEAR)
        mask = mask.resize((Wt, Ht), Image.NEAREST)
        return img, mask

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.is_train:
            img, mask = self._random_flip(img, mask)
            img, mask = self._resize(img, mask)
            # 你要是想更稳一点，可以先把这行注释掉
            img = self.color_jitter(img)
        else:
            img, mask = self._resize(img, mask)

        img = F.to_tensor(img)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, mask
