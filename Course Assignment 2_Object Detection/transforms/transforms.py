# transforms/transforms.py
import random
from typing import Tuple, Dict, Any

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter


class DetectionTransform:
    """
    对图像 + bbox 同步做变换：
    - resize 到固定尺寸
    - 随机水平翻转（仅训练）
    - 颜色抖动（仅训练）
    - Tensor 化 + Normalize
    """

    def __init__(
        self,
        train: bool = True,
        resize_size: Tuple[int, int] = (480, 480),
    ):
        self.train = train
        self.resize_size = resize_size  # (h, w)

        # ImageNet 归一化参数
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if train:
            self.color_jitter = ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )

    def __call__(self, image, target: Dict[str, Any]):
        """
        image: PIL.Image (来自 VOCDetection)
        target: {
            "boxes": Tensor[N, 4]  (xmin, ymin, xmax, ymax)，像素坐标
            ...
        }
        """

        # -------------------------
        # 1) 记录原始尺寸并做 resize（在 PIL 上）
        # -------------------------
        orig_w, orig_h = image.size  # PIL: (w, h)

        if self.resize_size is not None:
            new_h, new_w = self.resize_size

            # resize 图像（PIL）
            image = F.resize(image, (new_h, new_w))

            # 同步缩放 bbox
            boxes = target["boxes"].clone()
            scale_w = new_w / orig_w
            scale_h = new_h / orig_h

            # x 坐标按 w 比例缩放，y 坐标按 h 比例缩放
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
            target["boxes"] = boxes

        # -------------------------
        # 2) PIL -> Tensor
        # -------------------------
        image = F.to_tensor(image)  # [C, H, W], [0, 1]

        # -------------------------
        # 3) 训练时的数据增强
        # -------------------------
        if self.train:
            # 3.1 随机水平翻转
            if random.random() < 0.5:
                _, h, w = image.shape
                image = torch.flip(image, dims=[2])  # 在宽度维度上翻转

                boxes = target["boxes"]
                # 水平翻转：x' = w - x
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                target["boxes"] = boxes

            # 3.2 随机颜色抖动
            # ColorJitter 可以作用在 PIL，也可以作用在 tensor，
            # 这里转成 PIL 再转回 tensor，行为更直观
            image_pil = F.to_pil_image(image)
            image_pil = self.color_jitter(image_pil)
            image = F.to_tensor(image_pil)

        # -------------------------
        # 4) Normalize
        # -------------------------
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target
