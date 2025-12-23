# datasets/voc_dataset.py
import os
import torch
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from PIL import Image


class VOCDataset(Dataset):
    """
    适配你当前的 VOC2012 数据集结构（分类版 ImageSets/Main/*.txt）
    自动从所有类别的 txt 中合并出 detection 的 train/val split。
    """

    CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    ]

    def __init__(self, root, year="2012", image_set="trainval", transforms=None):
        self.root = root
        voc_root = os.path.join(root, f"VOC{year}")
        self.transforms = transforms

        # ─────────────────────────────
        # 1. 自动根据你的数据集结构构造 detection split
        # ─────────────────────────────
        split_dir = os.path.join(voc_root, "ImageSets", "Main")

        if image_set == "trainval":
            split_files = [f"{c}_trainval.txt" for c in self.CLASSES]
        elif image_set == "train":
            split_files = [f"{c}_train.txt" for c in self.CLASSES]
        elif image_set == "val":
            split_files = [f"{c}_val.txt" for c in self.CLASSES]
        else:
            raise ValueError(f"Unknown split {image_set}")

        img_ids = set()

        for fname in split_files:
            path = os.path.join(split_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    # 每行格式为： "2008_000003 1"
                    parts = line.split()
                    img_id = parts[0]
                    img_ids.add(img_id)

        self.ids = sorted(list(img_ids))

        # ─────────────────────────────
        # 2. 准备路径
        # ─────────────────────────────
        self.image_dir = os.path.join(voc_root, "JPEGImages")
        self.annotations_dir = os.path.join(voc_root, "Annotations")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # 图片路径
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        # 读取图像

        image = Image.open(img_path).convert("RGB")

        # ─────────────────────────────
        # 3. 解析 XML 获取 bbox 和类别
        # ─────────────────────────────
        anno_file = os.path.join(self.annotations_dir, f"{img_id}.xml")
        tree = ET.parse(anno_file)
        objs = tree.findall("object")

        boxes = []
        labels = []

        for obj in objs:
            cls = obj.find("name").text.lower().strip()
            if cls not in self.CLASSES:
                continue
            labels.append(self.CLASSES.index(cls) + 1)

            xml_box = obj.find("bndbox")
            bbox = [
                float(xml_box.find("xmin").text),
                float(xml_box.find("ymin").text),
                float(xml_box.find("xmax").text),
                float(xml_box.find("ymax").text),
            ]
            boxes.append(bbox)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        # ─────────────────────────────
        # 4. transforms（含 resize、flip、norm）
        # ─────────────────────────────
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
