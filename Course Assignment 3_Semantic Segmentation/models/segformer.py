# models/segformer.py
from typing import Dict

from transformers import SegformerForSemanticSegmentation

from config import NUM_CLASSES, IGNORE_INDEX, SEGFORMER_PRETRAINED_NAME


# def get_segformer_model(num_classes: int = NUM_CLASSES,
#                         ignore_index: int = IGNORE_INDEX):
#     """
#     构建 SegFormer-B0 语义分割模型。
#     使用 HuggingFace 预训练权重，然后替换 head 为 num_classes。
#     第一次运行会从网上下载权重。
#     """
#     id2label: Dict[int, str] = {i: f"class_{i}" for i in range(num_classes)}
#     label2id: Dict[str, int] = {v: k for k, v in id2label.items()}
#
#     model = SegformerForSemanticSegmentation.from_pretrained(
#         SEGFORMER_PRETRAINED_NAME,
#         num_labels=num_classes,
#         id2label=id2label,
#         label2id=label2id,
#         ignore_mismatched_sizes=True,   # 分类头会自动重新初始化
#     )
#
#     # 设置 ignore_index
#     model.config.semantic_loss_ignore_index = ignore_index
#     return model

# models/segformer.py
import os
from typing import Dict
from transformers import SegformerForSemanticSegmentation
from config import NUM_CLASSES, IGNORE_INDEX, SEGFORMER_PRETRAINED_NAME

from pathlib import Path
from transformers import SegformerForSemanticSegmentation
from config import NUM_CLASSES, IGNORE_INDEX

def get_segformer_model(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    base = Path(__file__).resolve().parent          # .../models
    local_dir = base / "segformer-b1-finetuned-ade-512-512"      # 你的文件夹名

    model = SegformerForSemanticSegmentation.from_pretrained(
        str(local_dir),
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.config.semantic_loss_ignore_index = ignore_index
    return model



