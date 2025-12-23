# eval.py
import os
import torch
from torch.utils.data import DataLoader

from config import VOC_ROOT, NUM_CLASSES, DEVICE, BATCH_SIZE, NUM_WORKERS, WEIGHTS_DIR
from datasets.voc_dataset import VOCDataset
from transforms.transforms import DetectionTransform
from models.faster_rcnn_resnet101 import get_faster_rcnn_resnet101
from utils.metrics import evaluate_loss_only, evaluate_map


def collate_fn(batch):
    return tuple(zip(*batch))


def main(weight_file):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    val_dataset = VOCDataset(
        root=VOC_ROOT,
        year="2012",
        image_set="val",
        transforms=DetectionTransform(train=False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = get_faster_rcnn_resnet101(num_classes=NUM_CLASSES, pretrained_backbone=False)
    state_dict = torch.load(weight_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    loss_metrics = evaluate_loss_only(model, val_loader, device)
    map_metrics = evaluate_map(model, val_loader, device)

    print(f"[Eval] val_loss={loss_metrics['val_loss']:.4f}")
    print(f"[Eval] mAP={map_metrics['mAP']:.4f}")


if __name__ == "__main__":
    # 举例：使用第 10 个 epoch
    weight_path = os.path.join(WEIGHTS_DIR, "model_epoch_10.pth")
    main(weight_path)
