# infer.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import VOC_ROOT, NUM_CLASSES, DEVICE, BATCH_SIZE, NUM_WORKERS, WEIGHTS_DIR
from datasets.voc_dataset import VOCDataset
from transforms.transforms import DetectionTransform
from models.faster_rcnn_resnet101 import get_faster_rcnn_resnet101
from utils.visualize import visualize_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


def main(weight_file, num_images=5):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    dataset = VOCDataset(
        root=VOC_ROOT,
        year="2012",
        image_set="val",
        transforms=DetectionTransform(train=False)
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
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

    # class_names 用 dataset 的
    class_names = dataset.class_names

    with torch.no_grad():
        pbar = tqdm(data_loader, total=num_images, desc="Inference", unit="img")
        for i, (images, targets) in enumerate(pbar):
            if i >= num_images:
                break
            image = images[0].to(device)

            outputs = model([image])
            output = outputs[0]

            visualize_predictions(
                image.cpu(),
                output,
                class_names,
                score_thresh=0.5,
                save_name=f"pred_{i}.png"
            )


if __name__ == "__main__":
    weight_path = os.path.join(WEIGHTS_DIR, "model_epoch_10.pth")
    main(weight_path, num_images=5)
