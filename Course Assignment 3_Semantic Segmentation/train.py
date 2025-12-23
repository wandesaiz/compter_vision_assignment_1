# train.py
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    DATA_ROOT, WEIGHTS_DIR, LOG_DIR, NUM_EPOCHS, BATCH_SIZE,
    NUM_WORKERS, LR, WEIGHT_DECAY, LR_STEP_SIZE, LR_GAMMA,
    DEVICE,IMAGE_SIZE, SEED,
)
import torch.backends.cudnn as cudnn
from models import get_segformer_model
from utils.logger import CSVLogger
from utils.metrics import evaluate_loss_only, evaluate_miou
from utils.visualize import visualize_segmentation
from datasets.cityscapes_custom import CityscapesCustom
from transforms.seg_transforms import SegmentationTransform
import torch.nn.functional as F


def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def save_model_structure(model, save_path: str):
    # 简单把模型结构文本保存下来，方便作业附录
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(model))
    print(f"[Model] Structure saved to {save_path}")


# def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, print_freq=50):
#     model.train()
#     running_loss = 0.0
#     num_batches = len(dataloader)
#
#     pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f"[Train] Epoch {epoch}")
#     for step, batch in pbar:
#         images = batch["pixel_values"].to(device)
#         labels = batch["labels"].to(device)
#
#         optimizer.zero_grad(set_to_none=True)
#
#         if USE_AMP:
#             # with autocast():
#             #     outputs = model(pixel_values=images, labels=labels)
#             #     loss = outputs.loss
#             with autocast():
#                 outputs = model(pixel_values=images)
#                 logits = outputs.logits
#                 logits = torch.nn.functional.interpolate(
#                     logits,
#                     size=labels.shape[-2:],
#                     mode="bilinear",
#                     align_corners=False
#                 )
#                 loss = torch.nn.functional.cross_entropy(
#                     logits,
#                     labels,
#                     ignore_index=255
#                 )
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             # outputs = model(pixel_values=images, labels=labels)
#             # loss = outputs.loss
#             outputs = model(pixel_values=images)
#             logits = outputs.logits
#             # 将模型输出上采样到 label 大小
#             logits = torch.nn.functional.interpolate(
#                 logits,
#                 size=labels.shape[-2:],  # (512, 1024)
#                 mode="bilinear",
#                 align_corners=False
#             )
#
#             loss = torch.nn.functional.cross_entropy(
#                 logits,
#                 labels,
#                 ignore_index=255
#             )
#
#             loss.backward()
#             optimizer.step()
#
#         running_loss += loss.item()
#         if (step + 1) % print_freq == 0:
#             avg_loss = running_loss / (step + 1)
#             pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
#
#     epoch_loss = running_loss / max(1, num_batches)
#     return epoch_loss
def train_one_epoch(model, dataloader, optimizer, device, epoch, print_freq=50):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f"[Train] Epoch {epoch}")
    for step, batch in pbar:
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],   # (H,W)
            mode="bilinear",
            align_corners=False,
        )

        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=255,
            reduction="mean",
        )

        loss.backward()

        # 梯度裁剪，强力防炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        if (step + 1) % print_freq == 0:
            avg_loss = running_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    epoch_loss = running_loss / max(1, num_batches)
    return epoch_loss


def main():
    if not os.path.exists(DATA_ROOT):
        raise RuntimeError(f"[Error] DATA_ROOT not found: {DATA_ROOT}")

    set_seed(SEED)
    print(f"[Config] Using device: {DEVICE}")

    # ========= Dataset & DataLoader =========
    # train_dataset = VOCSegDataset(split="train", image_size=IMAGE_SIZE, is_train=True)
    # val_dataset = VOCSegDataset(split="val", image_size=IMAGE_SIZE, is_train=False)

    train_tf = SegmentationTransform(is_train=True)
    val_tf = SegmentationTransform(is_train=False)
    train_dataset = CityscapesCustom(DATA_ROOT, split="train", transform=train_tf)
    val_dataset = CityscapesCustom(DATA_ROOT, split="val", transform=val_tf)

    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ========= Model =========
    model = get_segformer_model()
    model.to(DEVICE)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[MultiGPU] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    print("[Model] SegFormer-B0 initialized.")

    # ========= Optimizer & Scheduler =========
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # ========= Logger =========
    csv_logger = CSVLogger(os.path.join(LOG_DIR, "train_log.csv"))
    save_model_structure(model, os.path.join(WEIGHTS_DIR, "model_structure.txt"))


    best_miou = 0.0
    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch} / {NUM_EPOCHS} ==========")

        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, print_freq=50)
        val_loss = evaluate_loss_only(model, val_loader, DEVICE)
        miou, per_class_iou = evaluate_miou(model, val_loader, DEVICE)

        lr_scheduler.step()

        print(f"[Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f}")

        # 记录到 CSV
        csv_logger.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            miou=miou,
            lr=optimizer.param_groups[0]["lr"],
        )

        # 每个 epoch 在 val 集合上可视化一张
        with torch.no_grad():
            for batch in val_loader:
                images = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                ids = batch["id"]
                outputs = model(pixel_values=images)
                preds = torch.argmax(outputs.logits, dim=1)
                # 只画 batch 中第 0 张
                visualize_segmentation(
                    images[0].cpu(), labels[0].cpu(), preds[0].cpu(),
                    save_name=f"epoch_{epoch}_id_{ids[0]}.png",
                )
                break

        # 保存 best 模型
        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(WEIGHTS_DIR, f"best_segformer_miou_{miou:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "miou": miou,
                },
                best_path,
            )
            print(f"[Checkpoint] New best mIoU {miou:.4f}, saved to {best_path}")

        # 每个 epoch 也保存一个 last_model
        last_path = os.path.join(WEIGHTS_DIR, "last_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "miou": miou,
            },
            last_path,
        )

    total_time = time.time() - total_start
    print(f"\n[Train] Finished. Best mIoU = {best_miou:.4f}. "
          f"Total time = {total_time / 3600:.2f}h")


if __name__ == "__main__":
    main()

