# train.py
import os
import time
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils.logger import CSVLogger
from utils.metrics import evaluate_loss_only, evaluate_map
from utils.visualize import visualize_predictions
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


from config import (
    VOC_ROOT, NUM_CLASSES, DEVICE, BATCH_SIZE, NUM_WORKERS,
    NUM_EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, LR_STEP_SIZE, LR_GAMMA,
    LOG_DIR, WEIGHTS_DIR
)

from datasets.voc_dataset import VOCDataset
from transforms.transforms import DetectionTransform
from models.faster_rcnn_resnet101 import get_faster_rcnn_resnet101
from utils.logger import CSVLogger
from utils.metrics import evaluate_loss_only

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")


# writer = SummaryWriter(log_dir="runs/exp1")

loss_history = []    # 用来画 matplotlib loss 图
mAP_history = []     # 用来画 matplotlib mAP 图
val_loss_history = []
lr_history = []

def collate_fn(batch):
    """Custom collate function to handle variable-size targets."""
    return tuple(zip(*batch))


def save_model_structure(model, path):
    with open(path, "w") as f:
        f.write(str(model))
    print(f"[Model] Saved model structure to {path}")


def main():
    print("===== Initializing Training =====")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # ===============================
    #   Dataset & Dataloader
    # ===============================
    print("[Data] Loading VOC Dataset...")

    # train_dataset = VOCDataset(
    #     root=VOC_ROOT,
    #     year="2012",              # 你说你要切换到 VOC2012，我直接为你设置好了
    #     image_set="train",
    #     transforms=DetectionTransform(train=True)
    # )
    train_dataset = ConcatDataset([
        VOCDataset(root=VOC_ROOT, year="2007", image_set="trainval", transforms=DetectionTransform(train=True)),
        VOCDataset(root=VOC_ROOT, year="2012", image_set="trainval", transforms=DetectionTransform(train=True)),
    ])

    val_dataset = VOCDataset(
        root=VOC_ROOT,
        year="2012",
        image_set="val",
        transforms=DetectionTransform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,             # 验证不用太大
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        collate_fn=collate_fn
    )

    # ===============================
    #   Model
    # ===============================
    print("[Model] Building Faster R-CNN...")
    model = get_faster_rcnn_resnet101(num_classes=NUM_CLASSES, pretrained_backbone=True)
    model.to(device)

    # if torch.cuda.device_count() > 1:
    #     print(f"[GPU] Using {torch.cuda.device_count()} GPUs with DataParallel")
    #     model = torch.nn.DataParallel(model)

    # ===============================
    #   Optimizer & Scheduler
    # ===============================
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # ===============================
    #   Logger & Model Structure
    # ===============================
    csv_logger = CSVLogger(os.path.join(LOG_DIR, "train_log.csv"))
    save_model_structure(model, os.path.join(WEIGHTS_DIR, "model_structure.txt"))

    # AMP
    scaler = GradScaler()

    total_start = time.time()
    best_map = 0.0

    # ===============================
    #   TRAINING LOOP
    # ===============================
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/{NUM_EPOCHS}", unit="batch")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            epoch_loss += loss_value

            # ==== TensorBoard ====
            global_step = (epoch - 1) * len(train_loader) + pbar.n
            # writer.add_scalar("train/total_loss", loss_value, global_step)

            # ==== Matplotlib data ====
            if pbar.n % 100 == 0:  # 每 100 iteration 记录一次
                loss_history.append(loss_value)

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": optimizer.param_groups[0]["lr"]
            })

        epoch_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start

        print(f"[Epoch {epoch}] train_loss={epoch_loss:.4f} | time={epoch_time:.2f}s ({epoch_time/60:.2f} min)")

        # ===============================
        #   VALIDATION
        # ===============================
        val_metrics = evaluate_loss_only(model, val_loader, device)
        val_loss = val_metrics["val_loss"]

        map_metrics = evaluate_map(model, val_loader, device)
        val_map = map_metrics["mAP"]

        # ==== Matplotlib data ====
        val_loss_history.append(val_loss)
        mAP_history.append(val_map)

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        lr_history.append(lr)  # 保存学习率历史

        # ==== TensorBoard ====
        # writer.add_scalar("val/mAP", val_map, epoch)
        # writer.add_scalar("val/loss", val_loss, epoch)
        # writer.add_scalar("train/lr", lr, epoch)

        print(f"[Epoch {epoch}] val_loss={val_loss:.4f} | mAP={val_map:.4f} | lr={lr:.6f}")

        # 3) 记录到 csv 日志
        csv_logger.log(
            epoch=epoch,
            train_loss=epoch_loss,
            val_loss=val_loss,
            mAP=val_map,
            lr=lr,
            epoch_time=epoch_time,
        )

        # 4) 保存 best mAP 模型
        if val_map > best_map:
            best_map = val_map
            best_path = os.path.join(WEIGHTS_DIR, "best.pth")
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), best_path)
            print(f"[Checkpoint] New best mAP={best_map:.4f}, saved to {best_path}")

        # ===============================
        #   可视化：每个 epoch 画一张验证图的预测结果
        # ===============================
        model.eval()
        with torch.no_grad():
            # 从 val_loader 取一张图
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                output = outputs[0]

                # 类名列表：背景 + VOC20类
                class_names = ["__background__"] + list(val_dataset.CLASSES)

                # 只画第一张就 break
                visualize_predictions(
                    images[0].cpu(),
                    output,
                    class_names,
                    score_thresh=0.5,
                    save_name=f"epoch_{epoch}_pred.png",
                )
                break
        model.train()



        # Save weights
        weight_path = os.path.join(WEIGHTS_DIR, f"model_epoch_{epoch}.pth")
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), weight_path)
        print(f"[Checkpoint] Saved weights to {weight_path}")

    # ===============================
    #   END OF TRAINING
    # ===============================
    total_time = time.time() - total_start
    print("\n===== Training Completed =====")
    print(f"Total time: {total_time:.2f} sec ({total_time/60:.2f} min)")

    # ---- Training Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="train_loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Iteration (sampled every 100 steps)")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    # ---- Validation Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(val_loss_history, label="val_loss")
    plt.title("Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("val_loss_curve.png")
    plt.close()

    # ---- Validation mAP ----
    plt.figure(figsize=(10, 5))
    plt.plot(mAP_history, label="mAP")
    plt.title("Validation mAP Curve")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.savefig("mAP_curve.png")
    plt.close()

    # ---- Learning Rate ----
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, label="learning_rate")
    plt.title("Learning Rate Curve")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.savefig("lr_curve.png")
    plt.close()

    print("Saved loss_curve.png, val_loss_curve.png, mAP_curve.png, and lr_curve.png")


if __name__ == "__main__":
    main()
