# eval.py
import os
import re
import argparse
from typing import Tuple, Optional, Any, Dict

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from config import WEIGHTS_DIR, BATCH_SIZE, NUM_WORKERS, DEVICE, DATA_ROOT
from models import get_segformer_model
from datasets.cityscapes_custom import CityscapesCustom
from transforms.seg_transforms import SegmentationTransform


def pick_best_checkpoint(weights_dir: str) -> str:
    ckpts = [
        f for f in os.listdir(weights_dir)
        if f.startswith("best_segformer_miou_") and f.endswith(".pt")
    ]
    if not ckpts:
        raise FileNotFoundError(f"No best checkpoints found in: {weights_dir}")

    def get_miou(fn: str) -> float:
        m = re.search(r"miou_([0-9.]+)\.pt$", fn)
        return float(m.group(1)) if m else -1.0

    best_fn = max(ckpts, key=get_miou)
    return os.path.join(weights_dir, best_fn)


def resolve_checkpoint_path(checkpoint: str) -> str:
    if checkpoint == "best":
        return pick_best_checkpoint(WEIGHTS_DIR)

    if checkpoint in ["last", "last_model.pt"]:
        return os.path.join(WEIGHTS_DIR, "last_model.pt")

    if os.path.isfile(checkpoint):
        return checkpoint

    cand = os.path.join(WEIGHTS_DIR, checkpoint)
    if os.path.isfile(cand):
        return cand

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint} (or {cand})")


def load_model_from_ckpt(ckpt_path: str, device) -> torch.nn.Module:
    print(f"[Eval] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        else:
            state = ckpt
    else:
        state = ckpt

    model = get_segformer_model()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def extract_images_targets(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Your dataset batch keys are: {'pixel_values', 'labels', 'id'}
    """
    if isinstance(batch, dict):
        images = batch["pixel_values"]
        targets = batch["labels"]
        return images, targets

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Works for HF SegFormer:
      model(pixel_values=images).logits
    Also supports direct model(images).
    """
    try:
        out = model(pixel_values=images)
    except TypeError:
        out = model(images)

    if hasattr(out, "logits"):
        return out.logits

    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "out" in out:
            return out["out"]

    if isinstance(out, (list, tuple)):
        return out[0]

    return out


@torch.no_grad()
def evaluate_one_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    ignore_index: int = 255,
    upsample_mode: str = "bilinear",
) -> Tuple[float, float, torch.Tensor]:
    """
    One pass: mean CE loss + mIoU + per-class IoU.
    Fixes size mismatch by upsampling logits to target H,W.
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    total_loss = 0.0
    n_batches = 0

    confmat = None
    num_classes: Optional[int] = None

    printed = False

    for batch in loader:
        if isinstance(batch, dict) and not printed:
            print(f"[Eval] Batch dict keys: {list(batch.keys())}")
            printed = True

        images, targets = extract_images_targets(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()  # [B,H,W]

        logits = forward_logits(model, images)  # [B,C,h,w]

        # --- IMPORTANT: upsample logits to target size ---
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode=upsample_mode,
                align_corners=False if upsample_mode in ["bilinear", "bicubic"] else None,
            )

        if num_classes is None:
            num_classes = logits.shape[1]
            confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        loss = criterion(logits, targets)
        total_loss += float(loss.item())
        n_batches += 1

        preds = torch.argmax(logits, dim=1)  # [B,H,W]

        preds = preds.view(-1).detach().cpu()
        t = targets.view(-1).detach().cpu()

        valid = (t != ignore_index)
        preds = preds[valid]
        t = t[valid]

        k = (t * num_classes + preds).to(torch.int64)
        bincount = torch.bincount(k, minlength=num_classes * num_classes)
        confmat += bincount.reshape(num_classes, num_classes)

    mean_loss = total_loss / max(n_batches, 1)

    intersection = torch.diag(confmat).to(torch.float32)
    gt_sum = confmat.sum(dim=1).to(torch.float32)
    pred_sum = confmat.sum(dim=0).to(torch.float32)
    union = gt_sum + pred_sum - intersection

    per_class_iou = intersection / torch.clamp(union, min=1.0)
    miou = float(per_class_iou.mean().item())

    return mean_loss, miou, per_class_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best",
                        help='"best", "last", filename in WEIGHTS_DIR, or a path to .pt')
    parser.add_argument("--ignore_index", type=int, default=255, help="Cityscapes void label is often 255")
    parser.add_argument("--save_npy", action="store_true", help="Save per_class_iou.npy to WEIGHTS_DIR")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint_path(args.ckpt)
    model = load_model_from_ckpt(ckpt_path, DEVICE)

    val_tf = SegmentationTransform(is_train=False)
    val_dataset = CityscapesCustom(DATA_ROOT, split="val", transform=val_tf)

    pin = True
    try:
        if isinstance(DEVICE, str):
            pin = DEVICE.startswith("cuda")
        else:
            pin = (DEVICE.type == "cuda")
    except Exception:
        pin = True

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    val_loss, miou, per_class_iou = evaluate_one_pass(
        model, val_loader, DEVICE, ignore_index=args.ignore_index
    )

    print(f"[Eval] ckpt: {os.path.basename(ckpt_path)}")
    print(f"[Eval] Val Loss: {val_loss:.4f}, mIoU: {miou:.4f}")
    print("[Eval] Per-class IoU:")
    for i, v in enumerate(per_class_iou.tolist()):
        print(f"  class {i}: {v:.4f}")

    if args.save_npy:
        out_path = os.path.join(WEIGHTS_DIR, "per_class_iou.npy")
        np.save(out_path, per_class_iou.numpy())
        print(f"[Eval] Saved per-class IoU to: {out_path}")


if __name__ == "__main__":
    main()
