# infer.py
import os
import argparse

import torch
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

from utils.visualize import cityscapes_palette, decode_segmap, _denormalize
from config import DEVICE, IMAGE_SIZE, WEIGHTS_DIR
from models import get_segformer_model


def load_model(ckpt_path: str):
    model = get_segformer_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


def _get_resize_size_for_pil():
    """把 config.IMAGE_SIZE 转成 PIL 需要的 (W, H)"""
    if isinstance(IMAGE_SIZE, int):
        return (IMAGE_SIZE, IMAGE_SIZE)
    h, w = IMAGE_SIZE
    return (w, h)   # PIL: (width, height)


def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")

    resize_size = _get_resize_size_for_pil()
    img = img.resize(resize_size, Image.BILINEAR)

    img_t = F.to_tensor(img)
    img_t = F.normalize(
        img_t,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return img_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="image path")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    args = parser.parse_args()

    if args.ckpt is None:
        ckpts = [f for f in os.listdir(WEIGHTS_DIR) if "best" in f and f.endswith(".pt")]
        if not ckpts:
            raise FileNotFoundError("No best checkpoint found.")
        ckpt_path = os.path.join(WEIGHTS_DIR, sorted(ckpts)[-1])
    else:
        ckpt_path = args.ckpt

    print(f"[Infer] Using checkpoint: {ckpt_path}")
    model = load_model(ckpt_path)

    if not os.path.exists(args.img):
        raise FileNotFoundError(args.img)

    img_pil = Image.open(args.img)
    img_t = preprocess_image(img_pil).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

    with torch.no_grad():
        outputs = model(pixel_values=img_t)
        pred = torch.argmax(outputs.logits, dim=1)[0]  # [H,W]

    # 可视化
    img_denorm = _denormalize(img_t[0].cpu())
    pred_np = pred.cpu().numpy()

    # Cityscapes 调色盘
    palette = cityscapes_palette()
    pred_color = decode_segmap(pred_np, palette)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_denorm)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(pred_color)
    axs[1].set_title("Prediction")
    axs[1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(WEIGHTS_DIR, "infer_result.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Infer] Result saved to {out_path}")


if __name__ == "__main__":
    main()
