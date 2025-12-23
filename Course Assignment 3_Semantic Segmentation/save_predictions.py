import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config import DEVICE, IMAGE_SIZE, WEIGHTS_DIR, FIG_DIR

from models import get_segformer_model

# ---- VOC 21类标准 palette（用于把 label PNG 以调色板方式保存）----
def voc_palette():
    # VOC 官方常用 palette（256*3）
    palette = [0] * (256 * 3)
    for j in range(256):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette

def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    # mask_np: [H,W] int
    m = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    m.putpalette(voc_palette())
    return m.convert("RGB")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best.pth",
                        help="checkpoint file under WEIGHTS_DIR, e.g. best.pth or last.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="VOC split to export")
    parser.add_argument("--outdir", type=str, default="predictions",
                        help="output folder name under FIG_DIR")
    parser.add_argument("--save_raw", action="store_true",
                        help="also save raw mask (palette PNG)")
    args = parser.parse_args()

    # 1) load model
    ckpt_path = os.path.join(WEIGHTS_DIR, args.ckpt)
    model = get_segformer_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE).eval()
    print(f"[OK] Loaded checkpoint: {ckpt_path}")

    # 2) dataset / loader
    ds = VOCSegDataset(split=args.split, image_size=IMAGE_SIZE, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # 3) output dirs
    out_root = os.path.join(FIG_DIR, args.outdir, args.split)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "mask_rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "overlay"), exist_ok=True)
    if args.save_raw:
        os.makedirs(os.path.join(out_root, "mask_raw"), exist_ok=True)

    for batch in loader:
        img = batch["pixel_values"].to(DEVICE)      # [1,3,H,W]
        gt = batch["labels"].cpu().numpy()[0]       # [H,W]
        img_id = batch["id"][0]

        # 4) forward
        out = model(pixel_values=img)
        logits = out.logits if hasattr(out, "logits") else out  # 兼容写法

        # 5) resize logits -> label size
        H, W = gt.shape
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)  # [H,W]

        # 6) save mask RGB (palette -> RGB)
        pred_rgb = colorize_mask(pred)
        pred_rgb.save(os.path.join(out_root, "mask_rgb", f"{img_id}.png"))

        # 7) optionally save raw palette mask (mode=P)
        if args.save_raw:
            m = Image.fromarray(pred, mode="P")
            m.putpalette(voc_palette())
            m.save(os.path.join(out_root, "mask_raw", f"{img_id}.png"))

        # 8) overlay (把输入图反归一化：简单做法——按 [0,1] clamp，可视化足够用)
        #    如果你 transforms 里做了 mean/std normalize，想更“真实”，我也可以按你 transforms 具体参数给你反归一化版本
        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        base = (img_np * 255).astype(np.uint8)
        base_pil = Image.fromarray(base)

        overlay = Image.blend(base_pil.convert("RGB"), pred_rgb, alpha=0.5)
        overlay.save(os.path.join(out_root, "overlay", f"{img_id}.png"))

    print(f"[Done] Saved to: {out_root}")

if __name__ == "__main__":
    main()




# conda activate cv3
# cd "Course Assignment 3_Semantic Segmentation"
# python save_predictions.py --ckpt best.pth --split val --save_raw