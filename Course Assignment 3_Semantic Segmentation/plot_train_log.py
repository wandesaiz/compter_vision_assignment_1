# plot_train_log.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from config import FIG_DIR


def plot_and_save(df, x_col, y_col, title, ylabel, out_path):
    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_csv",
        type=str,
        default="runs_segformer_cityscapes_b4_3/logs/train_log.csv",
        help="Path to train_log.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=FIG_DIR,
        help="Output dir for plots (default: FIG_DIR from config.py)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.log_csv)
    required = ["epoch", "train_loss", "val_loss", "miou"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {args.log_csv}. Found: {list(df.columns)}")

    # 画三张图
    plot_and_save(
        df, "epoch", "train_loss",
        "Training Loss vs Epoch", "train_loss",
        os.path.join(args.out_dir, "train_loss_curve.png")
    )
    plot_and_save(
        df, "epoch", "val_loss",
        "Validation Loss vs Epoch", "val_loss",
        os.path.join(args.out_dir, "val_loss_curve.png")
    )
    plot_and_save(
        df, "epoch", "miou",
        "mIoU vs Epoch", "mIoU",
        os.path.join(args.out_dir, "miou_curve.png")
    )


if __name__ == "__main__":
    main()
