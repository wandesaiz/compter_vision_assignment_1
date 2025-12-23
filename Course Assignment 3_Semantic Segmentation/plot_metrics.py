import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from config import FIG_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to train_log.csv")
    parser.add_argument("--outdir", type=str, default="metrics_plots", help="folder under FIG_DIR")
    args = parser.parse_args()

    os.makedirs(os.path.join(FIG_DIR, args.outdir), exist_ok=True)
    df = pd.read_csv(args.csv)

    # Loss 曲线
    if "epoch" in df.columns and "train_loss" in df.columns and "val_loss" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, args.outdir, "loss_curve.png"), dpi=200)
        plt.close()

    # mIoU 曲线
    if "epoch" in df.columns and "miou" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["miou"], label="mIoU")
        plt.xlabel("epoch")
        plt.ylabel("mIoU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, args.outdir, "miou_curve.png"), dpi=200)
        plt.close()

    print(f"[Done] Plots saved to {os.path.join(FIG_DIR, args.outdir)}")

if __name__ == "__main__":
    main()


# python plot_metrics.py --csv path/to/train_log.csv