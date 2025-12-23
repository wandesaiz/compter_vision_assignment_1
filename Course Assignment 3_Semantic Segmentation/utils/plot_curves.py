# utils/plot_curves.py
import os
import csv
import matplotlib.pyplot as plt

from config import FIG_DIR


def plot_from_csv(csv_path, x_key="epoch", y_keys=("train_loss", "val_loss"), save_name="loss_curve.png"):
    xs = []
    ys = {k: [] for k in y_keys}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[x_key]))
            for k in y_keys:
                ys[k].append(float(row[k]))

    for k, v in ys.items():
        plt.plot(xs, v, label=k)

    plt.xlabel(x_key)
    plt.legend()
    plt.grid(True)
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved curve to {save_path}")


def main(csv_path: str):
    # 1) loss 曲线
    plot_from_csv(
        csv_path,
        x_key="epoch",
        y_keys=("train_loss", "val_loss"),
        save_name="loss_curve.png",
    )

    # 2) mIoU 曲线
    plot_from_csv(
        csv_path,
        x_key="epoch",
        y_keys=("miou",),
        save_name="miou_curve.png",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to train_log.csv")
    args = parser.parse_args()

    main(args.csv)

