# utils/plot_curves.py
import os
import csv
import matplotlib.pyplot as plt

from config import FIG_DIR


def plot_from_csv(csv_path, x_key="epoch", y_keys=("train_loss", "val_loss"), save_name="learning_curve.png"):
    xs = []
    ys = {k: [] for k in y_keys}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[x_key]))
            for k in y_keys:
                if k in row and row[k] != "":
                    ys[k].append(float(row[k]))
                else:
                    ys[k].append(None)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    for k in y_keys:
        ax.plot(xs, ys[k], label=k)

    ax.set_xlabel(x_key)
    ax.set_ylabel("loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved curve to {save_path}")


if __name__ == "__main__":
    from config import LOG_DIR

    csv_path = os.path.join(LOG_DIR, "train_log.csv")

    # 1) 画 loss 曲线
    plot_from_csv(
        csv_path,
        x_key="epoch",
        y_keys=("train_loss", "val_loss"),
        save_name="loss_curve.png",
    )

    # 2) 画 mAP 曲线
    plot_from_csv(
        csv_path,
        x_key="epoch",
        y_keys=("mAP",),
        save_name="mAP_curve.png",
    )
