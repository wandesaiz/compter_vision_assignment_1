import os
import numpy as np
import matplotlib.pyplot as plt
from config import WEIGHTS_DIR, FIG_DIR

def main():
    arr = np.load(os.path.join(WEIGHTS_DIR, "per_class_iou.npy"))  # [C]
    plt.figure(figsize=(12,4))
    plt.bar(np.arange(len(arr)), arr)
    plt.xlabel("class id")
    plt.ylabel("IoU")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "per_class_iou.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[Done] {out}")

if __name__ == "__main__":
    main()


# python eval.py            # 先生成 per_class_iou.npy
# python plot_per_class_iou.py
