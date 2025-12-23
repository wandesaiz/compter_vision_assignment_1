# voc_test_infer.py
import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from config import NUM_CLASSES, DEVICE, NUM_WORKERS, WEIGHTS_DIR
from transforms.transforms import DetectionTransform
from models.faster_rcnn_resnet101 import get_faster_rcnn_resnet101
from datasets.voc_dataset import VOCDataset
from utils.visualize import visualize_predictions


# ===============================================
#            TEST DATASET (CUSTOM)
# ===============================================
class VOCTestDataset(Dataset):
    """
    使用独立 test 文件夹：
        data/VOC2012_test/
            ImageSets/Main/test.txt
            JPEGImages/*.jpg
    不使用 VOC_ROOT/VOC2012 !!
    """

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms

        split_file = os.path.join(root, "ImageSets", "Main", "test.txt")
        img_dir = os.path.join(root, "JPEGImages")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"没有找到 test.txt: {split_file}")

        with open(split_file, "r") as f:
            self.ids = [x.strip() for x in f.readlines() if x.strip()]

        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # test 集无标注 → 构造空 target
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": img_id,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


# ===============================================
#                 MODEL LOADING
# ===============================================
def load_model(weight_path, device):
    print(f"[Model] 加载 FasterRCNN 模型权重: {weight_path}")
    model = get_faster_rcnn_resnet101(num_classes=NUM_CLASSES, pretrained_backbone=False)

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ===============================================
#                 SAVE TXT RESULTS
# ===============================================
def save_txt_results(output_dir, img_id, detections, class_names):
    txt_path = os.path.join(output_dir, f"{img_id}.txt")
    with open(txt_path, "w") as f:
        for box, label, score in zip(
            detections["boxes"], detections["labels"], detections["scores"]
        ):
            x1, y1, x2, y2 = box.tolist()
            f.write(f"{class_names[label]} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")


# ===============================================
#                 INFERENCE LOOP
# ===============================================
def run_inference(
    test_root: str,
    weight_path: str,
    max_images: int = 0,
    score_thresh: float = 0.5,
):
    # 输出目录（与训练区分）
    save_dir = "runs/figs_test"
    os.makedirs(save_dir, exist_ok=True)

    txt_dir = "runs/pred_txt"
    os.makedirs(txt_dir, exist_ok=True)

    json_dir = "runs/pred_json"
    os.makedirs(json_dir, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Dataset
    dataset = VOCTestDataset(
        root=test_root,
        transforms=DetectionTransform(train=False)
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    print(f"[Data] 加载到 {len(dataset)} 张 test 图片")

    # Load model
    model = load_model(weight_path, device)

    # class names（含 background）
    class_names = ["__background__"] + list(VOCDataset.CLASSES)

    # final results
    pred_summary = []

    num_to_run = len(dataset) if max_images <= 0 else min(max_images, len(dataset))
    print(f"[Infer] 对前 {num_to_run} 张图片做推理 + 可视化\n")

    start_time = time.time()

    with torch.no_grad():
        pbar = tqdm(
            loader,
            total=num_to_run,
            desc="Testing",
            unit="img",
            dynamic_ncols=True,  # 自动适配窗口宽度
            mininterval=0.3,  # 最多每 0.3 秒更新一次 → 不会刷屏
            leave=True  # 推理完成后只保留一条进度条
        )

        for idx, (images, targets) in enumerate(pbar):
            if idx >= num_to_run:
                break

            image = images[0].to(device)
            img_id = targets[0]["image_id"]

            outputs = model([image])[0]

            # 过滤置信度
            keep_idx = outputs["scores"] >= score_thresh
            for k in outputs:
                outputs[k] = outputs[k][keep_idx]

            # 保存 txt
            save_txt_results(txt_dir, img_id, outputs, class_names)

            # 保存 json
            json_path = os.path.join(json_dir, f"{img_id}.json")
            with open(json_path, "w") as jf:
                jf.write(json.dumps({
                    "image_id": img_id,
                    "detections": [
                        {
                            "class": class_names[int(label)],
                            "score": float(score),
                            "bbox": [float(x) for x in box.tolist()]
                        }
                        for box, label, score in zip(
                            outputs["boxes"], outputs["labels"], outputs["scores"]
                        )
                    ]
                }, indent=4))

            # 汇总信息
            pred_summary.append({
                "img_id": img_id,
                "num_boxes": len(outputs["boxes"]),
            })

            # 可视化（PNG）
            # save_path = os.path.join(save_dir, f"{img_id}.png")
            # visualize_predictions(
            #     image.cpu(), outputs, class_names, score_thresh, save_path
            # )
            save_name = f"test_{img_id}.png"
            visualize_predictions(
                image.cpu(),
                outputs,
                class_names,
                score_thresh,
                save_name,
            )

            pbar.set_postfix({"pred_boxes": len(outputs["boxes"])})

    elapsed = time.time() - start_time
    print(f"\n[Done] 推理完成，总耗时 {elapsed:.2f} 秒，FPS = {num_to_run / elapsed:.2f}")

    # 保存 summary
    with open("runs/test_summary.json", "w") as f:
        json.dump(pred_summary, f, indent=4)

    print("[Saved] 所有预测保存在 runs/ 目录内")
    print("  - 可视化图像：runs/figs_test")
    print("  - txt 格式结果：runs/pred_txt")
    print("  - json 格式结果：runs/pred_json")
    print("  - 总结文件：runs/test_summary.json")


# ===============================================
#                 RUN MAIN
# ===============================================
if __name__ == "__main__":
    TEST_ROOT = "data/VOCdevkit/VOC2012_test"     # ← 你的 test 目录
    BEST_WEIGHT = os.path.join(WEIGHTS_DIR, "best.pth")

    run_inference(
        test_root=TEST_ROOT,
        weight_path=BEST_WEIGHT,
        max_images=0,
        score_thresh=0.5,
    )
