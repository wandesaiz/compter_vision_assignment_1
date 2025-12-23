import os
import torch

# ---------------------
# Path Config
# ---------------------

# PROJECT_ROOT = r"D:/Project/Computer_vision/Course Assignment 2_Object Detection"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# VOC 根路径（train.py 需要 VOC_ROOT）
VOC_ROOT = os.path.join(PROJECT_ROOT, "data", "VOCdevkit")

# VOC2012 完整路径
DATA_ROOT = os.path.join(VOC_ROOT, "VOC2012")

IMAGE_DIR = os.path.join(DATA_ROOT, "JPEGImages")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "Annotations")

TRAIN_LIST = os.path.join(DATA_ROOT, "ImageSets", "Main", "train.txt")
VAL_LIST  = os.path.join(DATA_ROOT, "ImageSets", "Main", "val.txt")

# ---------------------
# Save Directories
# ---------------------
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
LOG_DIR = os.path.join(RUNS_DIR, "logs")
WEIGHTS_DIR = os.path.join(RUNS_DIR, "weights")  # train.py 用的是 WEIGHTS_DIR
FIG_DIR = os.path.join(RUNS_DIR, "figs")
VIS_DIR = os.path.join(RUNS_DIR, "vis")

for d in [LOG_DIR, WEIGHTS_DIR, FIG_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------
# Training Config
# ---------------------
NUM_CLASSES = 21       # VOC = 20 classes + background
BATCH_SIZE = 1
NUM_WORKERS = 2
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# 学习率调度器
LR_STEP_SIZE = 5       # 每 5 epoch 降一次 LR
LR_GAMMA = 0.1         # LR *= 0.1

NUM_EPOCHS = 30
PRINT_FREQ = 50

# AMP 混合精度
USE_AMP = True

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
