import os
import torch

# ---------------------
# Path Config
# ---------------------

# 工程根目录（当前文件所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 如果你想统一放在 data 里，可以把 VOCdevkit 拷到：
#   PROJECT_ROOT/data/VOCdevkit/VOC2012
# 也可以直接改成你现在服务器上的路径
VOC_ROOT = os.path.join(PROJECT_ROOT, "data", "VOCdevkit")
# CITY_ROOT = os.path.join(PROJECT_ROOT, "data", "CityScapes")
CITY_ROOT = os.path.join(PROJECT_ROOT, "data", "CityScapes")


# VOC2012 根目录
# DATA_ROOT = os.path.join(VOC_ROOT, "VOC2012")
DATA_ROOT = CITY_ROOT
LEFTIMG_ROOT = os.path.join(DATA_ROOT, "leftImg8bit")
GTFINE_ROOT  = os.path.join(DATA_ROOT, "gtFine")

#
# IMAGE_DIR = os.path.join(DATA_ROOT, "JPEGImages")
# MASK_DIR = os.path.join(DATA_ROOT, "SegmentationClass")


# IMAGESETS_SEG = os.path.join(DATA_ROOT, "ImageSets", "Segmentation")


# 输出目录（日志、权重、可视化图像等）
# RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_segformer_voc")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_segformer_cityscapes_b1_3")
WEIGHTS_DIR = os.path.join(RUNS_DIR, "weights")
LOG_DIR = os.path.join(RUNS_DIR, "logs")
FIG_DIR = os.path.join(RUNS_DIR, "figs")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------
# Dataset / Model Config
# ---------------------

# NUM_CLASSES = 21         # VOC: 20 类 + 背景
# IGNORE_INDEX = 255
NUM_CLASSES = 19
IGNORE_INDEX = 255

# IMAGE_SIZE = 512         # 统一 resize/crop 尺寸
# IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 4

IMAGE_SIZE = (512, 1024)      # 如果显存够

# 预训练 SegFormer 模型名称
SEGFORMER_PRETRAINED_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"

# ---------------------
# Train Hyper-Params
# ---------------------

LR = 6e-5
WEIGHT_DECAY = 0.01

# 学习率调度器
LR_STEP_SIZE = 10        # 每 10 epoch 降一次 LR
LR_GAMMA = 0.5

NUM_EPOCHS = 40
PRINT_FREQ = 50



# 随机种子
SEED = 42

# device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# config.py
RANDOM_SCALE_MIN = 0.5
RANDOM_SCALE_MAX = 2.0

