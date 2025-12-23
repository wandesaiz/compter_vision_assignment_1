# 导包
import torch
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import sys


import matplotlib.pyplot as plt
import logging



from datetime import datetime
import torch.optim as optim


import torch

# 加载模型
model = torch.load("convnet_SGD_CosineAnnealingLR.pth", map_location=torch.device('cpu'))

# 查看模型信息
print(model)
