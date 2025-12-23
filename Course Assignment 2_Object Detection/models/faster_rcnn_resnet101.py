# models/faster_rcnn_resnet101.py

import torch
from torch import nn
from torchvision.models import resnet101
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_faster_rcnn_resnet101(num_classes=21, pretrained_backbone=True):
    # 1. 加载 ResNet101 backbone
    backbone = resnet101(pretrained=pretrained_backbone)

    # --- 2. 使用 IntermediateLayerGetter 提取层输出 ---
    # ResNet101 中的关键 block 名字就是 layer1,2,3,4（这是正确名字！）
    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }

    # 将 backbone 转换为能输出特征图的结构
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # --- 3. FPN: 四层输入通道 ---
    in_channels_list = [256, 512, 1024, 2048]  # ResNet101 的输出通道
    out_channels = 256

    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

    # --- 4. 合成带 FPN 的 backbone 模块 ---
    class BackboneWithFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = out_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    backbone_with_fpn = BackboneWithFPN(backbone, fpn)

    # --- 5. anchor sizes 自定义（默认也可以） ---
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # --- 6. 构建 Faster R-CNN ---
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
    )

    return model
