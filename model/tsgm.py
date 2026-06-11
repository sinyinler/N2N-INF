"""
TSGM — 时间感知空间图模块（Time-Aware Spatial Graph Module）。

对应论文 §IV-D（图 6）。两个紧耦合阶段：
  1) Time alignment：把 ITE 的连续时间嵌入注入逐像素特征；
  2) Space alignment：在 ws×ws 局部窗内做 cross-graph attention，
     建立跨帧对应、聚合语义一致像素，替代噪声敏感的光流。

配置（见 configs）：window=7（7x7 窗），temporal_radius=2（对齐输入窗半径，
论文写 3，为与 T=5 自洽改 2），heads=4。
边界截断时做 boundary renormalization。

作用：这是本方案相对"普通 N2N"最大的增益——无光流、抗噪的跨帧对齐。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations

import torch
from torch import nn


class TimeAwareSpatialGraphModule(nn.Module):
    def __init__(self, window: int = 7, temporal_radius: int = 2, heads: int = 4):
        super().__init__()
        raise NotImplementedError("骨架阶段，待实现")

    def forward(self, feats, time_embed):
        """feats: 窗内各帧逐像素特征；time_embed: ITE 时间嵌入 -> 对齐融合后的特征。"""
        raise NotImplementedError("骨架阶段，待实现")
