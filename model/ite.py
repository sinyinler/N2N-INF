"""
ITE — 隐式时间嵌入（Implicit Temporal Embedding）。

对应论文 §IV-C。把归一化时间戳 t_tilde=(t-1)/(T-1) 映射成连续时间特征：
  - 3 层 sin-MLP（SIREN 式激活），hidden=64（式 33）；
  - 可选多倍频带 sin/cos 堆叠（式 34）；
  - 时间特征沿空间广播后与编码特征拼接（式 35）；
  - 前向/后向传播分别注入，最后融合（式 38）。

作用：给每个空间位置注入全局时间位置，提供连续时间先验，替代光流。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations

import torch
from torch import nn


class ImplicitTemporalEmbedding(nn.Module):
    def __init__(self, layers: int = 3, hidden: int = 64):
        super().__init__()
        raise NotImplementedError("骨架阶段，待实现")

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """t_norm: 归一化时间戳 -> 时间嵌入向量 e_t。"""
        raise NotImplementedError("骨架阶段，待实现")
