"""
INF 空间解码头（BS-INF 去掉盲点后的版本）。

对应论文 §IV-B（Blind-Spot Implicit Neural Field），但**删除盲点 mask**：
  - 删除 MaskConv 与掩码坐标场（原文式 17/29）；
  - 保留：DilatedConv 局部特征聚合（式 26）
          + 坐标归一化到 [-1,1]（式 27）
          + Fourier 多倍频带位置编码（式 28，频带数 = configs.model.fourier_bands）
          + 坐标 INF 的 MLP（式 30）
          + 局部特征与坐标 INF 特征的融合 MLP（式 31）。

输入：逐帧编码特征 F_tau (B, C, H, W) + 归一化坐标 (x_tilde, y_tilde, t_tilde)
输出：每像素隐表示 / 重建值。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations

import torch
from torch import nn


class FourierPositionEncoding(nn.Module):
    """坐标 -> 多倍频带 [sin(2^b·π·p), cos(2^b·π·p)]_{b=0..B}（式 28）。"""

    def __init__(self, num_bands: int = 10):
        super().__init__()
        self.num_bands = num_bands
        raise NotImplementedError("骨架阶段，待实现")


class INFHead(nn.Module):
    """坐标 INF + 局部 dilated 特征 融合的去噪空间头（无盲点）。"""

    def __init__(self, in_channels: int, fourier_bands: int = 10):
        super().__init__()
        raise NotImplementedError("骨架阶段，待实现")

    def forward(self, feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("骨架阶段，待实现")
