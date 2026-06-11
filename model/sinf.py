"""
SINF 顶层组装（BSN→N2N 版本）。

把各模块按论文 §IV-A 的总 pipeline 串起来（原文式 between 23）：
    I_t (滑窗) --(Ff,Fb backbone + INF头)--> 特征
                --(ITE)--> 注入连续时间
                --(TSGM)--> 跨帧时空对齐
                --(F_Θ INF头)--> 重建中心帧 Ĉ_t

与论文不同：
  - backbone 复用 model/denoiser.py（降级为逐帧编码器，先用轻量配置）；
  - 无盲点 mask；监督走 N2N（标签帧由 dataset 保证在输入窗外）。

输入：滑窗 {I_{t-K},...,I_{t+K}}（K=2, T=5）+ 归一化时间坐标。
输出：中心帧干净估计 Ĉ_t。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations

import torch
from torch import nn


class SINF(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # cfg 来自 configs/default.yaml 的 model 段
        raise NotImplementedError("骨架阶段，待实现")

    def forward(self, window: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
        """window: (B, T, C, H, W) 输入帧窗 -> 中心帧去噪结果 (B, C, H, W)。"""
        raise NotImplementedError("骨架阶段，待实现")
