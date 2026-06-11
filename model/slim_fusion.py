"""精简多帧去噪器（L2R 融合版）。

= 你证明好用的卷积 backbone（逐帧特征）+ TSGM（跨帧对齐融合）+ 卷积残差头。
**去掉坐标 INF / 逐像素 F_Θ**（已实测在 BFI 上有害）。

自监督由外部 L2R 提供：训练时把**中心帧**重腐蚀成 y₁ 放进窗中心，
邻帧（在去相关的偏移上，噪声与中心独立）当上下文，输出去噪中心帧、监督回中心帧。

输入：window (B,T,1,H,W)，中心帧在 index T//2（训练时已被重腐蚀）。
输出：去噪后的中心帧 (B,out,H,W)。
"""
from __future__ import annotations

import torch
from torch import nn

from model.backbone import FrameEncoder
from model.tsgm import TimeAwareSpatialGraphModule


class SlimFusion(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 tsgm_window: int = 7, tsgm_radius: int = 2, tsgm_heads: int = 4,
                 residual: bool = True):
        super().__init__()
        self.backbone = FrameEncoder(in_channels=in_channels)
        c = self.backbone.out_channels                      # 16
        assert c % tsgm_heads == 0, f"feat_c({c}) 需被 heads({tsgm_heads}) 整除"
        self.tsgm = TimeAwareSpatialGraphModule(dim=c, window=tsgm_window,
                                                temporal_radius=tsgm_radius, heads=tsgm_heads)
        self.head = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, padding_mode="reflect"), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
        )
        self.residual = residual

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        B, T, _C, H, W = window.shape
        feats = self.backbone.forward_window(window)        # (B,T,16,H,W)
        z = self.tsgm(feats)                                 # (B,16,H,W) 融合后的中心特征
        r = self.head(z)                                     # (B,out,H,W)
        if self.residual:
            return window[:, T // 2] - r                     # 残差：中心输入减预测噪声
        return r


if __name__ == "__main__":
    net = SlimFusion()
    window = torch.randn(2, 5, 1, 64, 64)
    out = net(window)
    print("window:", tuple(window.shape), "-> out:", tuple(out.shape))
    assert out.shape == (2, 1, 64, 64)
    out.mean().backward()
    print(f"参数量 {sum(p.numel() for p in net.parameters())/1e6:.4f}M  backward OK")
