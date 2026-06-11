"""
逐帧特征提取 backbone（复用现有 model/denoiser.py）。

把你已有的单帧去噪网络降级为"特征器"：保留 Encoder→Bridge→Decoder，
**去掉最后的 1×1 输出头**（denoiser.py 里的 Transformer_unit），
输出全分辨率的多通道特征图，供 INF 头 / ITE / TSGM 挂靠。

  输入：单帧 (B, 1, H, W)        或 帧窗 (B, T, 1, H, W)
  输出：特征 (B, C_f, H, W)       或      (B, T, C_f, H, W)
  其中 C_f = 16（decoder_3 的原生通道数，对应轻量配置 16-32-64-80）。

说明：
- 这里先用**共享权重**的单个 backbone 处理窗内每一帧；论文"双向编码器
  Ff/Fb"的具体接法（两套权重 vs 共享+前后向聚合）留到 sinf.py 组装时
  与用户对齐后再定（见 experiment_log.md 待对齐项）。
- 容量沿用现有轻量配置；论文 32-64-128-128-128 为对比实验 E4。
"""

from __future__ import annotations

import torch
from torch import nn

from model.denoiser import Encoder, Bridge, Decoder


class FrameEncoder(nn.Module):
    """单帧 -> 16 通道全分辨率特征图。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(input_channels=in_channels)
        self.bridge = Bridge()
        self.decoder = Decoder()
        self.out_channels = 16  # decoder_3 的输出通道

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> feat: (B, 16, H, W)。"""
        out_1, out_2, out_3 = self.encoder(x)
        bridge = self.bridge(out_3)
        feat = self.decoder(bridge, out_1, out_2, out_3)  # (B, 16, H, W)
        return feat

    def forward_window(self, window: torch.Tensor) -> torch.Tensor:
        """window: (B, T, 1, H, W) -> feats: (B, T, 16, H, W)。

        把时间维折叠进 batch 一次性过 backbone（共享权重），再展开。
        """
        B, T, C, H, W = window.shape
        x = window.reshape(B * T, C, H, W)
        feat = self.forward(x)                      # (B*T, 16, H, W)
        return feat.reshape(B, T, self.out_channels, H, W)


if __name__ == "__main__":
    # 形状自检：(B=2, T=5, 1, 128,128) -> (2, 5, 16, 128,128)
    net = FrameEncoder(in_channels=1)
    dummy = torch.randn(2, 5, 1, 128, 128)
    out = net.forward_window(dummy)
    print("window in :", tuple(dummy.shape))
    print("feats out :", tuple(out.shape))
    assert out.shape == (2, 5, 16, 128, 128), "形状不符！"
    n = sum(p.numel() for p in net.parameters())
    print(f"backbone 参数量: {n/1e6:.4f} M")
    print("OK")
