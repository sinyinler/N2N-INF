"""
INF 空间解码头（BS-INF 去掉盲点后的版本）。

对应论文 §IV-B，但**删除盲点 mask**：
  - 局部分支（式 26）：U_τ = DilatedConv(F_τ)（去掉 MaskConv），扩大感受野；
  - 坐标 INF 分支：
      * 坐标归一化到 [-1,1]（式 27）；
      * Fourier 多倍频带编码（式 28）：γ(p)=[sin(2^b·π·p),cos(2^b·π·p)]，p=(x̃,ỹ,t̃)；
      * **删除式 29 的二值盲点 mask**（盲点对散斑失效，N2N 不需要）；
      * INF MLP（式 30）：v_τ = f_θ(γ(x,y,τ), F_τ(x,y))，用 1×1 conv 实现逐像素 MLP；
  - 融合 MLP（式 31）：z_τ = MLP([U_τ, v_τ])。

输入：逐帧特征 F_τ (N, C_in, H, W) + 该帧归一化时间戳 t_norm (N,)
输出：逐像素融合特征 z_τ (N, out_channels, H, W)
（N 可以是 B 或 B*T；sinf.py 把时间维折进 batch 后调用。）
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierPositionEncoding(nn.Module):
    """坐标 (x̃,ỹ,t̃) -> 多倍频带 [sin(2^b·π·p), cos(2^b·π·p)]_{b=0..B-1}（式 28）。"""

    def __init__(self, num_bands: int = 10):
        super().__init__()
        self.num_bands = num_bands
        freqs = (2.0 ** torch.arange(num_bands)) * math.pi  # (num_bands,)
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 * num_bands * 3  # sin/cos × bands × (x,y,t)

    def forward(self, t_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # t_norm: (N,) 取值 [0,1]（来自 dataset 的全局归一化时间戳）
        N = t_norm.shape[0]
        device, dtype = t_norm.device, t_norm.dtype

        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")          # (H,W)
        gx = gx.expand(N, H, W)
        gy = gy.expand(N, H, W)
        gt = (t_norm.view(N, 1, 1) * 2.0 - 1.0).expand(N, H, W)  # [0,1]->[-1,1]

        coords = torch.stack([gx, gy, gt], dim=1)               # (N,3,H,W)
        c = coords.unsqueeze(2) * self.freqs.view(1, 1, -1, 1, 1)  # (N,3,bands,H,W)
        emb = torch.cat([torch.sin(c), torch.cos(c)], dim=2)    # (N,3,2*bands,H,W)
        return emb.reshape(N, self.out_dim, H, W)


class INFHead(nn.Module):
    """局部 dilated 特征 + 坐标 INF 融合的连续空间头（无盲点）。"""

    def __init__(self, in_channels: int = 16, fourier_bands: int = 10, out_channels: int = 16):
        super().__init__()
        self.out_channels = out_channels
        self.fourier = FourierPositionEncoding(fourier_bands)

        # 局部分支（式 26，去掉 MaskConv）：两层 dilated conv 扩大感受野
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=2, dilation=2, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        # 坐标 INF MLP f_θ（式 30）：1×1 conv = 逐像素 MLP
        inf_in = self.fourier.out_dim + in_channels
        self.coord_mlp = nn.Sequential(
            nn.Conv2d(inf_in, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
        )

        # 融合 MLP（式 31）：concat[U_τ, v_τ] -> z_τ
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 32, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, feat: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        N, _C, H, W = feat.shape
        U = self.local(feat)                                  # (N,32,H,W)
        gamma = self.fourier(t_norm, H, W)                    # (N,Fdim,H,W)
        v = self.coord_mlp(torch.cat([gamma, feat], dim=1))   # (N,32,H,W)
        z = self.fuse(torch.cat([U, v], dim=1))               # (N,out,H,W)
        return z


if __name__ == "__main__":
    head = INFHead(in_channels=16, fourier_bands=10, out_channels=16)
    feat = torch.randn(10, 16, 128, 128)          # N=B*T=10
    t_norm = torch.rand(10)                        # [0,1]
    z = head(feat, t_norm)
    print("feat in :", tuple(feat.shape))
    print("z out   :", tuple(z.shape), " Fourier dim =", head.fourier.out_dim)
    assert z.shape == (10, 16, 128, 128)
    print(f"INFHead 参数量: {sum(p.numel() for p in head.parameters())/1e6:.4f} M")
    print("OK")
