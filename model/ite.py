"""
ITE — 隐式时间嵌入（Implicit Temporal Embedding）。对应论文 §IV-C。

把归一化时间戳 t̃（dataset 已给，[0,1]）经 **SIREN 正弦 MLP** 映射成连续时间特征：
  - 3 层 sin-MLP，hidden=64（式 33：e_t = W2·sin(W1·t̃+b1)+b2）；
  - SIREN 周期激活提供高频时间表达 + 处处可微（论文图 3）；
  - 时间特征沿空间广播，与逐像素特征拼接（式 35：H_t=[z_τ, e_t]）。

作用：给每个空间位置注入连续时间位置，提供平滑时间先验，替代光流。

超参：w0 控制频率范围（论文未写死，取 SIREN 默认 30）；out_dim 为时间嵌入维度。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    """SIREN 正弦层：sin(w0 · (W x + b))，附 SIREN 初始化。"""

    def __init__(self, in_features: int, out_features: int, w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._siren_init(in_features)

    def _siren_init(self, in_features: int):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class ImplicitTemporalEmbedding(nn.Module):
    def __init__(self, layers: int = 3, hidden: int = 64, out_dim: int = 64, w0: float = 30.0):
        super().__init__()
        assert layers >= 2, "至少 2 层（首正弦层 + 末线性层）"
        self.out_dim = out_dim

        net: list[nn.Module] = [SineLayer(1, hidden, w0=w0, is_first=True)]
        for _ in range(layers - 2):
            net.append(SineLayer(hidden, hidden, w0=w0))
        net.append(nn.Linear(hidden, out_dim))  # 末层线性（式 33 的 W2·sin(...)+b2）
        self.net = nn.Sequential(*net)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        """t_norm: (N,) 归一化时间戳 -> 时间嵌入 e_t: (N, out_dim)。"""
        return self.net(t_norm.view(-1, 1))

    @staticmethod
    def broadcast_concat(feat: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """把时间嵌入沿空间广播后与特征拼接（式 35）。
        feat (N,C,H,W) + e_t (N,d) -> (N, C+d, H, W)。"""
        N, _C, H, W = feat.shape
        e = e_t.view(N, -1, 1, 1).expand(-1, -1, H, W)
        return torch.cat([feat, e], dim=1)


if __name__ == "__main__":
    ite = ImplicitTemporalEmbedding(layers=3, hidden=64, out_dim=64)
    t_norm = torch.rand(10)                       # N=B*T
    e_t = ite(t_norm)
    feat = torch.randn(10, 16, 128, 128)
    fused = ImplicitTemporalEmbedding.broadcast_concat(feat, e_t)
    print("e_t      :", tuple(e_t.shape))
    print("fused    :", tuple(fused.shape), "(应为 16+64=80 通道)")
    assert e_t.shape == (10, 64) and fused.shape == (10, 80, 128, 128)
    print(f"ITE 参数量: {sum(p.numel() for p in ite.parameters())/1e6:.4f} M")
    print("OK")
