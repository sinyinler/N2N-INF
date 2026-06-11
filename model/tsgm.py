"""
TSGM — 时间感知空间图模块（Time-Aware Spatial Graph Module）。对应论文 §IV-D / 图 6。

对中心帧每个像素，在 ws×ws 局部窗内、跨时间窗 [t-K,t+K] 用 cross-graph attention
聚合语义一致的像素，输出对齐融合后的中心帧逐像素特征 z*_t（式 22）。
不依赖光流、不显式 warp——这是本方案相对普通 N2N 的核心增益。

两阶段（图 6）：
  - Time alignment：用基于相对帧偏移的可学习时间偏置注入注意力（时间权重）；
  - Space alignment：窗口化跨帧多头注意力建立跨帧对应。

实现约定（论文未写死处，已记入 experiment_log）：
  - 采用 **非重叠 ws×ws 窗口划分**（Swin 式）做局部图，比逐像素 N(x,y) 邻域轻得多；
  - **双向**通过"中心帧 query 同时 attend 窗内全部 T 帧（含过去/未来）"实现，
    不另搞两套编码器权重（与用户确认的共享方案一致）；
  - 残差从中心帧引出，后接 LayerNorm + 卷积 FFN。

输入：窗特征 x (B, T, C, H, W)
输出：对齐融合后的中心帧特征 (B, C, H, W)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """对通道维做 LayerNorm 的 2D 版本。"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class CrossFrameWindowAttention(nn.Module):
    """窗口化跨帧多头注意力：中心帧 query，窗内全部 T 帧 key/value。"""

    def __init__(self, dim: int, window: int = 7, heads: int = 4, temporal_len: int = 5):
        super().__init__()
        assert dim % heads == 0, f"dim({dim}) 必须能被 heads({heads}) 整除"
        self.dim = dim
        self.window = window
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        # time alignment：每个相对帧偏移、每个 head 一个可学习时间偏置
        self.temporal_bias = nn.Parameter(torch.zeros(temporal_len, heads))

    def forward(self, x: torch.Tensor, center_idx: int) -> torch.Tensor:
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        ws = self.window

        # pad 到 ws 的整数倍
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x.reshape(B * T, C, H, W), (0, pad_r, 0, pad_b)).reshape(B, T, C, H + pad_b, W + pad_r)
        Hp, Wp = H + pad_b, W + pad_r
        nh, nw = Hp // ws, Wp // ws
        N = ws * ws

        # 窗口划分 -> (B*nh*nw, T, N, C)
        xw = x.view(B, T, C, nh, ws, nw, ws).permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        xw = xw.view(B * nh * nw, T, N, C)
        Bn = xw.shape[0]

        q = self.to_q(xw[:, center_idx])                      # (Bn, N, C)
        k, v = self.to_kv(xw).chunk(2, dim=-1)                # 各 (Bn, T, N, C)

        q = q.view(Bn, N, self.heads, self.head_dim).permute(0, 2, 1, 3)          # (Bn,h,N,d)
        k = k.view(Bn, T, N, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)    # (Bn,h,T,N,d)
        v = v.view(Bn, T, N, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)

        attn = torch.einsum("bhnd,bhtmd->bhntm", q, k) * self.scale  # (Bn,h,N,T,N)
        attn = attn + self.temporal_bias.t().view(1, self.heads, 1, T, 1)  # 时间偏置
        attn = attn.reshape(Bn, self.heads, N, T * N).softmax(dim=-1)

        v = v.reshape(Bn, self.heads, T * N, self.head_dim)
        out = attn @ v                                        # (Bn,h,N,d)
        out = out.permute(0, 2, 1, 3).reshape(Bn, N, C)
        out = self.proj(out)

        # 窗口还原 -> (B,C,Hp,Wp) -> 裁回 (B,C,H,W)
        out = out.view(B, nh, nw, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, Hp, Wp)[:, :, :H, :W]
        return out


class TimeAwareSpatialGraphModule(nn.Module):
    def __init__(self, dim: int, window: int = 7, temporal_radius: int = 2, heads: int = 4):
        super().__init__()
        self.temporal_radius = temporal_radius
        temporal_len = 2 * temporal_radius + 1

        self.norm1 = LayerNorm2d(dim)
        self.attn = CrossFrameWindowAttention(dim, window=window, heads=heads, temporal_len=temporal_len)
        self.norm2 = LayerNorm2d(dim)
        # 卷积 FFN（图 6 的 Convolution + Feed-forward）
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C,H,W)，中心帧取窗中点
        B, T, C, H, W = x.shape
        center = T // 2

        x_norm = self.norm1(x.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
        attn_out = self.attn(x_norm, center)        # (B,C,H,W)
        y = x[:, center] + attn_out                 # 残差从中心帧引出
        y = y + self.ffn(self.norm2(y))             # FFN 残差
        return y


if __name__ == "__main__":
    # 用非 7 整数倍的 H,W 验证 padding；C 必须能被 heads 整除
    tsgm = TimeAwareSpatialGraphModule(dim=32, window=7, temporal_radius=2, heads=4)
    x = torch.randn(2, 5, 32, 50, 60)              # (B,T,C,H,W)，50/60 非 7 倍数
    y = tsgm(x)
    print("x in  :", tuple(x.shape))
    print("y out :", tuple(y.shape), "(应为 (2,32,50,60))")
    assert y.shape == (2, 32, 50, 60)
    print(f"TSGM 参数量: {sum(p.numel() for p in tsgm.parameters())/1e6:.4f} M")
    print("OK")
