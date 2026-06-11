# -*- coding: utf-8 -*-
"""单调重腐蚀器 h（移植自 L2R 项目，arXiv:2603.25869v1 Eq.20）。

  h(w) = k · normalize( conv_k( normalize( mMLP(w) ) ) )

mMLP 单调（正权重 + softplus）→ 逆 CDF 形状；conv_k 捕捉空间相关（散斑用 ksize=3）；
k=softplus(log_k) 控幅度。训练时 α(τ)=1，幅度由 k 学。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicLinear(nn.Module):
    """正权重线性层(softplus(w)≥0) → 复合后整体单调递增。"""

    def __init__(self, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i) * 0.1)
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x):
        return F.linear(x, F.softplus(self.w), self.b)


class Recorruptor(nn.Module):
    def __init__(self, hidden=64, ksize=3):
        super().__init__()
        self.net = nn.Sequential(
            MonotonicLinear(1, hidden), nn.Softplus(),
            MonotonicLinear(hidden, hidden), nn.Softplus(),
            MonotonicLinear(hidden, 1),
        )
        self.ksize = ksize
        ker = torch.zeros(1, 1, ksize, ksize)
        ker[0, 0, ksize // 2, ksize // 2] = 1.0     # 初始为恒等核(i.i.d.)
        self.kernel = nn.Parameter(ker)
        self.log_k = nn.Parameter(torch.tensor(0.0))   # k=softplus(log_k)，重腐蚀幅度

    def set_k(self, k):
        with torch.no_grad():
            self.log_k.fill_(math.log(math.expm1(max(k, 1e-3))))

    def _shape(self, w):
        s = w.shape
        o = self.net(w.reshape(-1, 1)).reshape(s)
        o = (o - o.mean()) / (o.std() + 1e-6)
        ker = self.kernel / (self.kernel.flatten().norm() + 1e-6)
        o = F.conv2d(o, ker, padding=self.ksize // 2)
        o = (o - o.mean()) / (o.std() + 1e-3)
        return o

    def forward(self, w):
        return F.softplus(self.log_k) * self._shape(w)


def id_pretrain(h, steps=400, batch=8, size=64, lr=1e-3, device="cpu", log=None):
    """Id-pretrain：把 mMLP 预训练成恒等(h 起步≈标准高斯)，稳住 min-max 初期。"""
    opt = torch.optim.Adam(h.net.parameters(), lr=lr)
    h.train()
    for it in range(1, steps + 1):
        w = torch.randn(batch, 1, size, size, device=device)
        out = h._shape(w)
        wn = (w - w.mean()) / (w.std() + 1e-6)
        loss = (out - wn).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log is not None and (it % 100 == 0 or it == 1):
            log(f"[id-pretrain] {it:4d}/{steps} loss={loss.item():.5f}")
    return h
