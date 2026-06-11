"""
SINF 顶层组装（BSN→N2N 版本）。对应论文 §IV-A 总 pipeline。

数据流：
    window (B,T,1,H,W) + t_coords (B,T)
      │  FrameEncoder（共享逐帧 backbone）           -> feats (B,T,16,H,W)
      │  INFHead（BS-INF 去盲点，逐帧 + Fourier 坐标）-> z_τ  (B,T,16,H,W)   式21
      │  ITE（SIREN 时间嵌入，广播拼接）             -> H_t  (B,T,32,H,W)   式35
      │  TSGM（窗口化跨帧图注意力，时空对齐）         -> z*_t (B,32,H,W)     式22
      │  FinalINFHead F_Θ（Fourier(中心帧坐标)+z*_t）-> Ĉ_t  (B,1,H,W)      式23
      ▼
    中心帧干净估计 Ĉ_t

与论文不同：无盲点 mask、无蒸馏；监督走 N2N（标签帧由 dataset 保证在输入窗外）。
双向通过 TSGM 的对称时间注意力实现（共享 backbone，不另搞两套权重）。
"""

from __future__ import annotations

import torch
from torch import nn

from model.backbone import FrameEncoder
from model.inf_head import INFHead, FourierPositionEncoding
from model.ite import ImplicitTemporalEmbedding
from model.tsgm import TimeAwareSpatialGraphModule


class FinalINFHead(nn.Module):
    """最终重建头 F_Θ（式 23）：F_Θ(γ(x,y,t_center), z*_t) -> Ĉ_t。

    论文实现：4 层 MLP、hidden=128、ReLU。用 1×1 conv 做逐像素 MLP。
    """

    def __init__(self, in_dim: int, fourier_bands: int = 10, hidden: int = 128,
                 layers: int = 4, out_channels: int = 1):
        super().__init__()
        self.fourier = FourierPositionEncoding(fourier_bands)
        d = in_dim + self.fourier.out_dim

        net: list[nn.Module] = [nn.Conv2d(d, hidden, 1), nn.ReLU(inplace=True)]
        for _ in range(layers - 2):
            net += [nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True)]
        net += [nn.Conv2d(hidden, out_channels, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, z_star: torch.Tensor, t_center: torch.Tensor) -> torch.Tensor:
        _B, _C, H, W = z_star.shape
        gamma = self.fourier(t_center, H, W)            # (B, Fdim, H, W)
        return self.net(torch.cat([gamma, z_star], dim=1))


class SINF(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        fourier_bands: int = 10,
        inf_out: int = 16,
        ite_layers: int = 3,
        ite_hidden: int = 64,
        ite_out: int = 16,
        ite_w0: float = 30.0,
        tsgm_window: int = 7,
        tsgm_radius: int = 2,
        tsgm_heads: int = 4,
        final_hidden: int = 128,
        final_layers: int = 4,
    ):
        super().__init__()
        self.backbone = FrameEncoder(in_channels=in_channels)
        feat_c = self.backbone.out_channels             # 16

        self.inf_head = INFHead(in_channels=feat_c, fourier_bands=fourier_bands, out_channels=inf_out)
        self.ite = ImplicitTemporalEmbedding(layers=ite_layers, hidden=ite_hidden, out_dim=ite_out, w0=ite_w0)

        tsgm_dim = inf_out + ite_out                    # 拼接后通道（需被 heads 整除）
        assert tsgm_dim % tsgm_heads == 0, f"tsgm_dim({tsgm_dim}) 需被 heads({tsgm_heads}) 整除"
        self.tsgm = TimeAwareSpatialGraphModule(
            dim=tsgm_dim, window=tsgm_window, temporal_radius=tsgm_radius, heads=tsgm_heads
        )
        self.final = FinalINFHead(
            in_dim=tsgm_dim, fourier_bands=fourier_bands,
            hidden=final_hidden, layers=final_layers, out_channels=out_channels,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "SINF":
        """从 configs/default.yaml 的 dict 构建。"""
        m = cfg["model"]
        return cls(
            in_channels=m.get("in_channels", 1),
            out_channels=m.get("out_channels", 1),
            fourier_bands=m.get("fourier_bands", 10),
            ite_layers=m.get("ite", {}).get("layers", 3),
            ite_hidden=m.get("ite", {}).get("hidden", 64),
            tsgm_window=m.get("tsgm", {}).get("window", 7),
            tsgm_radius=m.get("tsgm", {}).get("temporal_radius", 2),
            tsgm_heads=m.get("tsgm", {}).get("heads", 4),
            final_hidden=m.get("inf_head", {}).get("hidden", 128),
            final_layers=m.get("inf_head", {}).get("layers", 4),
        )

    def forward(self, window: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
        """window: (B,T,1,H,W)，t_coords: (B,T) -> 中心帧 Ĉ_t: (B,out,H,W)。"""
        B, T, _C, H, W = window.shape

        feats = self.backbone.forward_window(window)             # (B,T,16,H,W)
        feats_flat = feats.reshape(B * T, -1, H, W)
        t_flat = t_coords.reshape(B * T)

        z = self.inf_head(feats_flat, t_flat)                    # (B*T,16,H,W)
        e_t = self.ite(t_flat)                                   # (B*T, ite_out)
        h = ImplicitTemporalEmbedding.broadcast_concat(z, e_t)   # (B*T,32,H,W)
        h = h.reshape(B, T, -1, H, W)

        z_star = self.tsgm(h)                                    # (B,32,H,W)
        t_center = t_coords[:, T // 2]                           # (B,)
        out = self.final(z_star, t_center)                       # (B,out,H,W)
        return out


if __name__ == "__main__":
    net = SINF()
    window = torch.randn(2, 5, 1, 64, 64)
    t_coords = torch.rand(2, 5)
    out = net(window, t_coords)
    print("window  :", tuple(window.shape))
    print("output  :", tuple(out.shape), "(应为 (2,1,64,64))")
    assert out.shape == (2, 1, 64, 64)
    total = sum(p.numel() for p in net.parameters())
    print(f"SINF 总参数量: {total/1e6:.4f} M")
    # 反向传播自检
    out.mean().backward()
    print("backward OK")
