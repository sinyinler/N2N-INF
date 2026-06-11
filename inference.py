"""
全图分块滑窗推理（eval 与 train 全图预览共用）。

块大小 = 训练 crop（坐标归一化口径一致）；相邻块 50% 重叠 + 2D Hann 羽化加权，
消除接缝。每块对中心帧做一次 SINF 前向，按 Hann 权重累加融合。
"""

from __future__ import annotations

import torch


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    s = list(range(0, size - tile + 1, stride))
    if s[-1] != size - tile:
        s.append(size - tile)
    return s


def _hann2d(h: int, w: int) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False).clamp_min(1e-3)
    wx = torch.hann_window(w, periodic=False).clamp_min(1e-3)
    return torch.outer(wy, wx)


@torch.no_grad()
def tiled_denoise(model, window_full: torch.Tensor, t_coords: torch.Tensor,
                  tile: int = 256, overlap: int | None = None, device: str = "cuda") -> torch.Tensor:
    """window_full: (T,1,H,W) log 域；返回中心帧去噪 (H,W) log 域。

    overlap 默认取 tile//2（50% 重叠），配合 Hann 羽化基本消除接缝。
    传入的 model 应是底层 SINF（非 DataParallel 包装），调用方负责 eval 模式。
    """
    T, _, H, W = window_full.shape
    tile = min(tile, H, W)
    if overlap is None:
        overlap = tile // 2
    overlap = min(overlap, tile - 1)
    stride = max(1, tile - overlap)

    out = torch.zeros(H, W)
    acc = torch.zeros(H, W)
    blend = _hann2d(tile, tile)

    dh, dw = max(H - 1, 1), max(W - 1, 1)
    for y in _starts(H, tile, stride):
        for x in _starts(W, tile, stride):
            patch = window_full[:, :, y:y + tile, x:x + tile].unsqueeze(0).to(device)  # (1,T,1,tile,tile)
            tc = t_coords.unsqueeze(0).to(device)                                       # (1,T)
            # 该 tile 在整图里的绝对归一化坐标（与训练口径一致）
            box = torch.tensor([[2 * y / dh - 1, 2 * (y + tile - 1) / dh - 1,
                                 2 * x / dw - 1, 2 * (x + tile - 1) / dw - 1]],
                               dtype=torch.float32, device=device)
            pred = model(patch, tc, box)[0, 0].float().cpu()                            # (tile,tile)
            out[y:y + tile, x:x + tile] += pred * blend
            acc[y:y + tile, x:x + tile] += blend

    return out / acc.clamp_min(1e-6)
