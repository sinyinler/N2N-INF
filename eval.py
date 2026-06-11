"""
评估 / 出图入口。全图滑窗推理 + 可视化。

  python eval.py --ckpt results/sinf_last.pth --root <序列目录> --center 500

策略：
  - 对某中心帧 t，取窗 {t-K..t+K} 全分辨率帧，log1p；
  - **分块推理**：块大小=tile_size（与训练 crop 一致，保证坐标归一化口径一致），
    相邻块 overlap，用 2D Hann 羽化加权融合，消除接缝；
  - 输出中心帧去噪结果 Ĉ_t（log 域），存 npy + 灰度 png；
  - 细小血管 ROI 局部放大可用 utils/calc_image_metrics.py 交互式框选（规范第4条）。

注意：模型坐标在 INFHead 里按当前块 H,W 归一化到 [-1,1]，训练用的是 crop 块，
故 eval 必须用相同块大小分块推理才与训练口径一致（见 experiment_log 记录）。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from dataset.video_pair_dataset import _find_sequence_dirs, _list_frames, _load_2d, _log1p
from model.sinf import SINF


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
                  tile: int, overlap: int, device: str) -> torch.Tensor:
    """window_full: (T,1,H,W) log 域；返回中心帧去噪 (H,W) log 域。"""
    T, _, H, W = window_full.shape
    tile = min(tile, H, W)
    stride = max(1, tile - overlap)

    out = torch.zeros(H, W)
    acc = torch.zeros(H, W)
    blend = _hann2d(tile, tile)

    for y in _starts(H, tile, stride):
        for x in _starts(W, tile, stride):
            patch = window_full[:, :, y:y + tile, x:x + tile].unsqueeze(0).to(device)  # (1,T,1,tile,tile)
            tc = t_coords.unsqueeze(0).to(device)                                       # (1,T)
            pred = model(patch, tc)[0, 0].cpu()                                         # (tile,tile)
            out[y:y + tile, x:x + tile] += pred * blend
            acc[y:y + tile, x:x + tile] += blend

    return out / acc.clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="单条序列目录（直接含帧文件）")
    ap.add_argument("--center", type=int, default=None, help="中心帧索引，默认取序列中点")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inverse_log", action="store_true", help="额外存 expm1 还原到强度域的 npy")
    ap.add_argument("--out_dir", default="results/eval")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    K = cfg["model"].get("tsgm", {}).get("temporal_radius", 2)
    tile = cfg["eval"].get("tile_size", 256)
    overlap = cfg["eval"].get("tile_overlap", 32)

    # 找到序列帧
    seq_dirs = _find_sequence_dirs(args.root)
    seq_dir = args.root if _list_frames(args.root) else (seq_dirs[0] if seq_dirs else None)
    if seq_dir is None:
        raise RuntimeError(f"{args.root} 下未找到帧文件")
    files = _list_frames(seq_dir)
    n = len(files)
    center = args.center if args.center is not None else n // 2
    if not (K <= center < n - K):
        raise ValueError(f"center={center} 越界，需在 [{K}, {n - 1 - K}]")

    # 模型
    ckpt = torch.load(args.ckpt, map_location=device)
    model = SINF.from_config(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 窗 + log1p
    idxs = list(range(center - K, center + K + 1))
    frames = [_log1p(torch.from_numpy(_load_2d(os.path.join(seq_dir, files[i]))).float().unsqueeze(0)) for i in idxs]
    window_full = torch.stack(frames, dim=0)  # (T,1,H,W)
    denom = max(n - 1, 1)
    t_coords = torch.tensor([i / denom for i in idxs], dtype=torch.float32)

    den = tiled_denoise(model, window_full, t_coords, tile, overlap, device)  # (H,W) log 域

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"center{center:04d}")
    np.save(base + "_denoised_log.npy", den.numpy())
    if args.inverse_log:
        np.save(base + "_denoised.npy", np.expm1(den.numpy()))

    # 灰度对比图：中心输入 vs 去噪（log 域，分位显示）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center_in = window_full[K, 0].numpy()
    den_np = den.numpy()
    vmin = float(np.percentile(np.concatenate([center_in.ravel(), den_np.ravel()]), 1))
    vmax = float(np.percentile(np.concatenate([center_in.ravel(), den_np.ravel()]), 99))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, im, ti in zip(axes, [center_in, den_np], ["center input (log)", "denoised (log)"]):
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(ti)
        ax.axis("off")
    fig.savefig(base + "_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] center={center} 输出 -> {base}_*  (tile={tile}, overlap={overlap})")


if __name__ == "__main__":
    main()
