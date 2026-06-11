"""
多帧 N2N 数据集（B 路）。

把"两帧配对"改成"一个输入帧窗 + 一个被排除的标签帧"：
  - 输入窗：{t-K, ..., t, ..., t+K}，K = window_radius（默认 2，对应 T=5）；
  - N2N 标签：第 t±interval 帧，interval ∈ pair_intervals（默认 {7, 9}）；
  - ⚠️ N2N 正确性约束（方案①）：
      标签帧必须落在输入窗 [t-K, t+K] 之外，且窗内"最近帧"到标签的间隔 ≥ 5
      才满足去相关。由 interval ≥ K + 5 保证（K=2 → interval≥7）。
  - 配对只在同一序列文件夹内，绝不跨序列；
  - 同一个样本里，窗内所有帧 + 标签帧共用**同一处空间裁剪**，保证像素对应；
  - 强度变换：log1p（散斑方差稳定）。

本文件**自包含**：只依赖 utils/lbfreadnew.py，不 import dataset/data.py
（那份拷贝顶部引用了缺失的 utils/monotonic_vst.py，会 ImportError）。
learned-VST（对比实验 E3）以后补 monotonic_vst.py 时再接进来。
"""

from __future__ import annotations

import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.lbfreadnew import lbfreadnew


SUPPORTED_EXTS = (".npy", ".lbf")


# ------------------------------------------------------------------
# 一、小工具：自然排序 / 列帧 / 读单帧
# ------------------------------------------------------------------
def _natural_sort_key(name: str):
    """按数字块自然排序，忽略 '-'/'_' 分隔符，使 2-11_9 排在 2-11_10 前面。"""
    stem, ext = os.path.splitext(name.lower())
    parts = re.findall(r"\d+|[a-z]+", stem)
    key = [(0, int(p)) if p.isdigit() else (1, p) for p in parts]
    key.append((-1, ""))
    key.append((2, ext))
    return key


def _list_frames(folder: str) -> list[str]:
    """返回某文件夹内按自然序排好的受支持帧文件名（不含子目录）。"""
    if not os.path.isdir(folder):
        return []
    files = [
        name
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS
    ]
    return sorted(files, key=_natural_sort_key)


def _load_2d(path: str) -> np.ndarray:
    """读 .npy / .lbf -> 2D float32。多余的单通道维度自动压掉。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".lbf":
        arr = lbfreadnew(path)
    else:
        raise ValueError(f"不支持的帧文件类型: {path}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"期望 2D 帧，实际形状 {arr.shape}：{path}")
    return arr.astype(np.float32, copy=False)


def _find_sequence_dirs(root: str) -> list[str]:
    """递归找出所有"直接包含帧文件"的目录，每个这样的目录视作一条独立序列。

    兼容多种摆放：root/*.lbf、root/idx/*.lbf、root/idx/npy/*.npy、
    root/level/idx/npy/*.npy 都会各自成为一条序列。
    """
    seq_dirs = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        if _list_frames(dirpath):
            seq_dirs.append(dirpath)
    return sorted(seq_dirs)


def _log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


# ------------------------------------------------------------------
# 二、数据集
# ------------------------------------------------------------------
class VideoN2NDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        window_radius: int = 2,
        pair_intervals=(7, 9),
        crop_size: int = 512,
        intensity_transform: str = "log1p",
        random_crop: bool = True,
    ):
        self.root_dir = root_dir
        self.K = int(window_radius)
        self.intervals = tuple(int(i) for i in pair_intervals)
        self.crop_size = None if (crop_size is None or int(crop_size) <= 0) else int(crop_size)
        self.intensity_transform = str(intensity_transform).lower()
        self.random_crop = bool(random_crop)

        if self.intensity_transform not in {"log1p", "none"}:
            raise ValueError("intensity_transform 目前只支持 'log1p' 或 'none'")

        # 方案①去相关约束：窗内最近帧到标签间隔 = interval - K，需 ≥ 5
        if min(self.intervals) < self.K + 5:
            raise ValueError(
                f"间隔太小：min(intervals)={min(self.intervals)} 必须 ≥ window_radius+5"
                f"={self.K + 5}（方案①去相关约束），否则 N2N 标签与输入窗未去相关。"
            )

        # 发现序列：每条序列存 (目录, 帧文件名列表)
        self.sequences: list[tuple[str, list[str]]] = []
        for d in _find_sequence_dirs(root_dir):
            files = _list_frames(d)
            if len(files) >= 2 * self.K + 1 + 1:  # 至少能放下一个窗 + 一个窗外标签
                self.sequences.append((d, files))

        if not self.sequences:
            raise RuntimeError(
                f"在 {root_dir} 下未找到可用序列"
                f"（每条序列至少需 {2 * self.K + 2} 帧）。"
            )

        # 构建样本索引：(seq_id, center_t)，要求窗完整且至少存在一个合法标签
        self.samples: list[tuple[int, int]] = []
        for seq_id, (_d, files) in enumerate(self.sequences):
            n = len(files)
            for t in range(self.K, n - self.K):  # 保证 [t-K, t+K] 不越界
                if self._valid_targets(n, t):
                    self.samples.append((seq_id, t))

        if not self.samples:
            raise RuntimeError("没有构建出任何合法样本，请检查序列长度与 intervals 设置。")

    # --------- 合法标签：在界内、且在窗外（interval≥K+5 时天然窗外+去相关） ---------
    def _valid_targets(self, n: int, t: int) -> list[int]:
        targets = []
        for interval in self.intervals:
            for j in (t + interval, t - interval):
                if 0 <= j < n and not (t - self.K <= j <= t + self.K):
                    targets.append(j)
        return targets

    def __len__(self) -> int:
        return len(self.samples)

    # --------- 调试用：返回某样本的窗/候选标签信息，便于肉眼核对方案① ---------
    def sample_info(self, idx: int) -> dict:
        seq_id, t = self.samples[idx]
        seq_dir, files = self.sequences[seq_id]
        window = list(range(t - self.K, t + self.K + 1))
        return {
            "seq_dir": seq_dir,
            "num_frames": len(files),
            "center_t": t,
            "window_indices": window,
            "valid_target_indices": self._valid_targets(len(files), t),
        }

    # --------- 共享裁剪：窗内所有帧 + 标签帧用同一处裁剪 ---------
    def _make_crop(self, h: int, w: int) -> tuple[int, int, int] | None:
        if self.crop_size is None or self.crop_size >= min(h, w):
            return None  # 不裁剪 / 裁剪尺寸不小于原图 -> 用全图
        c = self.crop_size
        if self.random_crop:
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
        else:
            top = (h - c) // 2
            left = (w - c) // 2
        return top, left, c

    def _load_cropped(self, seq_dir: str, fname: str, crop) -> np.ndarray:
        img = _load_2d(os.path.join(seq_dir, fname))
        if crop is not None:
            top, left, c = crop
            img = img[top:top + c, left:left + c]
        return np.ascontiguousarray(img)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)
        if self.intensity_transform == "log1p":
            t = _log1p(t)
        return t

    def __getitem__(self, idx: int):
        seq_id, t = self.samples[idx]
        seq_dir, files = self.sequences[seq_id]
        n = len(files)

        # 随机挑一个合法标签帧（已保证在窗外 + 去相关）
        target_idx = random.choice(self._valid_targets(n, t))

        # 先按中心帧确定共享裁剪位置（同序列各帧同形状）
        center_img = _load_2d(os.path.join(seq_dir, files[t]))
        crop = self._make_crop(center_img.shape[0], center_img.shape[1])

        # 输入窗 {t-K..t+K}
        window_indices = range(t - self.K, t + self.K + 1)
        window = torch.stack(
            [self._to_tensor(self._load_cropped(seq_dir, files[i], crop)) for i in window_indices],
            dim=0,
        )  # (T, 1, H, W)

        # N2N 标签帧（窗外的另一份独立含噪观测）
        target = self._to_tensor(self._load_cropped(seq_dir, files[target_idx], crop))  # (1, H, W)

        # 窗内各帧的归一化全局时间戳 t_tilde=(i)/(N-1)（论文式 32 的 0-based 等价）
        denom = max(n - 1, 1)
        t_coords = torch.tensor([i / denom for i in window_indices], dtype=torch.float32)  # (T,)

        return window, t_coords, target


# ------------------------------------------------------------------
# 三、自检入口：跑通数据管线 + 肉眼核对方案①
#     用法：python -m dataset.video_pair_dataset <数据根目录>
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m dataset.video_pair_dataset <数据根目录>")
        raise SystemExit(1)

    root = sys.argv[1]
    ds = VideoN2NDataset(root, window_radius=2, pair_intervals=(7, 9), crop_size=512)
    print(f"序列数 = {len(ds.sequences)}，样本数 = {len(ds)}")

    K = ds.K
    # 抽几个样本核对：标签必须在窗外、且与窗内最近帧间隔 ≥5
    for idx in random.sample(range(len(ds)), k=min(5, len(ds))):
        info = ds.sample_info(idx)
        win = info["window_indices"]
        for tgt in info["valid_target_indices"]:
            assert tgt not in win, "❌ 标签落在输入窗内！"
            assert min(abs(tgt - w) for w in win) >= 5, "❌ 标签与窗内最近帧间隔 <5！"
        print(
            f"[ok] {os.path.relpath(info['seq_dir'], root)} | "
            f"N={info['num_frames']} center={info['center_t']} "
            f"window={win} targets={info['valid_target_indices']}"
        )

    # 真正取一个样本，确认张量形状
    window, t_coords, target = ds[0]
    print(f"window={tuple(window.shape)} (T,1,H,W)  "
          f"t_coords={tuple(t_coords.shape)} {t_coords.tolist()}  "
          f"target={tuple(target.shape)}")
    print("✅ 方案①约束校验通过，数据管线可用。")
