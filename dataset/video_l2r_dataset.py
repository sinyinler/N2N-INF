"""L2R 融合数据集：中心帧 + 去相关偏移上的上下文帧。

与 N2N 数据集的关键不同：上下文帧取在 **|offset|≥5** 的去相关位置——这样它们的噪声
与中心帧独立，满足 L2R 融合的无偏条件（已数值验证：相关上下文会引入偏差）。
不需要单独的 N2N 标签帧：L2R 的监督目标就是中心帧自己。

返回 window (T,1,H,W)，中心帧在 index = center_pos（offsets 对称时即 T//2）。
"""
from __future__ import annotations

import os
import random

import torch
from torch.utils.data import Dataset

from dataset.video_pair_dataset import (
    _find_sequence_dirs, _list_frames, _load_2d_region, _load_2d_shape, _log1p,
)


class VideoL2RDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        context_offsets=(-9, -7, 7, 9),
        crop_size: int = 256,
        intensity_transform: str = "log1p",
        random_crop: bool = True,
        npy_subdir: str = "npy",
        exclude_dirs=("bfi_nonoverlap",),
    ):
        self.offsets = sorted(set(int(o) for o in context_offsets) | {0})
        self.center_pos = self.offsets.index(0)
        self.maxoff = max(abs(o) for o in self.offsets)
        for o in self.offsets:
            if o != 0 and abs(o) < 5:
                raise ValueError(f"上下文偏移 {o} 的 |.|<5，与中心帧噪声未去相关，违反 L2R 无偏条件")

        self.crop_size = None if (not crop_size or int(crop_size) <= 0) else int(crop_size)
        self.intensity_transform = str(intensity_transform).lower()
        self.random_crop = bool(random_crop)

        self.sequences: list[tuple[str, list[str]]] = []
        for d in _find_sequence_dirs(root_dir, npy_subdir, tuple(exclude_dirs)):
            files = _list_frames(d)
            if len(files) >= 2 * self.maxoff + 1:
                self.sequences.append((d, files))
        if not self.sequences:
            raise RuntimeError(f"{root_dir} 下没有足够长的序列（每条≥{2*self.maxoff+1}帧）")

        self.samples: list[tuple[int, int]] = []
        for sid, (_d, files) in enumerate(self.sequences):
            n = len(files)
            for t in range(self.maxoff, n - self.maxoff):
                self.samples.append((sid, t))

    def __len__(self) -> int:
        return len(self.samples)

    def _make_crop(self, H, W):
        if self.crop_size is None or self.crop_size >= min(H, W):
            return None
        c = self.crop_size
        if self.random_crop:
            top, left = random.randint(0, H - c), random.randint(0, W - c)
        else:
            top, left = (H - c) // 2, (W - c) // 2
        return top, left, c

    def _to_tensor(self, img):
        t = torch.from_numpy(img).float().unsqueeze(0)
        if self.intensity_transform == "log1p":
            t = _log1p(t)
        return t

    def __getitem__(self, idx):
        sid, t = self.samples[idx]
        seq_dir, files = self.sequences[sid]
        H, W = _load_2d_shape(os.path.join(seq_dir, files[t]))
        crop = self._make_crop(H, W)
        window = torch.stack(
            [self._to_tensor(_load_2d_region(os.path.join(seq_dir, files[t + o]), crop)) for o in self.offsets],
            dim=0,
        )  # (T,1,H,W)，中心在 self.center_pos
        return window


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "D:/Desktop/lightweight_G/mix"
    ds = VideoL2RDataset(root, context_offsets=(-9, -7, 7, 9), crop_size=256)
    print(f"序列={len(ds.sequences)} 样本={len(ds)} 偏移={ds.offsets} 中心位置={ds.center_pos}")
    w = ds[0]
    print(f"window={tuple(w.shape)} (T,1,H,W)，中心帧 index={ds.center_pos}")
