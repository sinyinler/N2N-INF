"""
多帧 N2N 数据集（B 路新增）。

在现有 dataset/data.py 的基础上，把"两帧配对"改成"一个输入帧窗 + 一个被排除的标签帧"：
  - 输入窗：{t-K,...,t,...,t+K}，K = window_radius = 2（T=5）；
  - N2N 标签：第 t±interval 帧，interval ∈ {7,9}；
  - ⚠️ 正确性约束：标签帧 t±interval 必须落在输入窗 [t-K, t+K] 之外
    （方案①：窗内最近帧到标签间隔 ≥5 才去相关 → interval ≥ K+5 = 7）；
  - 配对只在同一序列文件夹内，不跨序列；
  - 强度变换：log1p（复用 data.py 的 log1p_torch）。

复用：data.py 的 lbf/npy 读取、自然排序、序列发现逻辑；
仅"取样逻辑"从单帧改为帧窗 + 排除式标签。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations

from torch.utils.data import Dataset


class VideoN2NDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        window_radius: int = 2,
        pair_intervals=(7, 9),
        crop_size: int = 512,
        intensity_transform: str = "log1p",
    ):
        # 校验：min(pair_intervals) 必须 >= window_radius + 5（方案①去相关约束）
        raise NotImplementedError("骨架阶段，待实现")

    def __len__(self):
        raise NotImplementedError("骨架阶段，待实现")

    def __getitem__(self, idx):
        """返回 (window, t_coords, target)；target 帧保证不在 window 内。"""
        raise NotImplementedError("骨架阶段，待实现")
