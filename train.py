"""
训练入口（骨架）。

流程（待实现）：
  1. 读 configs/default.yaml；
  2. 构建 VideoN2NDataset + DataLoader；
  3. 构建 SINF 模型；
  4. 损失：Charbonnier（默认）/ L2（对比实验 E1，由 config.loss.type 切换）；
     RTV 暂不挂（对比实验 E2）；
  5. Adam(lr=1e-4, betas=(0.9,0.999), wd=1e-6) + grad clip=5；
  6. 训练循环：预测中心帧 Ĉ_t，对 N2N 标签帧算 loss；
  7. 定期存 checkpoint 到 results/，并在 experiment_log.md 记一条。

注意：多帧 + TSGM 显存大，crop/batch 大概率需从 512/48 下调，先实测。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations


def main():
    raise NotImplementedError("骨架阶段，待实现")


if __name__ == "__main__":
    main()
