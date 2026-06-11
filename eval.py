"""
评估 / 出图入口（骨架）。

流程（待实现）：
  1. 加载 checkpoint；
  2. 对测试序列做**全图滑窗推理**（窗 T=5，逐帧滑动取中心帧输出）；
  3. 指标：复用 utils/calc_image_metrics.py 的 PSNR / SSIM / Pearson R；
  4. 出图：去噪前后对比 + **细小血管 ROI 局部放大**（规范第4条，
     指标好但磨平细血管时以图像为准）。

TODO(实现)：暂为骨架，不含逻辑。
"""

from __future__ import annotations


def main():
    raise NotImplementedError("骨架阶段，待实现")


if __name__ == "__main__":
    main()
