# BFI / LSCI 去噪 — SINF (BSN→N2N) 复现

复现 Hu et al. 2026 (TPAMI) *Learning Continuous Spatiotemporal Implicit Neural Fields for
Unsupervised Video Denoising* (SINF)，并将其自监督从 **BSN（盲点）替换为 N2N**，
以适配激光散斑血流成像（LSCI）的**空间相关散斑噪声**（盲点在此失效）。

保留论文相对普通 N2N 的核心增益：连续空间 INF 表示 + ITE 隐式时间嵌入 + TSGM 无光流时空对齐。

> 设计决策、刻意偏离论文之处、以及待办对比实验，全部记录在 [`experiment_log.md`](experiment_log.md)。

## 目录结构

```
model/                # 网络
  denoiser.py         # 现有单帧 CNN（将降级为逐帧编码 backbone）
  inf_head.py         # INF 空间解码头（BS-INF 去盲点）
  ite.py              # 隐式时间嵌入
  tsgm.py             # 时间感知空间图对齐
  sinf.py             # 顶层组装
dataset/
  data.py             # 现有数据逻辑（lbf/npy 读取、VST、配对）
  video_pair_dataset.py  # 多帧 N2N 配对（窗 + 排除式标签）
loss/                 # charbonnier.py / rtv.py（RTV 暂不挂）
utils/                # 指标、lbf 读取（原样复用）
configs/default.yaml  # 全部超参集中管理
train.py / eval.py    # 训练 / 评估出图入口
results/              # 输出、图像、checkpoint
experiment_log.md     # 实验全记录
```

## 当前状态

骨架阶段：模块为带注释的空壳，**尚未填实现**。建议实现顺序见 `experiment_log.md` §5。
