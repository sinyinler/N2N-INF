# 实验全记录 (experiment_log.md)

> 项目长期记忆。每出一次结果就在 **changelog** 追加一条（改了什么 / 为什么 / 配置+commit / 结果 / 结论与下一步）。
> 这份是底层流水账；组会 PPT 是它的展示层。

---

## 0. 项目背景

- **复现论文**：Hu et al. 2026, TPAMI, *Learning Continuous Spatiotemporal Implicit Neural Fields for Unsupervised Video Denoising* (SINF)。项目根目录附 PDF。
- **目标场景**：BFI / 激光散斑血流成像（LSCI，`.lbf` 数据），去噪要求——**噪声抑制 + 血管结构光滑，但细小血管不能磨平**。
- **核心改动（为什么不照搬原文）**：SINF 的全部自监督来自 **BSN（盲点 / J-invariance）**，而盲点成立要求**噪声空间独立**；LSCI 的散斑是**空间相关的结构化噪声**，盲点失效（前期已验证）。因此：
  - **把 BSN 换成 N2N**：N2N 只要求两次观测的噪声**相互独立**（靠帧间去相关满足），不要求空间独立。
  - **保留论文真正相对普通 N2N 的增益**：连续空间 INF 表示 + ITE（隐式时间嵌入）+ TSGM（无光流时空对齐）。
- **范围**：B 路（忠实复现 SINF，仅把 BSN 替换为 N2N）。

---

## 1. 锁定的设计决策（已与用户逐条确认）

| 项 | 决策 | 备注 |
|---|---|---|
| 复现范围 | B 路：多帧视频模型 = 滑窗 + 双向编码 + ITE + TSGM + INF 解码 | — |
| 自监督 | N2N（去掉盲点 mask、去掉蒸馏项） | — |
| 帧间独立性 | interval ≥ 5 帧才去相关 | 用户依据 |
| 输入窗 T | **T=5（半径 K=2，窗={t-2..t+2}）** | 对齐论文真实噪声分支 T 与方案① |
| N2N 标签间隔 | **只用 {7, 9}，弃用 5** | 方案①：窗内最近帧到标签需 ≥5；K=2 → interval≥7 |
| ⚠️ N2N 正确性约束 | **标签帧不得出现在输入窗内**，否则网络抄答案退化成恒等映射 | 单帧时天然满足；多帧必须在 dataset 里强制 |
| TSGM 时间半径 | **对齐到 2**（论文写 3，为与 T=5 自洽改 2） | — |
| backbone | **先用现有轻量配置 16-32-64-80** | 论文 32-64-128-128-128 列为对比实验（见 §3） |
| Fourier 坐标频带数 B | **10**（论文未写死，取推荐值） | — |
| 输出通道 | 1（LSCI 单通道） | 论文是 RGB 3 通道 |
| 损失 | **Charbonnier 默认** | L2 列为对比实验（见 §3） |
| RTV 正则 | **先不挂** | 细血管被磨平时作对比实验加回（见 §3） |
| 强度变换 (VST) | **先用 log1p** | learned-VST 列为后续创新实验（见 §3） |
| ITE | 3 层 sin-MLP，hidden=64 | 照搬论文 |
| INF 头 F_Θ | 4 层 MLP，hidden=128，ReLU | 照搬论文 |
| TSGM | 7×7 窗口，4 个 attention head | 照搬论文 |
| 优化器 | Adam，lr=1e-4，β=(0.9,0.999)，grad_clip=5，wd=1e-6 | 照搬论文 |
| 硬件 | 2× A500 工作站 | — |
| crop/batch | 旧配置 512×512 / batch=48（**待实测下调**，见 §4 风险） | 多帧模型显存翻倍 |

---

## 2. 相对论文的刻意偏离（复现时勿当 bug）

1. **删除盲点**：BS-INF 的 `MaskConv` + 掩码坐标场（原文式 17/29）全部去掉，只保留 `DilatedConv 局部特征 + 坐标 Fourier INF + 融合 MLP`，退化为"普通连续空间 INF 头"。
2. **删除蒸馏项**：论文真实噪声分支用 盲点 L2 + 蒸馏（λ=5e4，teacher 为预训练去噪器）。我们无 teacher，纯 N2N，**不要蒸馏项**。
3. **不用论文的合成 NLL 损失**：NLL（负对数似然，有监督概率损失）需要干净 GT，论文只用在合成噪声分支；我们是真实 LSCI 无 GT，走 N2N，故不用。
4. **强度域**：论文归一化到 [0,1]；我们走 log1p（散斑方差稳定），保留为领域适配。

---

## 3. 待办对比实验队列（⚠️ 别忘，后面要用）

> 以下都是当前"先用 A，效果不行再试 B"留下的对照项。每做一项，在 changelog 记结果并回填这里。

| # | 对比维度 | 默认（当前） | 对照项 | 触发条件 / 目的 | 状态 |
|---|---|---|---|---|---|
| E1 | 损失函数 | Charbonnier | **L2** | N2N 理论上配 L2 才严格无偏；验证 Charbonnier 是否引入偏置 | 待做 |
| E2 | RTV 正则 | 不挂 RTV | **加 RTV** | 当"INF 本身保细血管"不成立、细小血管被磨平时 | 待做 |
| E3 | 强度变换 | log1p | **learned-VST** | 作为创新点；先确认 log1p 能跑通再上 | 待做 |
| E4 | backbone 容量 | 轻量 16-32-64-80 | **论文 32-64-128-128-128** | 轻量效果不够时做容量增强对比 | 待做 |

---

## 4. 已知风险 / 待实测

- **显存**：旧 512×512 / batch=48 是单帧 CNN 的配置。多帧（×5）+ TSGM 图注意力 + 逐像素 INF 解码会大幅抬高显存，512/48 极可能 OOM。模型搭好后**先实测**，按需下调 crop（256/128）和 batch。
- **帧间运动**：用户反馈肉眼运动不明显，但血流确实在动 → TSGM 对齐的价值需在结果上验证（对比"有/无 TSGM"也可作为消融，待定）。
- **interval=5 弃用**：当前 {7,9} 都满足方案①；若后续想缩小窗到 K=1(T=3) 以复用 interval=6，需重新评估。
- **拷贝缺文件**：`dataset/data.py` 顶部 import `utils/monotonic_vst.py`，但该文件不在本目录拷贝里 → 直接 import data.py 会 ImportError。新的 `video_pair_dataset.py` 已写成自包含（不依赖它）。做 learned-VST 实验(E3) 时需补回 `utils/monotonic_vst.py`。
- **运行环境**：torch 在 conda env **`denoise`**（torch 2.8.0+cu126, CUDA 可用）；base 环境无 torch。跑脚本用 `D:/Anaconda/envs/denoise/python.exe`。

---

## 5. Changelog

### 2026-06-11 — 项目对齐与骨架搭建
- 完成论文精读、与用户逐条对齐设计决策（见 §1）。
- 确定 B 路：BSN→N2N，保留 ITE+TSGM+INF。
- 搭建项目骨架（带注释空模块 + `configs/default.yaml`），**尚未写实现逻辑**。
- 下一步：经用户审核骨架后，按模块填实现（建议顺序：dataset 多帧配对 → backbone 复用 → INF 头 → ITE → TSGM → sinf 顶层 → train/eval）。

### 2026-06-11 — 实现 dataset/video_pair_dataset.py（多帧 N2N 配对）
- 改了什么：实现多帧 N2N 数据集——滑窗输入 + 窗外 N2N 标签 + 共享裁剪 + log1p；自包含（仅依赖 utils/lbfreadnew）。
- 为什么：数据管线是 N2N 正确性命门，先跑通并核对方案①再搭网络。
- 关键设计：`__init__` 强制校验 `min(intervals) ≥ K+5`；`_valid_targets` 保证标签在窗外且去相关；同一样本窗内帧与标签共用一处裁剪。
- 验证（合成数据，denoise env）：20帧序列→16样本；遍历全部样本"违反方案①约束数=0"；window=(5,1,64,64)、target=(1,64,64)、t_coords=(5,)；log1p 生效；样本0 center=2/window=[0..4]/targets=[9,11] 符合预期。
- 结论/下一步：数据层可用。下一步复用 model/denoiser.py 降级为逐帧 backbone，再写 INF 头。
- 真实数据自检（`D:/Desktop/lightweight_G/mix`，真实集的一小部分）：该目录是 1 条连续序列 `0.npy..999.npy`（1000 帧，无子目录）→ 996 样本；方案①约束零违反；配对/形状/log1p 均正确。
  - 真实帧：**1208×1352** float32，值域 [1.49, 531]，无负值、非整数（处理后的散斑/血流指数图）。→ 512 随机裁剪富余；log1p 输入全正。
  - ⚠️ 全图 1208×1352 比 denoiser.py 假设的 680×680 大；eval 全图滑窗 ×5帧 显存重，需分块处理（归入既有 eval 显存项）。
  - 序列发现约定：`_find_sequence_dirs` 把"直接含帧文件的目录"各算一条序列；服务器上多序列若为 mix/&lt;子目录&gt;/*.npy 可正确分隔，不会跨序列配对。

### 2026-06-11 — 实现 model/backbone.py（逐帧特征器）
- 改了什么：复用 denoiser.py 的 Encoder/Bridge/Decoder，去掉最后 1×1 输出头，封装成 `FrameEncoder`：单帧→16ch 全分辨率特征；`forward_window` 把时间维折进 batch 共享权重处理。
- 为什么：INF头/ITE/TSGM 都要挂在逐帧特征上，先把这个底座做干净。
- 验证：(2,5,1,128,128)→(2,5,16,128,128)，参数量 0.0675M（轻量）。
- 待对齐（留到 sinf.py）：论文"双向编码器 Ff/Fb"是两套权重还是共享+前后向聚合，原文含糊，组装顶层时与用户确认。

### 2026-06-11 — 实现 model/inf_head.py（INF 空间头，去盲点）
- 改了什么：实现 BS-INF 去掉盲点 mask 的版本——局部 dilated 分支(式26) + 坐标 Fourier INF 分支(式27/28/30) + 融合 MLP(式31)；**删除式29 的盲点 mask**。坐标 MLP 用 1×1 conv 实现逐像素 MLP。
- 关键参数：fourier_bands=10 → Fourier 维=60（sin/cos×10×3坐标）；in/out=16ch；时间戳走 dataset 的全局归一化 t_norm。
- 验证：(10,16,128,128)→(10,16,128,128)，参数 0.026M。
- 注：t_norm 在长序列(N=1000)下窗内相邻帧 Δ≈0.001，时间分辨率很细；若 ITE 难以区分，后续可改为"窗内局部归一化"，留意。

### 2026-06-11 — 实现 model/ite.py（隐式时间嵌入）
- 改了什么：SIREN 正弦 MLP（3层 hidden64，式33）把 t_norm→时间嵌入 e_t；`broadcast_concat` 沿空间广播拼接到特征（式35）。含 SIREN 初始化。
- 关键参数：w0=30（论文未写死，取 SIREN 默认；可作消融）；out_dim=64。
- 验证：e_t (10,64)、拼接后 (10,80,128,128)，参数 0.0084M。

### 2026-06-11 — 实现 model/tsgm.py（完整 TSGM）
- 用户决策：直接上完整 TSGM（不走占位）；双向采纳"共享 backbone + 对称时间注意力"（中心帧同时 attend 前后帧），不搞两套编码器权重。
- 改了什么：窗口化跨帧多头注意力（非重叠 7×7 窗，Swin 式划分）+ 相对帧时间偏置（time alignment）+ 残差/LayerNorm/卷积FFN（式22、图6）。中心帧 query、窗内全 T 帧 key/value。
- 实现约定（论文未写死）：非重叠窗代替逐像素 N(x,y) 邻域（更轻）；H/W 自动 pad 到 ws 整数倍再裁回。
- 验证：(2,5,32,50,60)→(2,32,50,60)（含非7倍数 padding），参数 0.0092M。
- 待定：dim（TSGM 工作通道）在 sinf 组装时定；overlapping 窗 / 逐像素邻域可作精度对比实验。

### 2026-06-11 — 组装 model/sinf.py（完整 SINF）+ 真实数据贯通
- 改了什么：顶层把 backbone→INFHead(式21)→ITE(式35)→TSGM(式22)→FinalINFHead F_Θ(式23) 串起来；新增最终重建头 F_Θ（4层 MLP hidden128，1×1conv 逐像素）。提供 `from_config`。
- 通道流：feats16 → z_τ16 → +ITE(out16) → 32 → TSGM → 32 → F_Θ → 1。tsgm_dim=32 可被 4 头整除。
- 验证(dummy)：(2,5,1,64,64)→(2,1,64,64)，**总参数 0.1532M**，前向+反向 OK。
- 验证(真实数据贯通)：mix 数据 → DataLoader(bs2,crop256) → SINF → (2,1,256,256)(CUDA) → Charbonnier N2N loss=3.012 → backward OK。
- ⚠️ **显存实测：crop=256 / batch=2 → 峰值 5.40 GB**。旧 512/48 配置在多帧模型上完全不可行（需大幅下调 crop/batch）。第一次真跑前需用户确认 A500 单卡显存。

### 2026-06-11 — 实现 train.py / eval.py，整条 pipeline 跑通
- 改了什么：
  - `train.py`：yaml 驱动；VideoN2NDataset+DataLoader→SINF→Charbonnier(默认)/L2(E1切换)→Adam+gradclip；定期存 ckpt + 训练3联图可视化。
  - `eval.py`：全图**分块滑窗推理**（块=tile_size 与训练 crop 一致，保证坐标归一化口径；Hann 羽化融合消接缝）→ 中心帧去噪(log域)→ npy+对比png。
  - config 补 train(epochs/save_every/vis_every/out_dir 等) 与 eval(tile_size/overlap)。
- 验证(本地 3060, 真实 mix 数据)：train 5 iter loss 3.13→存 sinf_last.pth+vis；eval 对全图 1208×1352 分块推理→denoised_log.npy(6.5MB)+compare.png。管线全通（模型未训练，输出暂为噪声）。
- 本地环境：1×RTX 3060(12.9GB)；A500 在服务器（显存未知，待用户确认以定 crop/batch）。
- 重要记号 — 坐标归一化口径：INFHead 按"当前块 H,W"归一化坐标到[-1,1]，训练用 crop 块、eval 必须同尺寸分块才一致。坐标分支是否真有用（vs 仅作块内位置基）可作消融实验。

## 6. 里程碑
- **2026-06-11**：SINF(BSN→N2N) 全部模块 + train/eval 建成，真实数据端到端跑通。下一步＝在服务器 A500 上做第一次正式训练（需先定 crop/batch），产出首个 baseline 指标+图像。
