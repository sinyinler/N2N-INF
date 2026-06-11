"""
训练入口。N2N 自监督训练 SINF。

  python train.py --config configs/default.yaml --root <数据根目录>
  常用覆盖： --crop_size 256 --batch_size 2 --epochs 100 --device cuda

流程：
  1. 读 yaml；2. VideoN2NDataset + DataLoader；3. SINF.from_config；
  4. 损失 Charbonnier(默认)/L2(对比实验E1，由 loss.type 切换)；RTV 暂不挂(E2)；
  5. Adam + grad clip；6. 预测中心帧 Ĉ_t，对 N2N 标签帧算 loss；
  7. 定期存 checkpoint 到 results/ 并存训练可视化。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.video_pair_dataset import VideoN2NDataset
from loss.charbonnier import CharbonnierLoss
from model.sinf import SINF


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loss(loss_cfg: dict):
    """Charbonnier(默认) 或 L2（对比实验 E1）。"""
    t = str(loss_cfg.get("type", "charbonnier")).lower()
    if t == "charbonnier":
        return CharbonnierLoss(eps=float(loss_cfg.get("charbonnier_eps", 1e-3)))
    if t in ("l2", "mse"):
        return torch.nn.MSELoss()
    raise ValueError(f"未知 loss.type: {t}")


def save_train_vis(window, out, target, save_path):
    """存一张 3 联图：中心输入 | 去噪输出 | N2N标签（log 域，分位显示）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = window.shape[1]
    center_in = window[0, T // 2, 0].detach().cpu().numpy()
    out_img = out[0, 0].detach().cpu().numpy()
    tgt_img = target[0, 0].detach().cpu().numpy()

    imgs = [center_in, out_img, tgt_img]
    titles = ["center input", "denoised", "N2N target"]
    vmin = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 1))
    vmax = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 99))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, im, ti in zip(axes, imgs, titles):
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(ti)
        ax.axis("off")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--root", default=None, help="数据根目录（覆盖 config.data.root_dir）")
    ap.add_argument("--crop_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_iters", type=int, default=None, help="调试用：限制总迭代数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]

    root = args.root or dcfg.get("root_dir") or ""
    crop = args.crop_size if args.crop_size is not None else dcfg.get("crop_size", 256)
    bs = args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 2)
    epochs = args.epochs if args.epochs is not None else tcfg.get("epochs", 100)
    device = args.device or tcfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = tcfg.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    # 数据
    ds = VideoN2NDataset(
        root,
        window_radius=mcfg.get("tsgm", {}).get("temporal_radius", 2),
        pair_intervals=tuple(dcfg.get("pair_intervals", [7, 9])),
        crop_size=crop,
        intensity_transform=dcfg.get("intensity_transform", "log1p"),
        random_crop=True,
        npy_subdir=dcfg.get("npy_subdir", "npy"),
        exclude_dirs=tuple(dcfg.get("exclude_dirs", ["bfi_nonoverlap"])),
    )
    dl = DataLoader(ds, batch_size=bs, shuffle=True,
                    num_workers=tcfg.get("num_workers", 4), drop_last=True, pin_memory=(device == "cuda"))
    print(f"[data] root={root} 序列={len(ds.sequences)} 样本={len(ds)} crop={crop} bs={bs}")

    # 模型 / 损失 / 优化器
    model = SINF.from_config(cfg).to(device)
    use_dp = (device == "cuda" and torch.cuda.device_count() > 1 and tcfg.get("multi_gpu", True))
    if use_dp:
        model = torch.nn.DataParallel(model)
        print(f"[multi-gpu] DataParallel 启用，{torch.cuda.device_count()} 卡；batch_size={bs} 会按卡均分")
    loss_fn = build_loss(cfg["loss"])
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-4)),
        betas=tuple(tcfg.get("betas", [0.9, 0.999])),
        weight_decay=float(tcfg.get("weight_decay", 1e-6)),
    )
    clip = float(tcfg.get("grad_clip_norm", 5))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] SINF 参数 {n_params/1e6:.4f}M  loss={cfg['loss'].get('type')}  device={device}")

    it = 0
    for epoch in range(epochs):
        model.train()
        for window, t_coords, target in dl:
            window, t_coords, target = window.to(device), t_coords.to(device), target.to(device)
            out = model(window, t_coords)
            loss = loss_fn(out, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            if it % 50 == 0:
                print(f"epoch {epoch} iter {it} loss {loss.item():.5f}")
            if tcfg.get("vis_every") and it % int(tcfg["vis_every"]) == 0:
                save_train_vis(window, out, target, os.path.join(out_dir, "vis", f"it{it:06d}.png"))
            it += 1
            if args.max_iters and it >= args.max_iters:
                break
        if args.max_iters and it >= args.max_iters:
            break

        if epoch % int(tcfg.get("save_every", 1)) == 0:
            ckpt = os.path.join(out_dir, f"sinf_epoch{epoch:03d}.pth")
            state = (model.module if use_dp else model).state_dict()
            torch.save({"model": state, "epoch": epoch, "cfg": cfg}, ckpt)
            print(f"[ckpt] saved {ckpt}")

    final_state = (model.module if use_dp else model).state_dict()
    torch.save({"model": final_state, "epoch": epochs, "cfg": cfg},
               os.path.join(out_dir, "sinf_last.pth"))
    print("[done] 训练结束")


if __name__ == "__main__":
    main()
