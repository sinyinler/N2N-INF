"""诊断：单帧 L2R（轻量 Denoiser + 重腐蚀器，无多帧/TSGM）。

用来隔离"L2R 机制本身在我们代码里是否 work"。若这个能去噪 → 问题在多帧/TSGM 泄漏；
若也不去噪 → 问题在 recorruptor 或 L2R 损失实现。
f(y₁)=Denoiser(y₁)（和跑得很好的 n2n_baseline 用法一致，非残差）。
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader

from dataset.video_pair_dataset import VideoN2NDataset, _load_2d, _log1p
from model.denoiser import Denoiser
from model.recorruptor import Recorruptor, id_pretrain


def save_vis(inp, den, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    imgs = [inp, den]
    vmin = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 1))
    vmax = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 99))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for a, im, t in zip(ax, imgs, ["input", "L2R single denoised"]):
        a.imshow(im, cmap="gray", vmin=vmin, vmax=vmax); a.set_title(t); a.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_iters", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out_dir", default="results_l2r1")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = args.alpha
    os.makedirs(args.out_dir + "/vis", exist_ok=True)

    ds = VideoN2NDataset(args.root, window_radius=2, pair_intervals=(7, 9), crop_size=args.crop_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                    drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    K = ds.K
    f = Denoiser(input_channels=1).to(dev)
    h = Recorruptor(ksize=3).to(dev)
    id_pretrain(h, steps=400, device=dev, log=print)
    fid = lambda p, y: ((p - y) ** 2).mean()
    opt_f = torch.optim.Adam(f.parameters(), lr=1e-4)
    opt_h = torch.optim.Adam(h.parameters(), lr=1e-4)
    print(f"[model] f(Denoiser)={sum(p.numel() for p in f.parameters())/1e6:.4f}M  h={sum(p.numel() for p in h.parameters())/1e3:.1f}K")

    seq_dir, files = ds.sequences[0]
    c = len(files) // 2
    prev = _log1p(torch.from_numpy(_load_2d(os.path.join(seq_dir, files[c]))).float().unsqueeze(0))  # (1,H,W)

    sigma_set = False
    it = 0; done = False
    while not done:
        for window, tc, target, box in dl:
            y = window[:, K].to(dev)                 # 单帧 (B,1,H,W)
            if not sigma_set:
                sig = ((window[:, K] - window[:, K + 1]) / (2 ** 0.5)).std().item()
                h.set_k(sig); sigma_set = True; print(f"[init] σ_est={sig:.4f}")
            w = torch.randn_like(y)
            # max_h
            hw = h(w); pred = f(y + a * hw)
            L = fid(pred, y) + (2.0 / a) * (pred * hw).mean()
            if torch.isfinite(L):
                opt_h.zero_grad(set_to_none=True); (-L).backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), 5.0); opt_h.step()
            # min_f
            hw = h(w).detach(); pred = f(y + a * hw)
            Lf = fid(pred, y) + (2.0 / a) * (pred * hw).mean()
            opt_f.zero_grad(set_to_none=True); Lf.backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt_f.step()

            if it % 50 == 0:
                print(f"iter {it} Lf {Lf.item():.5f} k {torch.nn.functional.softplus(h.log_k).item():.4f}")
            if it % 200 == 0:
                f.eval()
                with torch.no_grad():
                    den = f(prev.unsqueeze(0).to(dev))[0, 0].cpu().numpy()
                f.train()
                save_vis(prev[0].numpy(), den, os.path.join(args.out_dir, "vis", f"it{it:06d}.png"))
            it += 1
            if it >= args.max_iters:
                done = True; break
    print("done")


if __name__ == "__main__":
    main()
