"""纯单帧 N2N baseline（同一轻量 backbone），用于和 SINF 对比。

输入中心帧 -> Denoiser 直接出图，监督用窗外的 N2N 标签帧。复用 VideoN2NDataset。
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader

from dataset.video_pair_dataset import VideoN2NDataset, _load_2d, _log1p
from loss.charbonnier import CharbonnierLoss
from model.denoiser import Denoiser


def save_vis(inp, den, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    imgs = [inp, den]
    vmin = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 1))
    vmax = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 99))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for a, im, t in zip(ax, imgs, ["center input (full)", "N2N denoised (full)"]):
        a.imshow(im, cmap="gray", vmin=vmin, vmax=vmax); a.set_title(t); a.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--out_dir", default="results_n2n")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir + "/vis", exist_ok=True)
    ds = VideoN2NDataset(args.root, window_radius=2, pair_intervals=(7, 9), crop_size=args.crop_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    K = ds.K
    net = Denoiser(input_channels=1).to(dev)
    lf = CharbonnierLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    print(f"[n2n] 序列={len(ds.sequences)} 样本={len(ds)} 参数={sum(p.numel() for p in net.parameters())/1e6:.4f}M")

    # 固定全图预览（中心帧）
    seq_dir, files = ds.sequences[0]
    c = len(files) // 2
    prev_in = _log1p(torch.from_numpy(_load_2d(os.path.join(seq_dir, files[c]))).float().unsqueeze(0))  # (1,H,W)

    it = 0
    done = False
    while not done:
        for window, tc, target, box in dl:
            x = window[:, K].to(dev)          # 中心帧 (B,1,H,W)
            target = target.to(dev)
            out = net(x)
            loss = lf(out, target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if it % 50 == 0:
                print(f"iter {it} loss {loss.item():.5f}")
            if it % 200 == 0:
                net.eval()
                with torch.no_grad():
                    den = net(prev_in.unsqueeze(0).to(dev))[0, 0].cpu().numpy()
                net.train()
                save_vis(prev_in[0].numpy(), den, os.path.join(args.out_dir, "vis", f"it{it:06d}.png"))
            it += 1
            if it >= args.max_iters:
                done = True; break
    torch.save({"model": net.state_dict()}, os.path.join(args.out_dir, "n2n_last.pth"))
    print("done")


if __name__ == "__main__":
    main()
