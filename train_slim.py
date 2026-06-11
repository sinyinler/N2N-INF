"""精简融合模型 + L2R 自监督 训练入口（静止阶段）。

f = SlimFusion（卷积 backbone + TSGM + 卷积头，无坐标INF）；h = Recorruptor。
目标(α=1)：L = fid(f(win_in), y) + (2/α)·<f(win_in), h(w)>，min_f max_h。
其中 y=中心帧，win_in=把中心换成 y₁=y+α·h(w) 的窗，邻帧(去相关偏移)当上下文。
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader

from dataset.video_l2r_dataset import VideoL2RDataset
from dataset.video_pair_dataset import _load_2d_region, _load_2d_shape, _log1p
from model.slim_fusion import SlimFusion
from model.recorruptor import Recorruptor, id_pretrain


def save_vis(inp, den, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    imgs = [inp, den]
    vmin = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 1))
    vmax = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 99))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for a, im, t in zip(ax, imgs, ["center input", "L2R-fusion denoised"]):
        a.imshow(im, cmap="gray", vmin=vmin, vmax=vmax); a.set_title(t); a.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def build_preview(ds, crop=384):
    seq_dir, files = ds.sequences[0]
    c = len(files) // 2
    H, W = _load_2d_shape(os.path.join(seq_dir, files[c]))
    crop = min(crop, H, W)
    top, left = (H - crop) // 2, (W - crop) // 2
    cr = (top, left, crop)
    frames = [_log1p(torch.from_numpy(_load_2d_region(os.path.join(seq_dir, files[c + o]), cr)).float().unsqueeze(0))
              for o in ds.offsets]
    win = torch.stack(frames, 0)                      # (T,1,crop,crop)
    return win, win[ds.center_pos, 0].numpy()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--loss", choices=["l2", "charb"], default="l2")
    ap.add_argument("--lr_f", type=float, default=1e-4)
    ap.add_argument("--lr_h", type=float, default=1e-4)
    ap.add_argument("--offsets", type=int, nargs="+", default=[-9, -7, 7, 9])
    ap.add_argument("--out_dir", default="results_slim")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir + "/vis", exist_ok=True)
    a = args.alpha

    ds = VideoL2RDataset(args.root, context_offsets=tuple(args.offsets), crop_size=args.crop_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                    drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    cpos = ds.center_pos
    print(f"[data] 序列={len(ds.sequences)} 样本={len(ds)} 偏移={ds.offsets} 中心={cpos}")

    f = SlimFusion(in_channels=1, tsgm_heads=4, residual=True).to(dev)
    h = Recorruptor(ksize=3).to(dev)
    id_pretrain(h, steps=400, device=dev, log=print)            # 稳住 min-max 初期

    if args.loss == "l2":
        fid = lambda p, y: ((p - y) ** 2).mean()
    else:
        fid = lambda p, y: torch.sqrt((p - y) ** 2 + 1e-6).mean()

    opt_f = torch.optim.Adam(f.parameters(), lr=args.lr_f)
    opt_h = torch.optim.Adam(h.parameters(), lr=args.lr_h)
    print(f"[model] f={sum(p.numel() for p in f.parameters())/1e6:.4f}M  h={sum(p.numel() for p in h.parameters())/1e3:.1f}K  loss={args.loss}")

    prev_win, prev_in = build_preview(ds)
    sigma_set = False
    it = 0
    done = False
    while not done:
        for window in dl:
            window = window.to(dev)                  # (B,T,1,H,W)
            y = window[:, cpos]                       # 中心帧 (B,1,H,W)
            if not sigma_set:                         # 用去相关帧对估噪声 std 初始化 k
                sig = ((window[:, cpos] - window[:, cpos + 1]) / (2 ** 0.5)).std().item()
                h.set_k(sig); sigma_set = True
                print(f"[init] σ_est={sig:.4f} -> k 初始化")

            w = torch.randn_like(y)
            # --- max_h：上升 L（更新 h） ---
            hw = h(w)
            win_in = window.clone(); win_in[:, cpos] = y + a * hw
            pred = f(win_in)
            L = fid(pred, y) + (2.0 / a) * (pred * hw).mean()
            if torch.isfinite(L):
                opt_h.zero_grad(set_to_none=True); (-L).backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), 5.0); opt_h.step()
            # --- min_f：下降 L（更新 f，h 冻结） ---
            hw = h(w).detach()
            win_in = window.clone(); win_in[:, cpos] = y + a * hw
            pred = f(win_in)
            Lf = fid(pred, y) + (2.0 / a) * (pred * hw).mean()
            opt_f.zero_grad(set_to_none=True); Lf.backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt_f.step()

            if it % 50 == 0:
                k = torch.nn.functional.softplus(h.log_k).item()
                print(f"iter {it} Lf {Lf.item():.5f} k {k:.4f}")
            if it % 200 == 0:
                f.eval()
                with torch.no_grad():
                    den = f(prev_win.unsqueeze(0).to(dev))[0, 0].cpu().numpy()
                f.train()
                save_vis(prev_in, den, os.path.join(args.out_dir, "vis", f"it{it:06d}.png"))
            if it > 0 and it % 2000 == 0:
                torch.save({"f": f.state_dict(), "h": h.state_dict(), "it": it},
                           os.path.join(args.out_dir, "slim_last.pth"))
            it += 1
            if it >= args.max_iters:
                done = True; break
    torch.save({"f": f.state_dict(), "h": h.state_dict(), "it": it},
               os.path.join(args.out_dir, "slim_last.pth"))
    print("done")


if __name__ == "__main__":
    main()
