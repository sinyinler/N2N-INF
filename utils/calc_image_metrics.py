import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from skimage.metrics import structural_similarity


def load_array(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        return np.load(path)

    if ext == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"No arrays found in npz file: {path}")
            return data[keys[0]]

    try:
        import imageio.v3 as iio

        return iio.imread(path)
    except Exception:
        from PIL import Image

        return np.array(Image.open(path))


def normalize_image_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {arr.shape}")

    # Heuristic for channel-first arrays like (3, H, W).
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"Only 2D grayscale or 3D HWC images are supported, got shape {arr.shape}"
        )

    return arr


def center_crop_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_h = min(a.shape[0], b.shape[0])
    target_w = min(a.shape[1], b.shape[1])

    def crop(x: np.ndarray) -> np.ndarray:
        top = (x.shape[0] - target_h) // 2
        left = (x.shape[1] - target_w) // 2
        if x.ndim == 2:
            return x[top : top + target_h, left : left + target_w]
        return x[top : top + target_h, left : left + target_w, ...]

    return crop(a), crop(b)


def center_crop_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if target_h > arr.shape[0] or target_w > arr.shape[1]:
        raise ValueError(
            f"Cannot crop array of shape {arr.shape} to {(target_h, target_w)}"
        )

    top = (arr.shape[0] - target_h) // 2
    left = (arr.shape[1] - target_w) // 2
    if arr.ndim == 2:
        return arr[top : top + target_h, left : left + target_w]
    return arr[top : top + target_h, left : left + target_w, ...]


def align_images_to_common_shape(
    reference: np.ndarray,
    images: list[np.ndarray],
    match_size: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    all_images = [reference, *images]
    reference_tail_shape = reference.shape[2:]
    reference_ndim = reference.ndim

    for idx, image in enumerate(images, start=1):
        if image.ndim != reference_ndim or image.shape[2:] != reference_tail_shape:
            raise ValueError(
                "All images must have the same channel layout as the reference. "
                f"Reference shape={reference.shape}, input {idx} shape={image.shape}."
            )

    if match_size == "center_crop":
        target_h = min(image.shape[0] for image in all_images)
        target_w = min(image.shape[1] for image in all_images)
        aligned = [center_crop_to_shape(image, target_h, target_w) for image in all_images]
    else:
        aligned = all_images

    target_shape = aligned[0].shape
    for idx, image in enumerate(aligned[1:], start=1):
        if image.shape != target_shape:
            raise ValueError(
                "Images must have the same full shape after alignment. "
                f"Reference shape={target_shape}, input {idx} shape={image.shape}."
            )

    return aligned[0], aligned[1:]


def infer_data_range_from_reference(reference: np.ndarray) -> float:
    reference_max = float(np.max(reference))
    if reference_max <= 0:
        raise ValueError(
            "Reference max must be positive when data_range is inferred from the reference image. "
            "Please provide --data_range explicitly for non-positive-valued references."
        )
    return reference_max


def calc_psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def calc_pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    a_flat = a_flat[mask]
    b_flat = b_flat[mask]

    if a_flat.size < 2:
        return float("nan")

    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return float("nan")

    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def calc_mssim(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    kwargs = {"data_range": data_range}
    if a.ndim == 3:
        kwargs["channel_axis"] = -1

    min_spatial = min(a.shape[0], a.shape[1])
    if min_spatial < 3:
        raise ValueError("SSIM requires image height and width to be at least 3 pixels")
    if min_spatial < 7:
        kwargs["win_size"] = min_spatial if min_spatial % 2 == 1 else min_spatial - 1

    return float(structural_similarity(a, b, **kwargs))


def image_stats(name: str, arr: np.ndarray) -> str:
    arr = arr.astype(np.float64, copy=False)
    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
    )


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:.6f}"


def prepare_display_image(arr: np.ndarray, display_min: float, display_max: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    denom = display_max - display_min
    if denom <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    scaled = (arr - display_min) / denom
    return np.clip(scaled, 0.0, 1.0)


def infer_display_bounds(
    images: list[np.ndarray],
    display_min: float | None,
    display_max: float | None,
    percentile_min: float | None = None,
    percentile_max: float | None = None,
) -> tuple[float, float]:
    finite_values = None

    # 分位数显示范围可以自动忽略极少数异常亮点，避免整张图都被压在 jet 的蓝色区域。
    if (
        (display_min is None and percentile_min is not None)
        or (display_max is None and percentile_max is not None)
    ):
        finite_arrays = [np.asarray(image, dtype=np.float64)[np.isfinite(image)] for image in images]
        finite_arrays = [values for values in finite_arrays if values.size > 0]
        if not finite_arrays:
            raise ValueError("No finite values found for percentile-based visualization bounds.")
        finite_values = np.concatenate(finite_arrays)

    if display_min is None:
        if percentile_min is not None:
            vmin = float(np.percentile(finite_values, percentile_min))
        else:
            vmin = min(float(np.min(image)) for image in images)
    else:
        vmin = float(display_min)

    if display_max is None:
        if percentile_max is not None:
            vmax = float(np.percentile(finite_values, percentile_max))
        else:
            vmax = max(float(np.max(image)) for image in images)
    else:
        vmax = float(display_max)

    return vmin, vmax


def create_display_norm(
    display_min: float,
    display_max: float,
    gamma: float,
):
    norm_max = display_max if display_max > display_min else display_min + 1e-12
    if gamma == 1.0:
        return mcolors.Normalize(vmin=display_min, vmax=norm_max)
    # gamma < 1 会把中高亮区域往暖色端推，gamma > 1 会让高亮区域更克制。
    return mcolors.PowerNorm(gamma=gamma, vmin=display_min, vmax=norm_max)


def render_image(
    ax: plt.Axes,
    image: np.ndarray,
    display_norm,
    cmap_name: str,
):
    # 二维图直接按原始数值和统一映射规则显示，这样 colorbar 才能对应真实数值。
    if image.ndim == 2:
        artist = ax.imshow(image, cmap=cmap_name, norm=display_norm)
    else:
        # 三通道图仍然走归一化显示；此时 colormap 不参与渲染。
        vis = prepare_display_image(image, display_norm.vmin, display_norm.vmax)
        artist = ax.imshow(vis)
    ax.axis("off")
    return artist


def add_shared_colorbar(
    fig: plt.Figure,
    axes: list[plt.Axes] | np.ndarray,
    cmap_name: str,
    display_norm,
) -> None:
    # 为整张图创建共享色条，让所有子图使用同一套“颜色 <-> 数值”映射。
    mappable = cm.ScalarMappable(norm=display_norm, cmap=cmap_name)
    mappable.set_array([])
    fig.colorbar(mappable, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02, shrink=0.96)


def infer_grid_shape(num_panels: int) -> tuple[int, int]:
    ncols = min(4, max(1, num_panels))
    nrows = math.ceil(num_panels / ncols)
    return nrows, ncols


def validate_roi(
    roi: list[int] | tuple[int, int, int, int] | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width and height must be positive, got {(x, y, w, h)}")
    if x < 0 or y < 0:
        raise ValueError(f"ROI x and y must be non-negative, got {(x, y, w, h)}")
    if x + w > image_shape[1] or y + h > image_shape[0]:
        raise ValueError(
            f"ROI {(x, y, w, h)} exceeds image bounds {(image_shape[1], image_shape[0])}."
        )
    return x, y, w, h


def crop_to_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    # 所有图像共用同一个 ROI，保证局部放大图仍然可以直接横向比较。
    x, y, w, h = roi
    if image.ndim == 2:
        return image[y : y + h, x : x + w]
    return image[y : y + h, x : x + w, ...]


def select_roi_interactively(
    image: np.ndarray,
    label: str,
    display_min: float | None,
    display_max: float | None,
    display_percentile_min: float | None,
    display_percentile_max: float | None,
    vis_cmap: str,
    vis_gamma: float,
) -> tuple[int, int, int, int]:
    # 弹出交互窗口让用户手动框选 ROI，避免再手算像素坐标。
    backend = plt.get_backend().lower()
    # 这里只拦截真正不能弹出交互窗口的后端；
    # TkAgg / QtAgg 虽然名字里带 agg，但它们本身仍然是可交互后端。
    non_interactive_backends = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
        "inline",
    }
    if backend in non_interactive_backends or backend.endswith("backend_inline"):
        raise RuntimeError(
            "Interactive ROI selection requires an interactive matplotlib backend, "
            f"but the current backend is {plt.get_backend()}."
        )

    vmin, vmax = infer_display_bounds(
        [image],
        display_min,
        display_max,
        percentile_min=display_percentile_min,
        percentile_max=display_percentile_max,
    )
    display_norm = create_display_norm(vmin, vmax, vis_gamma)

    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    if image.ndim == 2:
        ax.imshow(image, cmap=vis_cmap, norm=display_norm)
    else:
        vis = prepare_display_image(image, display_norm.vmin, display_norm.vmax)
        ax.imshow(vis)

    ax.set_title(f"{label}: drag ROI, press Enter to confirm, Esc to cancel")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    selection: dict[str, tuple[int, int, int, int] | None] = {"roi": None}
    info = fig.text(
        0.01,
        0.01,
        "No ROI selected yet",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    def on_select(eclick, erelease) -> None:
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0 = min(eclick.xdata, erelease.xdata)
        x1 = max(eclick.xdata, erelease.xdata)
        y0 = min(eclick.ydata, erelease.ydata)
        y1 = max(eclick.ydata, erelease.ydata)
        roi = (
            int(np.floor(x0)),
            int(np.floor(y0)),
            max(1, int(np.ceil(x1) - np.floor(x0))),
            max(1, int(np.ceil(y1) - np.floor(y0))),
        )
        selection["roi"] = validate_roi(roi, image.shape)
        roi_text = selection["roi"]
        info.set_text(f"Selected ROI: x={roi_text[0]}, y={roi_text[1]}, w={roi_text[2]}, h={roi_text[3]}")
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
        drag_from_anywhere=True,
    )

    def on_key(event) -> None:
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key in ("escape", "esc"):
            selection["roi"] = None
            plt.close(fig)

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    selector.set_active(False)
    plt.close(fig)

    if selection["roi"] is None:
        raise RuntimeError("No ROI selected. Drag a rectangle and press Enter to confirm.")
    return selection["roi"]


def save_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    img1_label: str,
    img2_label: str,
    save_path: str,
    psnr: float,
    mssim: float,
    pearson_r: float,
    display_min: float | None,
    display_max: float | None,
    display_percentile_min: float | None,
    display_percentile_max: float | None,
    vis_cmap: str,
    vis_colorbar: bool,
    vis_gamma: float,
    roi: tuple[int, int, int, int] | None = None,
) -> None:
    vmin, vmax = infer_display_bounds(
        [img1, img2],
        display_min,
        display_max,
        percentile_min=display_percentile_min,
        percentile_max=display_percentile_max,
    )
    display_norm = create_display_norm(vmin, vmax, vis_gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, image, label in zip(axes, (img1, img2), (img1_label, img2_label)):
        render_image(ax, image, display_norm, vis_cmap)
        if roi is not None:
            # 在全图上叠加 ROI 方框，方便定位局部放大区域来自哪里。
            ax.add_patch(
                Rectangle(
                    (roi[0], roi[1]),
                    roi[2],
                    roi[3],
                    fill=False,
                    edgecolor="#ff3b30",
                    linewidth=1.5,
                )
            )
        ax.set_title(label)

    # 只有二维单通道图像才适合挂共享色条，RGB 图不做这个映射。
    if vis_colorbar:
        if img1.ndim != 2 or img2.ndim != 2:
            raise ValueError("--vis_colorbar is only supported for 2D grayscale images.")
        add_shared_colorbar(fig, axes, vis_cmap, display_norm)

    # 不再添加整张图的大标题，避免遮挡主体显示区域。
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_batch_visualization(
    reference: np.ndarray,
    reference_label: str,
    items: list[dict[str, object]],
    save_path: str,
    display_min: float | None,
    display_max: float | None,
    display_percentile_min: float | None,
    display_percentile_max: float | None,
    vis_cmap: str,
    vis_colorbar: bool,
    vis_gamma: float,
    roi: tuple[int, int, int, int] | None = None,
    show_metrics_in_title: bool = True,
) -> None:
    # 批量模式按输入顺序先画所有待比较图，最后再画 reference。
    panel_entries = [
        *items,
        {
            "label": f"{reference_label}\n(reference)",
            "image": reference,
            "is_reference": True,
        },
    ]
    images = [entry["image"] for entry in panel_entries]
    vmin, vmax = infer_display_bounds(
        images,
        display_min,
        display_max,
        percentile_min=display_percentile_min,
        percentile_max=display_percentile_max,
    )
    display_norm = create_display_norm(vmin, vmax, vis_gamma)

    total_panels = len(images)
    nrows, ncols = infer_grid_shape(total_panels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.6 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes[:total_panels], panel_entries):
        render_image(ax, entry["image"], display_norm, vis_cmap)
        if roi is not None:
            # 批量模式下每张图都标出同一块 ROI，便于和单独 ROI 图对应查看。
            ax.add_patch(
                Rectangle(
                    (roi[0], roi[1]),
                    roi[2],
                    roi[3],
                    fill=False,
                    edgecolor="#ff3b30",
                    linewidth=1.5,
                )
            )
        if entry.get("is_reference", False):
            # reference 只展示名字，不重复写指标，避免和待比较图混淆。
            ax.set_title(str(entry["label"]), fontsize=10)
        elif not show_metrics_in_title:
            # ROI 放大图只保留标题，避免指标文字把局部细节挤得太满。
            ax.set_title(str(entry["label"]), fontsize=10)
        else:
            ax.set_title(
                (
                    f"{entry['label']}\n"
                    f"PSNR={format_metric(float(entry['psnr']))} dB\n"
                    f"MSSIM={format_metric(float(entry['mssim']))} | "
                    f"R={format_metric(float(entry['pearson_r']))}"
                ),
                fontsize=10,
            )

    for ax in axes[total_panels:]:
        ax.axis("off")

    # 批量模式下色条同样与所有子图共享，便于直接横向比较不同方法的亮度数值。
    if vis_colorbar:
        if reference.ndim != 2:
            raise ValueError("--vis_colorbar is only supported for 2D grayscale images.")
        add_shared_colorbar(fig, axes[:total_panels], vis_cmap, display_norm)

    # 不再添加整张图的大标题，避免遮挡主体显示区域。
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def derive_input_labels(paths: list[str], explicit_names: list[str] | None) -> list[str]:
    if explicit_names is not None:
        if len(explicit_names) != len(paths):
            raise ValueError(
                f"--input_names expects {len(paths)} names, got {len(explicit_names)}."
            )
        return explicit_names
    return [Path(path).stem for path in paths]


def save_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    img1_label: str,
    img2_label: str,
    save_path: str,
    psnr: float,
    mssim: float,
    pearson_r: float,
    display_min: float | None,
    display_max: float | None,
    display_percentile_min: float | None,
    display_percentile_max: float | None,
    vis_cmap: str,
    vis_colorbar: bool,
    vis_gamma: float,
    roi: tuple[int, int, int, int] | None = None,
) -> None:
    vmin, vmax = infer_display_bounds(
        [img1, img2],
        display_min,
        display_max,
        percentile_min=display_percentile_min,
        percentile_max=display_percentile_max,
    )
    display_norm = create_display_norm(vmin, vmax, vis_gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, image, label in zip(axes, (img1, img2), (img1_label, img2_label)):
        render_image(ax, image, display_norm, vis_cmap)
        if roi is not None:
            # 在全图上叠加 ROI 方框，方便定位局部放大区域来自哪里。
            ax.add_patch(
                Rectangle(
                    (roi[0], roi[1]),
                    roi[2],
                    roi[3],
                    fill=False,
                    edgecolor="#ff3b30",
                    linewidth=1.5,
                )
            )
        # 这里只保留一个简洁标题，不再把指标写到子图上。
        ax.set_title(label)

    # 为了让整张图更简洁，colorbar 不再显示；
    # --vis_colorbar 仅为兼容旧命令而保留，不再影响出图。
    _ = (psnr, mssim, pearson_r, vis_colorbar)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_batch_visualization(
    reference: np.ndarray,
    reference_label: str,
    items: list[dict[str, object]],
    save_path: str,
    display_min: float | None,
    display_max: float | None,
    display_percentile_min: float | None,
    display_percentile_max: float | None,
    vis_cmap: str,
    vis_colorbar: bool,
    vis_gamma: float,
    roi: tuple[int, int, int, int] | None = None,
    show_metrics_in_title: bool = True,
) -> None:
    # 按输入顺序先画所有待比较图，最后画 reference，和当前使用习惯保持一致。
    panel_entries = [
        *items,
        {
            "label": reference_label,
            "image": reference,
            "is_reference": True,
        },
    ]
    images = [entry["image"] for entry in panel_entries]
    vmin, vmax = infer_display_bounds(
        images,
        display_min,
        display_max,
        percentile_min=display_percentile_min,
        percentile_max=display_percentile_max,
    )
    display_norm = create_display_norm(vmin, vmax, vis_gamma)

    total_panels = len(images)
    nrows, ncols = infer_grid_shape(total_panels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.6 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes[:total_panels], panel_entries):
        render_image(ax, entry["image"], display_norm, vis_cmap)
        if roi is not None:
            # 所有子图都使用同一块 ROI 方框，方便全图和 ROI 图对应查看。
            ax.add_patch(
                Rectangle(
                    (roi[0], roi[1]),
                    roi[2],
                    roi[3],
                    fill=False,
                    edgecolor="#ff3b30",
                    linewidth=1.5,
                )
            )
        # 全图和 ROI 图都只保留一个标题，不再展示 PSNR / MSSIM / R。
        ax.set_title(str(entry["label"]), fontsize=10)

    for ax in axes[total_panels:]:
        ax.axis("off")

    # 这里同样不再显示 colorbar；旧参数继续保留兼容即可。
    _ = (vis_colorbar, show_metrics_in_title)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR, mean SSIM (MSSIM), and Pearson R for a single pair or for "
            "multiple inputs against one reference."
        )
    )
    parser.add_argument("--img1", help="Path to image/array 1 for single-pair mode")
    parser.add_argument("--img2", help="Path to image/array 2 for single-pair mode")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference image/array path for batch mode.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="One or more input images/arrays to compare against --reference.",
    )
    parser.add_argument(
        "--input_names",
        nargs="+",
        default=None,
        help="Optional display names for --inputs, in the same order.",
    )
    parser.add_argument(
        "--reference_name",
        type=str,
        default=None,
        help="Optional display name for --reference in batch mode.",
    )
    parser.add_argument(
        "--ref",
        choices=("img1", "img2"),
        default="img2",
        help="Which input should be treated as the reference image in single-pair mode.",
    )
    parser.add_argument(
        "--data_range",
        type=float,
        default=None,
        help="Metric data range. Defaults to the reference image max.",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=None,
        help="Optional lower clip applied to all compared images before metrics.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=None,
        help="Optional upper clip applied to all compared images before metrics.",
    )
    parser.add_argument(
        "--match_size",
        choices=("center_crop", "error"),
        default="center_crop",
        help="How to handle different image sizes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON only.",
    )
    parser.add_argument(
        "--save_vis",
        type=str,
        default=None,
        help="Optional path to save a visualization for the compared images.",
    )
    parser.add_argument(
        "--save_roi_vis",
        type=str,
        default=None,
        help="Optional path to save a cropped ROI visualization after manual selection.",
    )
    parser.add_argument(
        "--interactive_roi",
        action="store_true",
        help="Manually draw one ROI on the reference image and reuse it for all compared images.",
    )
    parser.add_argument(
        "--vis_min",
        type=float,
        default=None,
        help="Optional visualization lower bound shared by all visualized images.",
    )
    parser.add_argument(
        "--vis_max",
        type=float,
        default=None,
        help="Optional visualization upper bound shared by all visualized images.",
    )
    parser.add_argument(
        "--vis_percentile_min",
        type=float,
        default=None,
        help="Optional percentile lower bound for visualization when --vis_min is not set.",
    )
    parser.add_argument(
        "--vis_percentile_max",
        type=float,
        default=None,
        help="Optional percentile upper bound for visualization when --vis_max is not set.",
    )
    parser.add_argument(
        "--vis_cmap",
        choices=("gray", "viridis", "turbo", "jet"),
        default="gray",
        help="Colormap used for 2D visualizations.",
    )
    parser.add_argument(
        "--vis_colorbar",
        action="store_true",
        help="Deprecated flag kept for backward compatibility; colorbar is no longer drawn.",
    )
    parser.add_argument(
        "--vis_gamma",
        type=float,
        default=1.0,
        help="Gamma used for 2D visualization mapping. Values < 1 warm up mid/high intensities.",
    )
    args = parser.parse_args()

    if args.vis_gamma <= 0:
        parser.error("--vis_gamma must be positive.")
    if args.vis_percentile_min is not None and not 0.0 <= args.vis_percentile_min <= 100.0:
        parser.error("--vis_percentile_min must be within [0, 100].")
    if args.vis_percentile_max is not None and not 0.0 <= args.vis_percentile_max <= 100.0:
        parser.error("--vis_percentile_max must be within [0, 100].")
    if (
        args.vis_percentile_min is not None
        and args.vis_percentile_max is not None
        and args.vis_percentile_min >= args.vis_percentile_max
    ):
        parser.error("--vis_percentile_min must be smaller than --vis_percentile_max.")
    if args.save_roi_vis is not None and not args.interactive_roi:
        parser.error("--save_roi_vis requires --interactive_roi.")

    batch_mode = args.reference is not None or args.inputs is not None

    if batch_mode:
        if args.reference is None or args.inputs is None:
            parser.error("Batch mode requires both --reference and --inputs.")
        if args.img1 is not None or args.img2 is not None:
            parser.error("Do not mix --img1/--img2 with --reference/--inputs.")

        reference = normalize_image_shape(load_array(args.reference))
        inputs = [normalize_image_shape(load_array(path)) for path in args.inputs]
        reference, inputs = align_images_to_common_shape(reference, inputs, args.match_size)

        reference = reference.astype(np.float64, copy=False)
        inputs = [image.astype(np.float64, copy=False) for image in inputs]

        if args.clip_min is not None or args.clip_max is not None:
            clip_min = -np.inf if args.clip_min is None else args.clip_min
            clip_max = np.inf if args.clip_max is None else args.clip_max
            reference = np.clip(reference, clip_min, clip_max)
            inputs = [np.clip(image, clip_min, clip_max) for image in inputs]

        reference_label = (
            args.reference_name if args.reference_name is not None else Path(args.reference).stem
        )
        # 在 reference 上手动框选一次 ROI，后续所有输入图复用同一块区域。
        roi = None
        if args.interactive_roi:
            roi = select_roi_interactively(
                image=reference,
                label=reference_label,
                display_min=args.vis_min,
                display_max=args.vis_max,
                display_percentile_min=args.vis_percentile_min,
                display_percentile_max=args.vis_percentile_max,
                vis_cmap=args.vis_cmap,
                vis_gamma=args.vis_gamma,
            )

        input_labels = derive_input_labels(args.inputs, args.input_names)
        reference_max = float(np.max(reference))
        data_range = (
            float(args.data_range)
            if args.data_range is not None
            else infer_data_range_from_reference(reference)
        )

        results = []
        vis_items = []
        for path, label, image in zip(args.inputs, input_labels, inputs):
            psnr = calc_psnr(image, reference, data_range)
            mssim = calc_mssim(image, reference, data_range)
            pearson_r = calc_pearson_r(image, reference)
            results.append(
                {
                    "name": label,
                    "path": os.path.abspath(path),
                    "shape": list(image.shape),
                    "psnr": psnr,
                    "mssim": mssim,
                    "pearson_r": pearson_r,
                }
            )
            vis_items.append(
                {
                    "label": label,
                    "image": image,
                    "psnr": psnr,
                    "mssim": mssim,
                    "pearson_r": pearson_r,
                }
            )

        result = {
            "mode": "batch",
            "reference": os.path.abspath(args.reference),
            "reference_name": reference_label,
            "shape": list(reference.shape),
            "reference_max": reference_max,
            "data_range": data_range,
            "results": results,
        }
        if roi is not None:
            result["roi"] = {"xywh": list(roi)}

        if args.save_vis is not None:
            save_batch_visualization(
                reference=reference,
                reference_label=reference_label,
                items=vis_items,
                save_path=args.save_vis,
                display_min=args.vis_min,
                display_max=args.vis_max,
                display_percentile_min=args.vis_percentile_min,
                display_percentile_max=args.vis_percentile_max,
                vis_cmap=args.vis_cmap,
                vis_colorbar=args.vis_colorbar,
                vis_gamma=args.vis_gamma,
                roi=roi,
            )
            result["visualization"] = os.path.abspath(args.save_vis)

        if roi is not None and args.save_roi_vis is not None:
            # ROI 图沿用同一块矩形区域，对所有方法和 reference 做局部放大比较。
            reference_roi = crop_to_roi(reference, roi)
            roi_results = []
            roi_vis_items = []
            for path, label, image in zip(args.inputs, input_labels, inputs):
                image_roi = crop_to_roi(image, roi)
                roi_psnr = calc_psnr(image_roi, reference_roi, data_range)
                roi_mssim = calc_mssim(image_roi, reference_roi, data_range)
                roi_pearson_r = calc_pearson_r(image_roi, reference_roi)
                roi_results.append(
                    {
                        "name": label,
                        "path": os.path.abspath(path),
                        "shape": list(image_roi.shape),
                        "psnr": roi_psnr,
                        "mssim": roi_mssim,
                        "pearson_r": roi_pearson_r,
                    }
                )
                roi_vis_items.append(
                    {
                        "label": label,
                        "image": image_roi,
                        "psnr": roi_psnr,
                        "mssim": roi_mssim,
                        "pearson_r": roi_pearson_r,
                    }
                )

            save_batch_visualization(
                reference=reference_roi,
                reference_label=f"{reference_label} ROI",
                items=roi_vis_items,
                save_path=args.save_roi_vis,
                display_min=args.vis_min,
                display_max=args.vis_max,
                display_percentile_min=args.vis_percentile_min,
                display_percentile_max=args.vis_percentile_max,
                vis_cmap=args.vis_cmap,
                vis_colorbar=args.vis_colorbar,
                vis_gamma=args.vis_gamma,
                roi=None,
                show_metrics_in_title=False,
            )
            result["roi"]["shape"] = list(reference_roi.shape)
            result["roi"]["results"] = roi_results
            result["roi"]["visualization"] = os.path.abspath(args.save_roi_vis)

        if args.json:
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return

        print(image_stats(reference_label, reference))
        print(f"reference path = {os.path.abspath(args.reference)}")
        print(f"reference max = {reference_max:.6f}")
        print(f"data_range used = {data_range:.6f}")
        for item in results:
            print()
            print(f"{item['name']}: {item['path']}")
            print(f"shape = {tuple(item['shape'])}")
            print(f"PSNR = {format_metric(float(item['psnr']))} dB")
            print(f"MSSIM = {format_metric(float(item['mssim']))}")
            print(f"Pearson R = {format_metric(float(item['pearson_r']))}")
        if args.save_vis is not None:
            print()
            print(f"Visualization saved = {os.path.abspath(args.save_vis)}")
        if roi is not None:
            print()
            print(f"Selected ROI = x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        if roi is not None and args.save_roi_vis is not None:
            print(f"ROI visualization saved = {os.path.abspath(args.save_roi_vis)}")
        return

    if args.img1 is None or args.img2 is None:
        parser.error("Single-pair mode requires both --img1 and --img2.")

    img1 = normalize_image_shape(load_array(args.img1))
    img2 = normalize_image_shape(load_array(args.img2))
    img1, (img2,) = align_images_to_common_shape(img1, [img2], args.match_size)

    img1 = img1.astype(np.float64, copy=False)
    img2 = img2.astype(np.float64, copy=False)

    if args.clip_min is not None or args.clip_max is not None:
        clip_min = -np.inf if args.clip_min is None else args.clip_min
        clip_max = np.inf if args.clip_max is None else args.clip_max
        img1 = np.clip(img1, clip_min, clip_max)
        img2 = np.clip(img2, clip_min, clip_max)

    reference = img1 if args.ref == "img1" else img2
    reference_label = "img1" if args.ref == "img1" else "img2"
    # 单图模式下同样在 reference 图上手动框选 ROI。
    roi = None
    if args.interactive_roi:
        roi = select_roi_interactively(
            image=reference,
            label=reference_label,
            display_min=args.vis_min,
            display_max=args.vis_max,
            display_percentile_min=args.vis_percentile_min,
            display_percentile_max=args.vis_percentile_max,
            vis_cmap=args.vis_cmap,
            vis_gamma=args.vis_gamma,
        )

    reference_max = float(np.max(reference))
    data_range = (
        float(args.data_range)
        if args.data_range is not None
        else infer_data_range_from_reference(reference)
    )

    psnr = calc_psnr(img1, img2, data_range)
    mssim = calc_mssim(img1, img2, data_range)
    pearson_r = calc_pearson_r(img1, img2)

    result = {
        "mode": "single",
        "img1": os.path.abspath(args.img1),
        "img2": os.path.abspath(args.img2),
        "reference": args.ref,
        "shape": list(img1.shape),
        "reference_max": reference_max,
        "data_range": data_range,
        "psnr": psnr,
        "mssim": mssim,
        "pearson_r": pearson_r,
    }
    if roi is not None:
        result["roi"] = {"xywh": list(roi)}

    if args.save_vis is not None:
        save_visualization(
            img1=img1,
            img2=img2,
            img1_label="img1",
            img2_label=f"img2 ({args.ref})" if args.ref == "img2" else "img2",
            save_path=args.save_vis,
            psnr=psnr,
            mssim=mssim,
            pearson_r=pearson_r,
            display_min=args.vis_min,
            display_max=args.vis_max,
            display_percentile_min=args.vis_percentile_min,
            display_percentile_max=args.vis_percentile_max,
            vis_cmap=args.vis_cmap,
            vis_colorbar=args.vis_colorbar,
            vis_gamma=args.vis_gamma,
            roi=roi,
        )
        result["visualization"] = os.path.abspath(args.save_vis)

    if roi is not None and args.save_roi_vis is not None:
        img1_roi = crop_to_roi(img1, roi)
        img2_roi = crop_to_roi(img2, roi)
        roi_psnr = calc_psnr(img1_roi, img2_roi, data_range)
        roi_mssim = calc_mssim(img1_roi, img2_roi, data_range)
        roi_pearson_r = calc_pearson_r(img1_roi, img2_roi)
        save_visualization(
            img1=img1_roi,
            img2=img2_roi,
            img1_label="img1 ROI",
            img2_label=f"img2 ROI ({args.ref})" if args.ref == "img2" else "img2 ROI",
            save_path=args.save_roi_vis,
            psnr=roi_psnr,
            mssim=roi_mssim,
            pearson_r=roi_pearson_r,
            display_min=args.vis_min,
            display_max=args.vis_max,
            display_percentile_min=args.vis_percentile_min,
            display_percentile_max=args.vis_percentile_max,
            vis_cmap=args.vis_cmap,
            vis_colorbar=args.vis_colorbar,
            vis_gamma=args.vis_gamma,
            roi=None,
        )
        result["roi"]["shape"] = list(img1_roi.shape)
        result["roi"]["psnr"] = roi_psnr
        result["roi"]["mssim"] = roi_mssim
        result["roi"]["pearson_r"] = roi_pearson_r
        result["roi"]["visualization"] = os.path.abspath(args.save_roi_vis)

    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return

    print(image_stats("img1", img1))
    print(image_stats("img2", img2))
    print(f"reference = {args.ref}")
    print(f"reference max = {reference_max:.6f}")
    print(f"data_range used = {data_range:.6f}")
    print(f"PSNR = {format_metric(psnr)} dB")
    print(f"MSSIM = {format_metric(mssim)}")
    print(f"Pearson R = {format_metric(pearson_r)}")
    if args.save_vis is not None:
        print(f"Visualization saved = {os.path.abspath(args.save_vis)}")
    if roi is not None:
        print(f"Selected ROI = x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    if roi is not None and args.save_roi_vis is not None:
        print(f"ROI visualization saved = {os.path.abspath(args.save_roi_vis)}")


if __name__ == "__main__":
    main()
