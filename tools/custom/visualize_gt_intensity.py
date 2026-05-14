import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize projected Waymo LiDAR intensity maps without running training."
    )
    parser.add_argument("--scene", required=True, help="Scene id, e.g. 023")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--camera", type=int, default=0, help="Camera id")
    parser.add_argument(
        "--all_cameras",
        action="store_true",
        help="Visualize cameras 0-4 for the requested frame.",
    )
    parser.add_argument(
        "--data_root",
        default="data/waymo/processed/training",
        help="Root directory containing processed Waymo scene folders.",
    )
    parser.add_argument(
        "--output_dir",
        default="debug/gt_intensity",
        help="Directory where visualizations are saved.",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name. Training visualization uses turbo.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.55,
        help="Intensity overlay opacity on top of RGB image.",
    )
    parser.add_argument(
        "--scale",
        choices=["fixed", "percentile"],
        default="fixed",
        help="'fixed' matches training visualization by clipping to [0, 1]. "
        "'percentile' boosts contrast using nonzero value percentiles.",
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=1.0,
        help="Lower nonzero percentile used when --scale percentile is selected.",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=99.0,
        help="Upper nonzero percentile used when --scale percentile is selected.",
    )
    return parser.parse_args()


def find_rgb_path(scene_dir: Path, frame_name: str) -> Path:
    image_dir = scene_dir / "images"
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{frame_name}{ext}"
        if path.exists():
            return path
    return image_dir / f"{frame_name}.jpg"


def load_intensity(path: Path) -> np.ndarray:
    intensity = np.load(path)
    if intensity.ndim == 3 and intensity.shape[-1] == 1:
        intensity = intensity[..., 0]
    if intensity.ndim != 2:
        raise ValueError(f"Expected intensity shape [H, W] or [H, W, 1], got {intensity.shape}")
    return intensity.astype(np.float32)


def normalize_intensity(
    intensity: np.ndarray,
    scale: str,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    nonzero = intensity > 0
    if not nonzero.any():
        return np.zeros_like(intensity, dtype=np.float32)

    if scale == "fixed":
        return np.clip(intensity, 0.0, 1.0)

    values = intensity[nonzero]
    vmin = float(np.percentile(values, percentile_low))
    vmax = float(np.percentile(values, percentile_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalized = np.clip((intensity - vmin) / (vmax - vmin), 0.0, 1.0)
    normalized[~nonzero] = 0.0
    return normalized.astype(np.float32)


def intensity_to_rgb(
    intensity: np.ndarray,
    cmap_name: str,
    scale: str,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    normalized = normalize_intensity(intensity, scale, percentile_low, percentile_high)
    colored = plt.get_cmap(cmap_name)(normalized)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    colored[intensity <= 0] = 0
    return colored


def blend(rgb: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - alpha) * rgb.astype(np.float32) + alpha * overlay.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_histogram(intensity: np.ndarray, output_path: Path):
    nonzero = intensity[intensity > 0]
    fig = plt.figure(figsize=(8, 5))
    if nonzero.size > 0:
        plt.hist(nonzero, bins=80)
        plt.xlabel("nonzero intensity value")
        plt.ylabel("pixel count")
    else:
        plt.text(0.5, 0.5, "no nonzero intensity pixels", ha="center", va="center")
        plt.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_stats(frame_name: str, intensity: np.ndarray):
    flat = intensity.reshape(-1)
    finite = np.isfinite(flat)
    nonzero = finite & (flat > 0)
    coverage = float(nonzero.mean()) if flat.size > 0 else 0.0
    print(f"{frame_name}: shape={intensity.shape}, nonzero={int(nonzero.sum())}/{flat.size}, coverage={coverage:.6f}")
    if nonzero.any():
        values = flat[nonzero]
        print(
            "  nonzero min/p50/p95/p99/max/mean = "
            f"{values.min():.6f} / {np.percentile(values, 50):.6f} / "
            f"{np.percentile(values, 95):.6f} / {np.percentile(values, 99):.6f} / "
            f"{values.max():.6f} / {values.mean():.6f}"
        )


def visualize_one(scene_dir: Path, frame: int, camera: int, output_root: Path, args):
    frame_name = f"{frame:03d}_{camera}"
    intensity_path = scene_dir / "intensity" / f"{frame_name}.npy"
    rgb_path = find_rgb_path(scene_dir, frame_name)

    if not intensity_path.exists():
        raise FileNotFoundError(f"Intensity file not found: {intensity_path}")

    output_dir = output_root / scene_dir.name / frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    intensity = load_intensity(intensity_path)
    print_stats(frame_name, intensity)

    intensity_rgb = intensity_to_rgb(
        intensity,
        cmap_name=args.cmap,
        scale=args.scale,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )
    mask_rgb = ((intensity > 0)[..., None].repeat(3, axis=-1) * 255).astype(np.uint8)

    Image.fromarray(intensity_rgb).save(output_dir / "gt_intensity.png")
    Image.fromarray(mask_rgb).save(output_dir / "gt_intensity_mask.png")
    save_histogram(intensity, output_dir / "gt_intensity_hist.png")

    if rgb_path.exists():
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        Image.fromarray(rgb).save(output_dir / "rgb.png")
        if rgb.shape[:2] == intensity_rgb.shape[:2]:
            Image.fromarray(blend(rgb, intensity_rgb, args.overlay_alpha)).save(
                output_dir / "gt_intensity_overlay.png"
            )
        else:
            print(
                f"  overlay skipped: RGB shape {rgb.shape[:2]} does not match "
                f"intensity shape {intensity_rgb.shape[:2]}"
            )
    else:
        print(f"  RGB image not found, overlay skipped: {rgb_path}")

    print(f"  saved: {output_dir}")


def main():
    args = parse_args()
    scene = str(args.scene).zfill(3)
    scene_dir = Path(args.data_root) / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    cameras = range(5) if args.all_cameras else [args.camera]
    output_root = Path(args.output_dir)
    for camera in cameras:
        visualize_one(scene_dir, args.frame, camera, output_root, args)


if __name__ == "__main__":
    main()
