import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and visualize normal/material/intensity preprocessing outputs"
    )
    parser.add_argument("--scene", required=True, help="Scene id, e.g. 023")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--camera", type=int, default=0, help="Camera id")
    parser.add_argument(
        "--data_root",
        default="data/waymo/processed/training",
        help="Root directory of processed Waymo scenes",
    )
    parser.add_argument(
        "--output_dir",
        default="debug/prior_vis",
        help="Directory to save visualization outputs",
    )
    parser.add_argument(
        "--skip_materials",
        action="store_true",
        help="Validate normals only and skip material file checks",
    )
    parser.add_argument(
        "--skip_intensity",
        action="store_true",
        help="Skip projected intensity map checks",
    )
    return parser.parse_args()


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_image(array: np.ndarray, path: Path):
    Image.fromarray(array).save(path)


def normal_to_rgb(normals: np.ndarray) -> np.ndarray:
    clipped = np.clip(normals, -1.0, 1.0)
    return ((clipped + 1.0) * 127.5).astype(np.uint8)


def blend(base_image: np.ndarray, overlay_image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = base_image.astype(np.float32)
    overlay = overlay_image.astype(np.float32)
    return np.clip((1.0 - alpha) * base + alpha * overlay, 0, 255).astype(np.uint8)


def print_normal_stats(normals: np.ndarray):
    flat = normals.reshape(-1, 3)
    magnitudes = np.linalg.norm(flat, axis=-1)
    finite = np.all(np.isfinite(flat), axis=-1)
    valid = finite & (magnitudes > 0)
    print(f"Normal shape: {normals.shape}")
    print(f"Finite normals: {int(finite.sum())} / {len(finite)}")
    print(f"Valid normals: {int(valid.sum())} / {len(valid)}")
    if valid.any():
        valid_normals = flat[valid]
        valid_magnitudes = magnitudes[valid]
        near_unit = np.abs(valid_magnitudes - 1.0) < 0.05
        within_range = np.all(np.abs(valid_normals) <= 1.05, axis=-1)
        print("Normal component ranges:")
        print(f"  x: {valid_normals[:, 0].min():.6f} to {valid_normals[:, 0].max():.6f}")
        print(f"  y: {valid_normals[:, 1].min():.6f} to {valid_normals[:, 1].max():.6f}")
        print(f"  z: {valid_normals[:, 2].min():.6f} to {valid_normals[:, 2].max():.6f}")
        print(f"Normal magnitude mean: {valid_magnitudes.mean():.6f}")
        print(f"Near-unit normals (|norm-1| < 0.05): {int(near_unit.sum())} / {len(valid_magnitudes)}")
        print(f"Components within [-1.05, 1.05]: {int(within_range.sum())} / {len(valid_normals)}")


def print_intensity_stats(intensity: np.ndarray):
    values = intensity.reshape(-1)
    finite = np.isfinite(values)
    nonzero = finite & (values > 0)
    print(f"Intensity shape: {intensity.shape}")
    print(f"Finite intensity values: {int(finite.sum())} / {len(finite)}")
    print(f"Nonzero intensity pixels: {int(nonzero.sum())} / {len(values)}")
    if nonzero.any():
        nonzero_values = values[nonzero]
        print(f"Nonzero min/max: {nonzero_values.min():.6f} / {nonzero_values.max():.6f}")
        print(f"Nonzero mean: {nonzero_values.mean():.6f}")
        print(f"Nonzero p50/p95: {np.percentile(nonzero_values, 50):.6f} / {np.percentile(nonzero_values, 95):.6f}")


def validate_normals(normals: np.ndarray):
    issues = []
    if normals.ndim != 3 or normals.shape[-1] != 3:
        issues.append(f"Expected shape [H, W, 3], got {normals.shape}")
        return issues

    if not np.isfinite(normals).all():
        issues.append("Normals contain NaN or Inf values")

    flat = normals.reshape(-1, 3)
    magnitudes = np.linalg.norm(flat, axis=-1)
    valid = magnitudes > 0
    if not valid.any():
        issues.append("All normals are zero vectors")
        return issues

    valid_normals = flat[valid]
    valid_magnitudes = magnitudes[valid]
    if np.any(np.abs(valid_normals) > 1.1):
        issues.append("Normal components fall outside the expected [-1, 1] range")
    if valid_magnitudes.mean() < 0.9 or valid_magnitudes.mean() > 1.1:
        issues.append(f"Mean normal magnitude is suspicious: {valid_magnitudes.mean():.4f}")
    if np.std(valid_normals[:, 2]) < 1e-3:
        issues.append("Normals appear nearly constant; z-component variance is near zero")
    return issues


def validate_intensity(intensity: np.ndarray):
    issues = []
    if intensity.ndim == 3 and intensity.shape[-1] == 1:
        intensity = intensity[..., 0]
    if intensity.ndim != 2:
        issues.append(f"Expected shape [H, W] or [H, W, 1], got {intensity.shape}")
        return issues
    if not np.isfinite(intensity).all():
        issues.append("Intensity map contains NaN or Inf values")
    if np.any(intensity < 0):
        issues.append("Intensity map contains negative values")
    nonzero = intensity > 0
    if not nonzero.any():
        issues.append("All projected intensity pixels are zero")
        return issues
    coverage = float(nonzero.mean())
    if coverage < 1e-4:
        issues.append(f"Projected intensity coverage is suspiciously sparse: {coverage:.6f}")
    if np.percentile(intensity[nonzero], 95) <= 0:
        issues.append("Projected nonzero intensity values are not positive")
    return issues


def save_normal_visuals(normals: np.ndarray, rgb: np.ndarray, output_dir: Path):
    normal_rgb = normal_to_rgb(normals)
    save_image(normal_rgb, output_dir / "normal_rgb.png")

    fig = plt.figure(figsize=(14, 6))
    for idx, name in enumerate(["x", "y", "z"], start=1):
        plt.subplot(1, 3, idx)
        plt.imshow(normals[..., idx - 1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        plt.title(f"normal_{name}")
        plt.axis("off")
    save_figure(fig, output_dir / "normal_components.png")

    if rgb.shape[:2] == normals.shape[:2]:
        save_image(blend(rgb, normal_rgb), output_dir / "normal_overlay.png")
    else:
        print(
            "Normal overlay skipped because RGB and normal shapes do not match: "
            f"rgb={rgb.shape[:2]}, normal={normals.shape[:2]}"
        )


def intensity_to_rgb(intensity: np.ndarray) -> np.ndarray:
    if intensity.ndim == 3 and intensity.shape[-1] == 1:
        intensity = intensity[..., 0]
    colored = np.zeros((*intensity.shape, 3), dtype=np.uint8)
    nonzero = intensity > 0
    if not nonzero.any():
        return colored
    vmin = float(intensity[nonzero].min())
    vmax = float(np.percentile(intensity[nonzero], 99))
    vmax = max(vmax, vmin + 1e-6)
    normed = np.clip((intensity - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap("inferno")
    colored = (cmap(normed)[..., :3] * 255).astype(np.uint8)
    colored[~nonzero] = 0
    return colored


def save_intensity_visuals(intensity: np.ndarray, rgb: np.ndarray, output_dir: Path):
    intensity_rgb = intensity_to_rgb(intensity)
    save_image(intensity_rgb, output_dir / "intensity_rgb.png")

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(intensity[..., 0] if intensity.ndim == 3 else intensity, cmap="inferno")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("projected_intensity")
    plt.axis("off")
    save_figure(fig, output_dir / "intensity_heatmap.png")

    nonzero_mask = ((intensity[..., 0] if intensity.ndim == 3 else intensity) > 0).astype(np.uint8) * 255
    save_image(nonzero_mask, output_dir / "intensity_mask.png")

    if rgb.shape[:2] == intensity_rgb.shape[:2]:
        save_image(blend(rgb, intensity_rgb), output_dir / "intensity_overlay.png")
    else:
        print(
            "Intensity overlay skipped because RGB and intensity shapes do not match: "
            f"rgb={rgb.shape[:2]}, intensity={intensity_rgb.shape[:2]}"
        )


def main():
    args = parse_args()
    scene = str(args.scene).zfill(3)
    frame_name = f"{args.frame:03d}_{args.camera}"

    scene_dir = Path(args.data_root) / scene
    normal_path = scene_dir / "normals" / "normal_npy" / f"{frame_name}_pred.npy"
    image_path = scene_dir / "images" / f"{frame_name}.jpg"
    albedo_path = scene_dir / "albedo_rgbx" / f"{frame_name}.jpg"
    roughness_path = scene_dir / "rough_rgbx" / f"{frame_name}.jpg"
    intensity_path = scene_dir / "intensity" / f"{frame_name}.npy"

    if not normal_path.exists():
        raise FileNotFoundError(f"Normal file not found: {normal_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    if not args.skip_intensity and not intensity_path.exists():
        raise FileNotFoundError(f"Intensity supervision file not found: {intensity_path}")
    if not args.skip_materials:
        if not albedo_path.exists():
            raise FileNotFoundError(f"Albedo prior file not found: {albedo_path}")
        if not roughness_path.exists():
            raise FileNotFoundError(f"Roughness prior file not found: {roughness_path}")

    output_dir = Path(args.output_dir) / scene / frame_name
    output_dir.mkdir(parents=True, exist_ok=True)

    normals = np.load(normal_path)
    rgb = np.array(Image.open(image_path).convert("RGB"))

    print(f"Normal file: {normal_path}")
    print(f"RGB file: {image_path}")
    print_normal_stats(normals)

    issues = validate_normals(normals)
    if issues:
        print("Normal validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Normal validation: PASS")

    save_normal_visuals(normals, rgb, output_dir)

    if not args.skip_intensity:
        intensity = np.load(intensity_path)
        print(f"Intensity file: {intensity_path}")
        print_intensity_stats(intensity)
        intensity_issues = validate_intensity(intensity)
        if intensity_issues:
            print("Intensity validation issues:")
            for issue in intensity_issues:
                print(f"  - {issue}")
        else:
            print("Intensity validation: PASS")
        save_intensity_visuals(intensity, rgb, output_dir)

    if not args.skip_materials:
        albedo = np.array(Image.open(albedo_path).convert("RGB"))
        roughness = np.array(Image.open(roughness_path).convert("RGB"))
        print(f"Albedo file: {albedo_path}")
        print(f"Roughness file: {roughness_path}")
        print(f"Albedo shape: {albedo.shape}")
        print(f"Roughness shape: {roughness.shape}")
        save_image(albedo, output_dir / "albedo_rgbx.png")
        save_image(roughness, output_dir / "rough_rgbx.png")

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
