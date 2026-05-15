#!/usr/bin/env python3
"""Visualize cached SAM region labels used by stage-2 RGB-LiDAR consistency."""

import argparse
import glob
import os
from typing import Optional, Sequence, Set

import numpy as np
from PIL import Image, ImageDraw


def parse_int_set(spec: Optional[str]) -> Optional[Set[int]]:
    if spec is None or spec.strip() == "":
        return None

    values = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(chunk))
    return values


def resolve_scene_dir(data_root: Optional[str], scene_idx: Optional[int]) -> Optional[str]:
    if data_root is None or scene_idx is None:
        return None
    return os.path.join(data_root, f"{scene_idx:03d}")


def resolve_region_dir(args: argparse.Namespace) -> str:
    if args.region_dir is not None:
        return args.region_dir

    scene_dir = resolve_scene_dir(args.data_root, args.scene_idx)
    if scene_dir is None:
        raise ValueError("Pass either --region_dir or both --data_root and --scene_idx")
    return os.path.join(scene_dir, args.region_map_dir)


def resolve_image_dir(args: argparse.Namespace) -> Optional[str]:
    if args.image_dir is not None:
        return args.image_dir

    scene_dir = resolve_scene_dir(args.data_root, args.scene_idx)
    if scene_dir is None:
        return None
    return os.path.join(scene_dir, "images")


def filename_frame_cam(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[-1])
    except ValueError:
        return None, None


def colorize_labels(labels: np.ndarray, seed: int = 17) -> np.ndarray:
    label_ids = labels.astype(np.uint64)
    red = (label_ids * 37 + seed * 13 + 41) % 192 + 48
    green = (label_ids * 67 + seed * 17 + 89) % 192 + 48
    blue = (label_ids * 97 + seed * 19 + 131) % 192 + 48
    colors = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    colors[labels == 0] = 0
    return colors


def compute_boundaries(labels: np.ndarray, width: int) -> np.ndarray:
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]

    for _ in range(max(0, width - 1)):
        expanded = boundaries.copy()
        expanded[1:, :] |= boundaries[:-1, :]
        expanded[:-1, :] |= boundaries[1:, :]
        expanded[:, 1:] |= boundaries[:, :-1]
        expanded[:, :-1] |= boundaries[:, 1:]
        boundaries = expanded
    return boundaries


def find_image(image_dir: Optional[str], stem: str) -> Optional[str]:
    if image_dir is None:
        return None
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(image_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def load_rgb(image_path: Optional[str], size_hw: Sequence[int]) -> np.ndarray:
    h, w = int(size_hw[0]), int(size_hw[1])
    if image_path is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    image = Image.open(image_path).convert("RGB")
    if image.size != (w, h):
        image = image.resize((w, h), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_h = 26
    canvas = Image.new("RGB", (image.width, image.height + title_h), (20, 20, 20))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(240, 240, 240))
    return canvas


def make_panel(rgb: np.ndarray, label_rgb: np.ndarray, overlay: np.ndarray, title: str) -> Image.Image:
    panels = [
        add_title(Image.fromarray(rgb), "RGB"),
        add_title(Image.fromarray(label_rgb), "SAM regions"),
        add_title(Image.fromarray(overlay), title),
    ]
    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def visualize_one(
    label_path: str,
    image_dir: Optional[str],
    output_dir: str,
    alpha: float,
    boundary_width: int,
    seed: int,
    save_separate: bool,
) -> None:
    labels = np.load(label_path)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label map in {label_path}, got shape {labels.shape}")

    stem = os.path.splitext(os.path.basename(label_path))[0]
    image_path = find_image(image_dir, stem)
    rgb = load_rgb(image_path, labels.shape)
    label_rgb = colorize_labels(labels, seed=seed)
    boundaries = compute_boundaries(labels, boundary_width)

    overlay = (rgb.astype(np.float32) * (1.0 - alpha) + label_rgb.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay[boundaries] = np.array([255, 255, 255], dtype=np.uint8)

    panel = make_panel(rgb, label_rgb, overlay, "Overlay + boundaries")
    panel.save(os.path.join(output_dir, f"{stem}_sam_regions.png"))

    if save_separate:
        Image.fromarray(label_rgb).save(os.path.join(output_dir, f"{stem}_labels.png"))
        Image.fromarray(overlay).save(os.path.join(output_dir, f"{stem}_overlay.png"))


def main(args: argparse.Namespace) -> None:
    region_dir = resolve_region_dir(args)
    image_dir = resolve_image_dir(args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = region_dir.rstrip(os.sep) + "_vis"
    os.makedirs(output_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(region_dir, "*.npy")))
    if not label_files:
        raise FileNotFoundError(f"No .npy region maps found in {region_dir}")

    frame_filter = parse_int_set(args.frames)
    selected = []
    for path in label_files:
        frame_id, cam_id = filename_frame_cam(path)
        if args.camera_id is not None and cam_id != args.camera_id:
            continue
        if frame_filter is not None and frame_id not in frame_filter:
            continue
        selected.append(path)

    selected = selected[:: max(1, args.stride)]
    if args.max_images is not None and args.max_images > 0:
        selected = selected[: args.max_images]
    if not selected:
        raise RuntimeError("No region maps matched the requested filters")

    for path in selected:
        visualize_one(
            label_path=path,
            image_dir=image_dir,
            output_dir=output_dir,
            alpha=args.alpha,
            boundary_width=args.boundary_width,
            seed=args.seed,
            save_separate=args.save_separate,
        )

    print(f"Saved {len(selected)} SAM region visualization(s) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize cached SAM region labels")
    parser.add_argument("--region_dir", type=str, default=None, help="Directory containing cached .npy label maps")
    parser.add_argument("--data_root", type=str, default=None, help="Waymo processed training root")
    parser.add_argument("--scene_idx", type=int, default=None, help="Scene index, e.g. 23")
    parser.add_argument("--region_map_dir", type=str, default="region_sam_reflectivity")
    parser.add_argument("--image_dir", type=str, default=None, help="Optional RGB image directory for overlays")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--camera_id", type=int, default=None, help="Only visualize one camera id")
    parser.add_argument("--frames", type=str, default=None, help="Comma/range filter, e.g. 0,5,10-15")
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--boundary_width", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save_separate", action="store_true")
    main(parser.parse_args())
