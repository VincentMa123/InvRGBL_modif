#!/usr/bin/env python
"""Visualize shadow maps from a trained checkpoint.

Renders the scene from both the main camera and the sun camera,
saving side-by-side comparisons of RGB, sun visibility, and shadow depth.
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images
from utils.misc import import_str
from utils.logging import setup_logging

logger = logging.getLogger()


def load_checkpoint(args: argparse.Namespace, device: torch.device):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    cfg.trainer.render.eval_use_pbr_rgb = False

    # Enable shadow map rendering
    cfg.trainer.render.use_shadow_map = True
    cfg.trainer.render.shadow_map_size = args.shadow_map_size

    # --- FIX 1: Peek checkpoint to get the original num_full_images ---
    ckpt = torch.load(args.resume_from, map_location="cpu")
    saved_num_full_images = ckpt.get("num_full_images", None)
    # Fallback: read embedding shape directly from AffineTransform weights
    if saved_num_full_images is None and "Affine" in ckpt.get("models", {}):
        saved_num_full_images = ckpt["models"]["Affine"]["embedding.weight"].shape[0]

    # --- FIX 2: Limit dataset frames to only what we need for viz ---
    # Override end_timestep so we only load frames up to the max requested frame + 1
    frame_indices = [int(x.strip()) for x in args.frames.split(",")] if args.frames else [0]
    max_frame_needed = max(frame_indices)
    # Ensure we load at least enough frames for the checkpoint's Affine embedding,
    # but cap LiDAR memory by limiting end_timestep.
    # We need start..end to cover max_frame_needed.
    cfg.data.start_timestep = 0
    cfg.data.end_timestep = max(max_frame_needed + 1, 2)

    dataset = DrivingDataset(data_cfg=cfg.data)

    # Use checkpoint's num_full_images if available; otherwise fall back to dataset.
    # This prevents AffineTransform embedding size mismatch.
    num_full_images = saved_num_full_images if saved_num_full_images is not None else len(dataset.full_image_set)

    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=num_full_images,
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    trainer.set_eval()
    if hasattr(trainer, "render_each_class"):
        trainer.render_each_class = False
    return cfg, dataset, trainer


def save_depth_as_image(depth: torch.Tensor, path: str, cmap: str = "turbo"):
    """Save a depth map as a colored PNG."""
    depth_np = depth.detach().cpu().numpy()
    finite = np.isfinite(depth_np)
    # Prefer non-zero finite depths for scaling (zero = background/no-hit)
    valid = finite & (depth_np > 0)
    if not valid.any():
        if not finite.any():
            img = np.zeros_like(depth_np, dtype=np.uint8)
            Image.fromarray(img).save(path)
            return
        scale_mask = finite
    else:
        scale_mask = valid

    # Normalize to [0, 1] using 1st-99th percentile of valid depths
    lo, hi = np.percentile(depth_np[scale_mask], [1.0, 99.0])
    if hi <= lo:
        lo, hi = depth_np[scale_mask].min(), depth_np[scale_mask].max()
    logger.info("Depth scale lo=%.3f hi=%.3f for %s", float(lo), float(hi), os.path.basename(path))
    norm = np.clip((depth_np - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    if cmap == "gray":
        gray = (norm * 255).astype(np.uint8)
        Image.fromarray(gray, mode="L").save(path)
        return

    # Turbo colormap (matplotlib-like)
    from matplotlib import colormaps
    turbo = colormaps["turbo"]
    rgb = (turbo(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def save_visibility_as_image(vis: torch.Tensor, path: str):
    """Save a binary visibility map (white = sun, black = shadow)."""
    vis_np = vis.detach().cpu().numpy()
    if vis_np.ndim == 3:
        vis_np = vis_np[..., 0]
    gray = (vis_np * 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)


def save_rgb_as_image(rgb: torch.Tensor, path: str):
    """Save an RGB tensor [H, W, 3] as PNG."""
    rgb_np = rgb.detach().cpu().clamp(0, 1).numpy()
    img = (rgb_np * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def make_composite(rgb, vis, shadow_depth, output_path):
    """Create a 2x2 grid: RGB | SunVis | ShadowDepth | ShadowOverlay."""
    from PIL import Image

    h, w = rgb.shape[:2]
    grid = Image.new("RGB", (w * 2, h * 2))

    # Top-left: RGB
    rgb_np = rgb.detach().cpu().clamp(0, 1).numpy()
    grid.paste(Image.fromarray((rgb_np * 255).astype(np.uint8)), (0, 0))

    # Top-right: Sun visibility (white = sunlit, black = shadow)
    vis_np = vis.detach().cpu().numpy()
    if vis_np.ndim == 3:
        vis_np = vis_np[..., 0]
    vis_img = Image.fromarray((vis_np * 255).astype(np.uint8), mode="L").convert("RGB")
    grid.paste(vis_img, (w, 0))

    # Bottom-left: Shadow depth map
    depth_np = shadow_depth.detach().cpu().numpy()
    finite = np.isfinite(depth_np)
    valid = finite & (depth_np > 0)
    if valid.any():
        lo, hi = np.percentile(depth_np[valid], [1.0, 99.0])
        if hi <= lo:
            lo, hi = depth_np[valid].min(), depth_np[valid].max()
        logger.info("Composite depth scale lo=%.3f hi=%.3f for %s", float(lo), float(hi), os.path.basename(output_path))
        norm = np.clip((depth_np - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    elif finite.any():
        lo, hi = np.percentile(depth_np[finite], [1.0, 99.0])
        if hi <= lo:
            lo, hi = depth_np[finite].min(), depth_np[finite].max()
        logger.info("Composite (fallback) depth scale lo=%.3f hi=%.3f for %s", float(lo), float(hi), os.path.basename(output_path))
        norm = np.clip((depth_np - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    else:
        norm = np.zeros_like(depth_np)
    try:
        from matplotlib import colormaps
        turbo = colormaps["turbo"]
        depth_rgb = (turbo(norm)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        depth_rgb = (norm * 255).astype(np.uint8)
        depth_rgb = np.stack([depth_rgb] * 3, axis=-1)
    grid.paste(Image.fromarray(depth_rgb), (0, h))

    # Bottom-right: RGB with shadow overlay (red = shadow)
    overlay = rgb_np.copy()
    shadow_mask = vis_np < 0.5
    overlay[shadow_mask] = overlay[shadow_mask] * 0.3 + np.array([0.7, 0.0, 0.0]) * 0.7
    grid.paste(Image.fromarray((overlay * 255).astype(np.uint8)), (w, h))

    grid.save(output_path)


def main(args: argparse.Namespace):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, dataset, trainer = load_checkpoint(args, device)

    split_dataset = dataset.full_image_set
    if args.split == "test" and dataset.test_image_set is not None:
        split_dataset = dataset.test_image_set

    num_cams = dataset.pixel_source.num_cams
    total_frames = len(split_dataset) // num_cams

    frame_indices = None
    if args.frames is not None:
        frame_indices = [int(x.strip()) for x in args.frames.split(",")]
    if frame_indices is None:
        frame_indices = list(range(min(args.max_frames, total_frames)))

    logger.info("Visualizing shadow maps for %d frame(s)", len(frame_indices))

    for frame_idx in frame_indices:
        for cam_id in range(num_cams):
            img_idx = frame_idx * num_cams + cam_id
            if img_idx >= len(split_dataset):
                continue

            image_infos, cam_infos = split_dataset[img_idx]
            image_infos = {k: v.to(device) if torch.is_tensor(v) else v for k, v in image_infos.items()}
            cam_infos = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cam_infos.items()}

            with torch.no_grad():
                outputs = trainer(image_infos, cam_infos)

            frame_key = trainer.cur_frame.item()
            shadow_map = None
            if hasattr(trainer, '_sun_shadow_map_list') and frame_key in trainer._sun_shadow_map_list:
                shadow_map = trainer._sun_shadow_map_list[frame_key]
                # Diagnostic: print depth stats
                depth_min = shadow_map.min().item()
                depth_max = shadow_map.max().item()
                depth_mean = shadow_map.mean().item()
                logger.info("Frame %d shadow depth — min: %.3f, max: %.3f, mean: %.3f, range: %.3f",
                            frame_idx, depth_min, depth_max, depth_mean, depth_max - depth_min)
            else:
                logger.warning("No shadow map cached for frame %d (key %d)", frame_idx, frame_key)
                continue

            rgb = outputs["rgb"].clamp(0, 1)
            sun_vis = outputs.get("rendered_sun_visibility", torch.ones_like(rgb[:, :, :1]))

            base_name = f"frame{frame_idx:03d}_cam{cam_id}"

            # Save individual images
            save_rgb_as_image(rgb, os.path.join(output_dir, f"{base_name}_rgb.png"))
            save_visibility_as_image(sun_vis, os.path.join(output_dir, f"{base_name}_sunvis.png"))
            save_depth_as_image(shadow_map, os.path.join(output_dir, f"{base_name}_shadow_depth.png"), cmap="turbo")

            # Save composite
            make_composite(rgb, sun_vis, shadow_map, os.path.join(output_dir, f"{base_name}_composite.png"))

            logger.info("Saved %s shadow map visualization", base_name)

    logger.info("All visualizations saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize shadow maps from a trained checkpoint")
    parser.add_argument("--resume_from", type=str, required=True, help="checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--split", choices=["full", "test"], default="full")
    parser.add_argument("--frames", type=str, default=None, help="comma-separated frame indices, e.g. '0,10,20'")
    parser.add_argument("--max_frames", type=int, default=5, help="if --frames not given, visualize first N")
    parser.add_argument("--shadow_map_size", type=int, default=2048)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="OmegaConf overrides")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.resume_from), "shadow_map_vis")

    main(args)
