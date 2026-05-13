import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images, render_novel_views, save_videos
from utils.misc import import_str
from utils.logging import setup_logging


logger = logging.getLogger()


DEFAULT_KEYS = [
    "rgbs",
    "rendered_pbr",
    "rendered_albedos",
    "rendered_roughness",
    "rendered_metallic",
    "rendered_reflectivity",
    "rendered_sun_visibility",
    "depths",
]


def parse_vec(text: Optional[str], name: str, allow_scalar: bool = False) -> Optional[torch.Tensor]:
    if text is None:
        return None
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if allow_scalar and len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"{name} must be three comma-separated values, got: {text}")
    return torch.tensor(values, dtype=torch.float32)


def parse_keys(text: Optional[str]) -> List[str]:
    if text is None:
        return DEFAULT_KEYS
    keys = [k.strip() for k in text.split(",") if k.strip()]
    return keys if keys else DEFAULT_KEYS


def load_spotlights(path: Optional[str]) -> Optional[List[Dict]]:
    if path is None:
        return None
    with open(path, "r") as f:
        spotlights = json.load(f)
    if not isinstance(spotlights, list):
        raise ValueError("--spotlights must point to a JSON list of spotlight dictionaries")
    for i, light in enumerate(spotlights):
        if "position" not in light or "intensity" not in light:
            raise ValueError(f"spotlight {i} needs at least 'position' and 'intensity'")
    return spotlights


def load_envmap_tensor(path: str, device: torch.device) -> torch.Tensor:
    if path.endswith(".npy"):
        env = torch.from_numpy(np.load(path)).float()
    else:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("base", "envmap", "EnvMap#base"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        if not torch.is_tensor(loaded):
            raise ValueError("--envmap_path must contain a tensor or a dict with a base/envmap tensor")
        env = loaded.float()
    if env.ndim != 3 or env.shape[-1] != 3:
        raise ValueError(f"environment map must have shape [H, W, 3], got {tuple(env.shape)}")
    return env.to(device)


def resize_envmap(env: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if env.shape[0] == height and env.shape[1] == width:
        return env
    chw = env.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(chw, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0)


def apply_relighting(trainer, args: argparse.Namespace, device: torch.device) -> None:
    if "Sky" in trainer.models:
        sky = trainer.models["Sky"]
        sun_direction = parse_vec(args.sun_direction, "--sun_direction")
        if sun_direction is not None:
            sun_direction = sun_direction.to(device)
            sky.anno_sun_direction = sun_direction / sun_direction.norm()
            if hasattr(sky, "sun_direction"):
                sky.sun_direction.data.copy_(sky.anno_sun_direction.to(sky.sun_direction.device))

        sun_intensity = parse_vec(args.sun_intensity, "--sun_intensity", allow_scalar=True)
        if sun_intensity is not None and hasattr(sky, "sun_intensity"):
            sky.sun_intensity.data.copy_(sun_intensity.to(sky.sun_intensity.device))

    if "EnvMap" in trainer.models:
        env_map = trainer.models["EnvMap"]
        with torch.no_grad():
            if args.envmap_path is not None:
                env = load_envmap_tensor(args.envmap_path, env_map.base.device)
                env = resize_envmap(env, env_map.base.shape[0], env_map.base.shape[1])
                env_map.base.data.copy_(env)

            if args.env_constant is not None:
                color = parse_vec(args.env_constant, "--env_constant", allow_scalar=True)
                env_map.base.data.fill_(1.0)
                env_map.base.data.mul_(color.to(env_map.base.device))

            env_color = parse_vec(args.env_color, "--env_color", allow_scalar=True)
            if env_color is not None:
                env_map.base.data.mul_(env_color.to(env_map.base.device))

            env_map.base.data.mul_(float(args.env_scale))
            env_map.base.data.clamp_(min=0.0)

        env_map.build_mips()

    trainer.relight_min_roughness = float(args.min_roughness)
    trainer.relight_force_dielectric = bool(args.force_dielectric)
    trainer.spotlights = load_spotlights(args.spotlights)


def build_dataset_and_trainer(args: argparse.Namespace):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    cfg.logging.save_seperate_video = not args.save_catted_videos
    cfg.trainer.render.eval_use_pbr_rgb = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    trainer.set_eval()
    apply_relighting(trainer, args, device)
    return cfg, dataset, trainer


def render_split(
    trainer,
    dataset,
    split_dataset,
    split_name: str,
    output_dir: str,
    keys: List[str],
    cfg,
    args: argparse.Namespace,
) -> None:
    if split_dataset is None:
        logger.info("Skipping %s split because it is not available", split_name)
        return

    logger.info("Rendering %s relighting split", split_name)
    results = render_images(
        trainer=trainer,
        dataset=split_dataset,
        compute_metrics=False,
        compute_error_map=False,
    )
    save_path = os.path.join(output_dir, f"{split_name}_{trainer.step}.mp4")
    save_videos(
        results,
        save_path,
        layout=dataset.layout,
        num_timestamps=(
            dataset.num_test_timesteps if split_name == "test" else dataset.num_img_timesteps
        ),
        keys=keys,
        num_cams=dataset.pixel_source.num_cams,
        save_seperate_video=cfg.logging.save_seperate_video,
        save_images=args.save_images,
        fps=cfg.render.fps,
        verbose=True,
    )


def render_novel_if_requested(trainer, dataset, cfg, output_dir: str, args: argparse.Namespace) -> None:
    if not args.render_novel:
        return
    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is None:
        logger.info("Skipping novel views because cfg.render.render_novel is not configured")
        return

    if args.split == "none" and hasattr(trainer, "rebuild_all_visibility"):
        sun_dir = None
        if "Sky" in trainer.models:
            sun_dir = trainer.models["Sky"].get_sun_direction()
        trainer.rebuild_all_visibility(dataset.num_img_timesteps, sun_direction=sun_dir)

    render_traj = dataset.get_novel_render_traj(
        traj_types=render_novel_cfg.traj_types,
        target_frames=render_novel_cfg.get("frames", dataset.frame_num),
    )
    novel_dir = os.path.join(output_dir, f"novel_{trainer.step}")
    os.makedirs(novel_dir, exist_ok=True)
    for traj_type, traj in render_traj.items():
        render_data = dataset.prepare_novel_view_render_data(traj)
        save_path = os.path.join(novel_dir, f"{traj_type}.mp4")
        render_novel_views(
            trainer,
            render_data,
            save_path,
            fps=render_novel_cfg.get("fps", cfg.render.fps),
        )


def main(args: argparse.Namespace) -> None:
    log_dir = os.path.dirname(args.resume_from)
    output_dir = args.output_dir or os.path.join(log_dir, "videos_relight", args.name)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)

    cfg, dataset, trainer = build_dataset_and_trainer(args)
    keys = parse_keys(args.keys)

    logger.info("Saving relighting results to %s", output_dir)
    logger.info("Video keys: %s", ", ".join(keys))

    if args.split in ("full", "both"):
        render_split(
            trainer, dataset, dataset.full_image_set, "full", output_dir, keys, cfg, args
        )
    if args.split in ("test", "both"):
        render_split(
            trainer, dataset, dataset.test_image_set, "test", output_dir, keys, cfg, args
        )

    render_novel_if_requested(trainer, dataset, cfg, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render relit InvRGBL videos from a trained checkpoint")
    parser.add_argument("--resume_from", type=str, required=True, help="checkpoint path")
    parser.add_argument("--name", type=str, default="relight", help="output subfolder name")
    parser.add_argument("--output_dir", type=str, default=None, help="explicit output directory")
    parser.add_argument("--split", choices=["full", "test", "both", "none"], default="full")
    parser.add_argument("--render_novel", action="store_true", help="render configured novel trajectories")
    parser.add_argument("--keys", type=str, default=None, help="comma-separated video keys to save")
    parser.add_argument("--save_catted_videos", action="store_true", help="save one concatenated video per split")
    parser.add_argument("--save_images", action="store_true", help="also save per-frame PNGs for separate videos")

    parser.add_argument("--sun_direction", type=str, default=None, help="x,y,z world-space sun direction")
    parser.add_argument("--sun_intensity", type=str, default=None, help="scalar or r,g,b sun intensity")
    parser.add_argument("--envmap_path", type=str, default=None, help=".pt or .npy EnvMap tensor [H,W,3]")
    parser.add_argument("--env_scale", type=float, default=1.0, help="multiply learned EnvMap radiance")
    parser.add_argument("--env_color", type=str, default=None, help="scalar or r,g,b EnvMap tint multiplier")
    parser.add_argument("--env_constant", type=str, default=None, help="replace EnvMap with scalar or r,g,b constant")
    parser.add_argument("--min_roughness", type=float, default=0.35, help="eval-time roughness floor")
    parser.add_argument("--force_dielectric", action="store_true", help="set metallic to zero during relighting")
    parser.add_argument("--spotlights", type=str, default=None, help="JSON list of spotlight dictionaries")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="OmegaConf overrides")
    main(parser.parse_args())
