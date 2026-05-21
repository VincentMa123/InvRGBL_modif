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
from datasets.base.split_wrapper import SplitWrapper
from models.video_utils import render_images, render_novel_views, save_videos
from utils.misc import import_str
from utils.logging import setup_logging


logger = logging.getLogger()


DEFAULT_KEYS = [
    "rgbs",
    "rendered_pbr",
    "rendered_pbr_layered",
    "rendered_pbr_background",
    "rendered_pbr_dynamic",
    "rendered_pbr_dynamic_composite_mask",
    "rendered_albedos",
    "rendered_roughness",
    "rendered_metallic",
    "rendered_reflectivity",
    "rendered_sun_visibility",
    "rendered_dynamic_box_sun_visibility",
    "rendered_dynamic_box_contact_shadow",
    "rendered_dynamic_box_contact_shadow_raw",
    "rendered_dynamic_box_contact_apply_mask",
    "rendered_dynamic_opacity",
    "depths",
]


PRESETS = {
    "afternoon": {
        "sun_direction": "-0.45,-0.35,0.82",
        "sun_intensity": "4.0,3.8,3.4",
        "env_scale": 1.15,
        "env_color": "1.02,1.04,1.08",
        "min_roughness": 0.35,
    },
    "sunset": {
        "sun_direction": "-0.92,-0.20,0.20",
        "sun_intensity": "5.0,2.4,0.9",
        "env_scale": 0.70,
        "env_color": "1.35,0.78,0.45",
        "min_roughness": 0.40,
    },
    "night": {
        "sun_direction": "-0.45,-0.35,0.82",
        "sun_intensity": "0.0,0.0,0.0",
        "env_constant": "0.015,0.025,0.055",
        "min_roughness": 0.45,
        "force_dielectric": True,
        "sky_scale": 0.04,
    },
}


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


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-6)
    return x + torch.log(-torch.expm1(-x))


def apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset == "none":
        return
    preset = PRESETS[args.preset]
    if args.sun_direction is None and "sun_direction" in preset:
        args.sun_direction = preset["sun_direction"]
    if args.sun_intensity is None and "sun_intensity" in preset:
        args.sun_intensity = preset["sun_intensity"]
    if args.env_color is None and "env_color" in preset:
        args.env_color = preset["env_color"]
    if args.env_constant is None and "env_constant" in preset:
        args.env_constant = preset["env_constant"]
    if args.env_scale == 1.0 and "env_scale" in preset:
        args.env_scale = preset["env_scale"]
    if args.min_roughness == 0.35 and "min_roughness" in preset:
        args.min_roughness = preset["min_roughness"]
    if not args.force_dielectric and preset.get("force_dielectric", False):
        args.force_dielectric = True
    if args.sky_scale == 1.0 and "sky_scale" in preset:
        args.sky_scale = preset["sky_scale"]


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


def load_envmap_tensor(path: str, device: torch.device) -> tuple[torch.Tensor, bool]:
    is_parameter = False
    if path.endswith(".npy"):
        env = torch.from_numpy(np.load(path)).float()
    else:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("base", "EnvMap#base", "envmap"):
                if key in loaded:
                    is_parameter = key in ("base", "EnvMap#base")
                    loaded = loaded[key]
                    break
        if not torch.is_tensor(loaded):
            raise ValueError("--envmap_path must contain a tensor or a dict with a base/envmap tensor")
        env = loaded.float()
    if env.ndim != 3 or env.shape[-1] != 3:
        raise ValueError(f"environment map must have shape [H, W, 3], got {tuple(env.shape)}")
    return env.to(device), is_parameter


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

        if hasattr(sky, "shs") and args.sky_scale != 1.0:
            with torch.no_grad():
                sky.shs.data = sky.shs.data * float(args.sky_scale)

    if "EnvMap" in trainer.models:
        env_map = trainer.models["EnvMap"]
        with torch.no_grad():
            radiance = F.softplus(env_map.base.data)
            if args.envmap_path is not None:
                env, is_parameter = load_envmap_tensor(args.envmap_path, env_map.base.device)
                env = resize_envmap(env, env_map.base.shape[0], env_map.base.shape[1])
                radiance = F.softplus(env) if is_parameter else env.clamp_min(0.0)

            if args.env_constant is not None:
                color = parse_vec(args.env_constant, "--env_constant", allow_scalar=True)
                radiance = color.to(env_map.base.device).view(1, 1, 3).expand_as(radiance).clone()

            env_color = parse_vec(args.env_color, "--env_color", allow_scalar=True)
            if env_color is not None:
                radiance = radiance * env_color.to(env_map.base.device).view(1, 1, 3)

            radiance = (radiance * float(args.env_scale)).clamp_min(0.0)
            env_map.base.data.copy_(inverse_softplus(radiance))

        env_map.build_mips()

    trainer.relight_min_roughness = float(args.min_roughness)
    trainer.relight_force_dielectric = bool(args.force_dielectric)
    trainer.spotlights = load_spotlights(args.spotlights)
    trainer.render_cfg["eval_exposure_scale"] = float(args.exposure_scale)
    trainer.render_cfg["eval_disable_affine"] = bool(args.disable_affine)
    trainer.render_cfg["env_diffuse_scale"] = float(args.env_diffuse_scale)
    trainer.render_cfg["env_specular_scale"] = float(args.env_specular_scale)
    trainer.render_cfg["env_diffuse_mode"] = args.env_diffuse_mode
    trainer.render_cfg["env_ambient_floor"] = float(args.env_ambient_floor)
    trainer.render_cfg["dynamic_box_sun_visibility"] = {
        "enabled": bool(args.dynamic_box_sun_visibility),
        "classes": args.dynamic_box_classes,
        "size_scale": args.dynamic_box_size_scale,
        "min_size": args.dynamic_box_min_size,
        "shadow_strength": float(args.dynamic_box_shadow_strength),
        "ray_epsilon": float(args.dynamic_box_ray_epsilon),
        "chunk_size": int(args.dynamic_box_chunk_size),
        "opacity_threshold": float(args.dynamic_box_opacity_threshold),
        "receiver_static_only": bool(args.dynamic_box_receiver_static_only),
        "receiver_dynamic_opacity_threshold": float(args.dynamic_box_receiver_dynamic_opacity_threshold),
        "skip_inside_boxes": bool(args.dynamic_box_skip_inside_boxes),
        "inside_margin": float(args.dynamic_box_inside_margin),
        "contact_shadow_strength": float(args.dynamic_box_contact_shadow_strength),
        "contact_shadow_receiver": args.dynamic_box_contact_shadow_receiver,
        "contact_shadow_height": float(args.dynamic_box_contact_shadow_height),
        "contact_shadow_softness": float(args.dynamic_box_contact_shadow_softness),
        "contact_shadow_dynamic_opacity_threshold": float(args.dynamic_box_contact_shadow_dynamic_opacity_threshold),
        "contact_shadow_apply_dynamic_opacity_threshold": float(args.dynamic_box_contact_shadow_apply_dynamic_opacity_threshold),
        "contact_shadow_apply_depth_tolerance": float(args.dynamic_box_contact_shadow_apply_depth_tolerance),
        "layer_separated_pbr": bool(args.dynamic_box_layer_separated_pbr),
        "layer_dynamic_depth_margin": float(args.dynamic_box_layer_dynamic_depth_margin),
    }


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


def limit_split_dataset(split_dataset, max_frames: Optional[int], num_cams: int):
    if split_dataset is None:
        return None, 0

    total_timestamps = (len(split_dataset) + num_cams - 1) // num_cams
    if max_frames is None or max_frames <= 0:
        return split_dataset, total_timestamps

    max_images = int(max_frames) * int(num_cams)
    limited_indices = split_dataset.split_indices[:max_images]
    limited = SplitWrapper(
        datasource=split_dataset.datasource,
        split_indices=limited_indices,
        split=split_dataset.split,
    )
    rendered_timestamps = (len(limited_indices) + num_cams - 1) // num_cams
    return limited, rendered_timestamps


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

    num_cams = dataset.pixel_source.num_cams
    split_dataset, num_timestamps = limit_split_dataset(
        split_dataset, args.max_frames, num_cams
    )
    if len(split_dataset) == 0:
        logger.info("Skipping %s split because frame limit produced no images", split_name)
        return

    logger.info("Rendering %s relighting split", split_name)
    if args.max_frames is not None and args.max_frames > 0:
        logger.info(
            "Limited %s relighting split to %d frame(s), %d image(s)",
            split_name,
            num_timestamps,
            len(split_dataset),
        )
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
        num_timestamps=num_timestamps,
        keys=keys,
        num_cams=num_cams,
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
    logger.info("Relighting preset: %s", args.preset)

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
    parser.add_argument("--preset", choices=["none", "afternoon", "sunset", "night"], default="none")
    parser.add_argument("--output_dir", type=str, default=None, help="explicit output directory")
    parser.add_argument("--split", choices=["full", "test", "both", "none"], default="full")
    parser.add_argument("--max_frames", type=int, default=None, help="render only the first N timesteps from each split")
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

    parser.add_argument("--sky_scale", type=float, default=1.0, help="scale visible sky background SH")
    parser.add_argument("--exposure_scale", type=float, default=1.0, help="multiply final relit RGB")
    parser.add_argument("--disable_affine", action="store_true", help="skip learned affine color correction")
    parser.add_argument("--env_diffuse_scale", type=float, default=1.0, help="scale EnvMap diffuse IBL")
    parser.add_argument("--env_specular_scale", type=float, default=1.0, help="scale EnvMap specular IBL")
    parser.add_argument("--env_diffuse_mode", choices=["learned", "neutral"], default="learned", help="diffuse IBL mode")
    parser.add_argument("--env_ambient_floor", type=float, default=0.0, help="minimum ambient added to diffuse")
    parser.add_argument("--dynamic_box_sun_visibility", action=argparse.BooleanOptionalAction, default=False, help="multiply sun visibility by per-pixel dynamic OBB ray tests")
    parser.add_argument("--dynamic_box_classes", type=str, default="RigidNodes", help="comma-separated dynamic box model classes")
    parser.add_argument("--dynamic_box_size_scale", type=str, default="1.05,1.05,1.05", help="scalar or x,y,z multiplier for dynamic boxes")
    parser.add_argument("--dynamic_box_min_size", type=str, default="0.0,0.0,0.0", help="scalar or x,y,z minimum dynamic box size")
    parser.add_argument("--dynamic_box_shadow_strength", type=float, default=1.0, help="0 leaves sun visibility unchanged, 1 fully blocks direct sun")
    parser.add_argument("--dynamic_box_ray_epsilon", type=float, default=0.05, help="minimum positive ray distance before a box can occlude")
    parser.add_argument("--dynamic_box_chunk_size", type=int, default=65536, help="number of valid pixels tested per ray-box chunk")
    parser.add_argument("--dynamic_box_opacity_threshold", type=float, default=0.01, help="ignore pixels below this raster opacity")
    parser.add_argument("--dynamic_box_receiver_static_only", action=argparse.BooleanOptionalAction, default=True, help="apply box shadows only to pixels that are not mostly dynamic objects")
    parser.add_argument("--dynamic_box_receiver_dynamic_opacity_threshold", type=float, default=0.2, help="dynamic opacity cutoff for static-only shadow receivers")
    parser.add_argument("--dynamic_box_skip_inside_boxes", action=argparse.BooleanOptionalAction, default=True, help="avoid self-shadowing pixels inside an object box")
    parser.add_argument("--dynamic_box_inside_margin", type=float, default=0.02, help="meters subtracted from boxes for inside/self-shadow detection")
    parser.add_argument("--dynamic_box_contact_shadow_strength", type=float, default=0.0, help="soft under-object contact shadow strength applied to sun and environment lighting")
    parser.add_argument("--dynamic_box_contact_shadow_receiver", choices=["full", "background"], default="full", help="surface used to evaluate contact shadows; background uses a background-only receiver depth pass")
    parser.add_argument("--dynamic_box_contact_shadow_height", type=float, default=0.75, help="vertical distance in meters over which dynamic OBB contact shadow fades from the box bottom")
    parser.add_argument("--dynamic_box_contact_shadow_softness", type=float, default=0.35, help="horizontal softness in meters outside the dynamic OBB footprint")
    parser.add_argument("--dynamic_box_contact_shadow_dynamic_opacity_threshold", type=float, default=0.8, help="ignore contact shadow on pixels with dynamic opacity above this threshold")
    parser.add_argument("--dynamic_box_contact_shadow_apply_dynamic_opacity_threshold", type=float, default=0.8, help="when using the background contact receiver, do not apply the contact term to pixels above this dynamic opacity")
    parser.add_argument("--dynamic_box_contact_shadow_apply_depth_tolerance", type=float, default=0.5, help="when using the background contact receiver, apply contact only where full and background depths agree within this many meters; negative disables this gate")
    parser.add_argument("--dynamic_box_layer_separated_pbr", action=argparse.BooleanOptionalAction, default=False, help="shade background and dynamic layers separately, apply dynamic box contact only to background, then composite dynamic over background")
    parser.add_argument("--dynamic_box_layer_dynamic_depth_margin", type=float, default=0.25, help="meters a dynamic layer must be in front of the background layer to cover it in layer-separated PBR")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="OmegaConf overrides")
    parsed_args = parser.parse_args()
    apply_preset_defaults(parsed_args)
    main(parsed_args)
