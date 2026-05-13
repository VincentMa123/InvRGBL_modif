"""
Relighting / nighttime inference script for InvRGBL.

Usage:
    # Relight with a custom sun direction
    python tools/relight.py \
        --config configs/invrgbl.yaml \
        --checkpoint work_dirs/invrgbl/scene/ckpts/final_model.pth \
        --output_dir work_dirs/invrgbl/scene/relight_afternoon \
        --sun_direction -0.5 -0.8 0.3 \
        --sun_intensity 2.5

    # Nighttime (sun below horizon, dim sky)
    python tools/relight.py \
        --config configs/invrgbl.yaml \
        --checkpoint work_dirs/invrgbl/scene/ckpts/final_model.pth \
        --output_dir work_dirs/invrgbl/scene/night \
        --mode night
"""

import argparse
import numpy as np
import os
import sys
import types
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.video_utils import render_images, save_videos


def initialize_envmap_from_sky(trainer, sky_scale=1.2, ground_scale=0.3, force=False):
    """Initialize EnvMap with the simple sky-ground gradient used by the older relight script."""
    if "EnvMap" not in trainer.models or "Sky" not in trainer.models:
        return False

    env_map = trainer.models["EnvMap"]
    sky = trainer.models["Sky"]
    init_val = 0.35  # init value in EnvironmentMap.__init__
    is_untrained = (
        env_map.base.data.mean().item() < init_val + 0.05
        and env_map.base.data.std().item() < 0.1
    )
    if not force and not is_untrained:
        env_map.build_mips()
        print("Using learned environment map from checkpoint.")
        return False

    learned_sky = sky.sky_intensity.data.clone().detach()
    H, W = env_map.h, env_map.w
    for i in range(H):
        t = i / max(H - 1, 1)
        sky_color = learned_sky * float(sky_scale)
        ground_color = learned_sky * float(ground_scale)
        env_map.base.data[i, :, :] = sky_color * (1.0 - t) + ground_color * t
    env_map.build_mips()

    mode = "Forced" if force else "Initialized"
    print(
        f"{mode} simple sky-ground environment map from sky intensity: "
        f"{learned_sky.cpu().tolist()} (sky_scale={sky_scale}, ground_scale={ground_scale})"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="InvRGBL relighting / nighttime inference")
    parser.add_argument("--config", default=None, help="Path to experiment config (e.g. configs/invrgbl.yaml)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pth")
    parser.add_argument("--output_dir", required=True, help="Where to save output videos")
    parser.add_argument(
        "--sun_direction",
        nargs=3,
        type=float,
        default=None,
        help="New sun direction vector, e.g. -0.5 -0.8 0.3",
    )
    parser.add_argument(
        "--sun_intensity",
        type=float,
        default=None,
        help="Multiplicative scale for sun intensity",
    )
    parser.add_argument(
        "--sky_intensity",
        nargs=3,
        type=float,
        default=None,
        help="Override sky dome color, e.g. 0.05 0.05 0.08 for night",
    )
    parser.add_argument(
        "--mode",
        choices=["relight", "night", "afternoon", "sunset"],
        default="relight",
        help="Preset: 'relight' uses custom sun, 'night' disables sun and dims sky, "
             "'afternoon' low warm sun, 'sunset' near-horizon orange sun",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name in config (e.g. waymo/1cams)",
    )
    parser.add_argument(
        "--render_set",
        choices=["test", "train", "full"],
        default="test",
        help="Which image set to render",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for output video",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Only render the first N images (useful for quick preview)",
    )
    parser.add_argument(
        "--visibility_samples",
        type=int,
        default=128,
        help="Number of Monte-Carlo incident rays per Gaussian at inference (paper uses 128)",
    )
    parser.add_argument(
        "--force_dielectric",
        action="store_true",
        help="Force metallic to zero during relighting PBR. Recommended for outdoor Waymo scenes with unreliable learned metallic.",
    )
    parser.add_argument(
        "--simple_envmap",
        action="store_true",
        help="Force the older Kimi-style EnvMap: a simple sky-ground gradient from learned sky_intensity.",
    )
    parser.add_argument(
        "--disable_rigid_nodes",
        action="store_true",
        help="Debug/render without RigidNodes. Useful when a moving-object Gaussian blob appears in relighting output.",
    )
    parser.add_argument(
        "--disable_affine",
        action="store_true",
        help="Set the learned Affine transform to identity. By default this is only done for night mode.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line "
             "(e.g. data.scene_idx=23 data.data_root=/path/to/data)",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # ------------------ Load config ------------------
    # Prefer the saved config next to the checkpoint (like eval.py) because it
    # contains the exact merged settings used during training.
    ckpt_dir = os.path.dirname(args.checkpoint)
    saved_cfg_path = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(saved_cfg_path):
        cfg = OmegaConf.load(saved_cfg_path)
    elif args.config is not None:
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError(
            f"No config found at {saved_cfg_path} and --config was not provided."
        )

    if args.dataset is not None:
        cfg.dataset = args.dataset

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    # Parse command-line overrides (same syntax as train.py)
    # These are merged LAST so they override dataset defaults.
    # Remember original timestep range from the saved config so we can warn
    # if the user tries to override it (that breaks checkpoint loading).
    orig_start = cfg.data.get("start_timestep", 0)
    orig_end   = cfg.data.get("end_timestep", -1)

    if args.opts is not None:
        opts_cfg = OmegaConf.from_cli(args.opts)
        cfg = OmegaConf.merge(cfg, opts_cfg)

    # Restore timestep range to match the checkpoint; use --max_frames to limit rendering.
    if cfg.data.get("start_timestep") != orig_start or cfg.data.get("end_timestep") != orig_end:
        print(
            f"WARNING: start/end_timestep was overridden in CLI opts. "
            f"Restoring to training values (start={orig_start}, end={orig_end}) "
            f"so checkpoint loads correctly. Use --max_frames to preview fewer images."
        )
        cfg.data.start_timestep = orig_start
        cfg.data.end_timestep = orig_end

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------ Build dataset & trainer ------------------
    dataset = DrivingDataset(data_cfg=cfg.data)

    trainer_type = import_str(cfg.trainer.type)
    trainer = trainer_type(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device="cuda",
    )
    trainer.eval()

    # ------------------ Load checkpoint ------------------
    print(f"Loading checkpoint from {args.checkpoint}")
    trainer.resume_from_checkpoint(args.checkpoint, load_only_model=True)
    trainer.relight_force_dielectric = args.force_dielectric
    if trainer.relight_force_dielectric:
        print("Forcing metallic=0 during relighting PBR.")
    if args.disable_rigid_nodes and "RigidNodes" in trainer.gaussian_classes:
        trainer.gaussian_classes.pop("RigidNodes")
        print("Disabled RigidNodes for this relighting render.")
    elif "RigidNodes" in trainer.models:
        rigid_model = trainer.models["RigidNodes"]
        if hasattr(rigid_model, "num_instances"):
            print(f"RigidNodes has {rigid_model.num_instances} instances (ids 0..{rigid_model.num_instances - 1}).")
    print(f"Resumed at step {trainer.step}")

    # Clamp roughness higher during relighting to avoid glossy artifacts
    # (specular blobs, inconsistent zebra crossings, splotchy textures)
    # Per-mode roughness clamp:
    # - afternoon/sunset: use a soft clamp to prevent extreme glossiness on
    #   surfaces that learned wrong roughness.
    # - night: street lamps are point lights; need higher clamp to kill sharp highlights.
    # - custom relight: learned env map still has captured sun spot -> aggressive clamp.
    if args.mode == "afternoon":
        trainer.relight_min_roughness = 0.20
    elif args.mode == "sunset":
        trainer.relight_min_roughness = 0.20
    elif args.mode == "night":
        trainer.relight_min_roughness = 0.40
    else:
        trainer.relight_min_roughness = 0.30
    print(f"Set relight_min_roughness = {trainer.relight_min_roughness} for '{args.mode}' mode")
    
    # Initialize EnvMap from learned sky intensity for backward compatibility.
    # --simple_envmap forces the older relight behavior that overwrites even a
    # trained EnvMap with a smooth sky-ground gradient.
    initialize_envmap_from_sky(trainer, force=args.simple_envmap)

    # ------------------ Configure lighting ------------------
    sky_model = trainer.models.get("Sky")
    if sky_model is None:
        print("Warning: No Sky model found in trainer; skipping lighting overrides.")

    new_sun_dir = None

    if args.mode == "night" and sky_model is not None:
        # Night preset: sun below horizon, dim sky dome and dim environment light.
        # Coordinate system: Z is UP.  Default sun dir has positive Z, so to put
        # the sun below the horizon we need negative Z.
        new_sun_dir = torch.tensor([0.0, 0.0, -1.0], device="cuda", dtype=torch.float32)
        sky_model.sun_intensity.data[:] = 0.0
        # Kill the sky dome SH completely so rgb_sky_blend is black.
        sky_model.shs.data[:] = 0.0
        # Environment light for PBR — moonlight level (~1-2% of daylight).
        sky_model.sky_intensity.data[:] = torch.tensor(
            [0.01, 0.01, 0.015], device=sky_model.sky_intensity.device, dtype=torch.float32
        )
        # Darken env map for night
        if "EnvMap" in trainer.models:
            env_map = trainer.models["EnvMap"]
            env_map.base.data *= 0.02
            env_map.build_mips()
            print("Darkened environment map for night mode.")
    elif args.mode == "afternoon" and sky_model is not None:
        # Afternoon: sun from the opposite side, same elevation, slightly warmer.
        new_sun_dir = torch.tensor([0.72, 0.65, 0.23], device="cuda", dtype=torch.float32)
        learned = sky_model.sun_intensity.data.clone()
        sky_model.sun_intensity.data[:] = learned * torch.tensor(
            [1.0, 0.95, 0.85], device=learned.device, dtype=torch.float32
        )
        learned_sky = sky_model.sky_intensity.data.clone()
        sky_model.sky_intensity.data[:] = learned_sky * torch.tensor(
            [0.95, 0.95, 0.90], device=learned_sky.device, dtype=torch.float32
        )
    elif args.mode == "sunset" and sky_model is not None:
        # Sunset: sun near horizon, dimmer, orange, warm sky.
        new_sun_dir = torch.tensor([0.4, 0.85, 0.2], device="cuda", dtype=torch.float32)
        learned = sky_model.sun_intensity.data.clone()
        # Dimmer sun for golden hour (0.7x overall) but warm tint
        sky_model.sun_intensity.data[:] = learned * torch.tensor(
            [0.85, 0.55, 0.30], device=learned.device, dtype=torch.float32
        )
        learned_sky = sky_model.sky_intensity.data.clone()
        sky_model.sky_intensity.data[:] = learned_sky * torch.tensor(
            [0.80, 0.65, 0.50], device=learned_sky.device, dtype=torch.float32
        )
    elif args.sun_direction is not None and sky_model is not None:
        new_sun_dir = torch.tensor(
            args.sun_direction, device="cuda", dtype=torch.float32
        )

    if args.sun_intensity is not None and sky_model is not None:
        sky_model.sun_intensity.data[:] *= args.sun_intensity

    if args.sky_intensity is not None and sky_model is not None:
        sky_model.sky_intensity.data[:] = torch.tensor(
            args.sky_intensity, device=sky_model.sky_intensity.device, dtype=torch.float32
        )

    # --- Inject new sun direction into image_space_pbr ---
    if new_sun_dir is not None and sky_model is not None:
        sky_model.anno_sun_direction = new_sun_dir
        print(f"Set image-space PBR sun direction: {new_sun_dir.cpu().tolist()}")

    # ------------------ Configure spotlights for night mode ------------------
    if args.mode == "night":
        spotlights = []
        
        # Camera IDs in this codebase are integers (e.g. 0 = front camera).
        # We pull the trajectory from every available camera and place lights along it.
        cam_ids = (
            dataset.pixel_source.camera_list
            if hasattr(dataset.pixel_source, "camera_list")
            else []
        )
        
        for cam_id in cam_ids:
            if cam_id not in dataset.pixel_source.camera_data:
                continue
            cam = dataset.pixel_source.camera_data[cam_id]
            cam_positions = cam.cam_to_worlds[:, :3, 3].cpu().numpy()
            cam_rights    = cam.cam_to_worlds[:, :3, 0].cpu().numpy()
            
            # --- Street lamps along the road ---
            # Spaced farther apart and with softer falloff so pools blend
            # instead of creating interference stripes.
            for i in range(0, len(cam_positions), 20):
                for side in [-1, 1]:
                    pos = cam_positions[i] + side * cam_rights[i] * 3.0 + np.array([0.0, 0.0, 15.0])
                    spotlights.append({
                        "position": pos.tolist(),
                        "intensity": 8.0,
                        "color": [1.0, 0.9, 0.6],
                        "direction": [0.0, 0.0, -1.0],
                        "cutoff_angle": 2.0 * np.pi / 3.0,  # 120° for very soft edges
                    })
        
        # If no cameras were found, fall back to fixed scene-corner lamps
        if len(spotlights) == 0:
            scene_origin = trainer.scene_origin.cpu().numpy() if torch.is_tensor(trainer.scene_origin) else np.array(trainer.scene_origin)
            for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                spotlights.append({
                    "position": [scene_origin[0] + ox * 10.0, scene_origin[1] + oy * 10.0, scene_origin[2] + 10.0],
                    "intensity": 8.0,
                    "color": [1.0, 0.9, 0.6],
                    "direction": [0.0, 0.0, -1.0],
                    "cutoff_angle": 2.0 * np.pi / 3.0,
                })
        
        trainer.spotlights = spotlights
        print(f"Configured {len(spotlights)} spotlights for night mode.")

    # ------------------ Disable learned affine transform for night / debugging ------------------
    # Kimi's older afternoon/sunset path kept Affine enabled, so keep it by
    # default for daytime relights and only force identity for night or opt-in.
    if "Affine" in trainer.models and (args.mode == "night" or args.disable_affine):
        for layer in trainer.models['Affine'].decoder:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.zero_()
                layer.bias.data.zero_()
        print("Disabled Affine transform (set to identity).")

    # ------------------ Monkey-patch collect_gaussians for relighting ------------------
    if new_sun_dir is not None and trainer.pbr:
        print(f"Recomputing visibility for sun direction: {new_sun_dir.cpu().tolist()}")

        # Use M incident-ray samples at inference (paper uses 16 for train, 128 for inference)
        trainer.visibility_sample_num = args.visibility_samples
        print(f"Using visibility_sample_num = {trainer.visibility_sample_num}")

        # Clear old visibility caches so shadows are regenerated
        trainer._visibility_tracings_list.clear()
        trainer._incident_dirs_list.clear()
        trainer._incident_areas_list.clear()
        for class_name in trainer.gaussian_classes.keys():
            model = trainer.models[class_name]
            if hasattr(model, "_visibility_tracing"):
                model._visibility_tracing = None
            if hasattr(model, "_incident_dirs"):
                model._incident_dirs = None
            if hasattr(model, "_incident_areas"):
                model._incident_areas = None

        original_collect = trainer.collect_gaussians

        def relight_collect(self, cam, image_ids, sun_direction=None, update=False):
            # Force update=True and inject the new sun direction
            return original_collect(cam, image_ids, sun_direction=new_sun_dir, update=True)

        trainer.collect_gaussians = types.MethodType(relight_collect, trainer)

    # ------------------ Render ------------------
    if args.render_set == "test" and dataset.test_image_set is not None:
        target_dataset = dataset.test_image_set
    elif args.render_set == "train" and dataset.train_image_set is not None:
        target_dataset = dataset.train_image_set
    else:
        target_dataset = dataset.full_image_set

    print(f"[DEBUG] render_set={args.render_set}, target_dataset.split={target_dataset.split}, len={len(target_dataset)}")

    max_frames = args.max_frames
    vis_indices = None
    if max_frames is not None and max_frames < len(target_dataset):
        vis_indices = list(range(max_frames))
        print(f"Rendering {args.render_set} set ({max_frames}/{len(target_dataset)} images) ...")
    else:
        print(f"Rendering {args.render_set} set ({len(target_dataset)} images) ...")
    print(f"[DEBUG] max_frames={max_frames}, vis_indices={vis_indices}")

    render_results = render_images(
        trainer=trainer,
        dataset=target_dataset,
        compute_metrics=True,
        compute_error_map=False,
        vis_indices=vis_indices,
    )

    # ------------------ Save videos ------------------
    candidate_keys = [
        "rgbs",
        "gt_rgbs",
        "rendered_albedos",
        "rendered_roughness",
        "rendered_metallic",
        "rendered_normal",
    ]
    # Only keep keys that were actually produced by the model
    render_keys = [k for k in candidate_keys if k in render_results]

    num_cams = dataset.pixel_source.num_cams
    rendered_count = len(vis_indices) if vis_indices is not None else len(target_dataset)
    num_timestamps = rendered_count // num_cams if num_cams > 0 else rendered_count

    video_path = os.path.join(args.output_dir, f"{args.mode}_{args.render_set}.mp4")

    print(f"Saving video to {video_path}")

    save_videos(
        render_results,
        video_path,
        layout=dataset.layout,
        num_timestamps=num_timestamps,
        keys=render_keys,
        num_cams=num_cams,
        save_seperate_video=True,
        fps=args.fps,
        verbose=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
