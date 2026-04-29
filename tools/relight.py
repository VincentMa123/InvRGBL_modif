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
import os
import sys
import types
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.video_utils import render_images, save_videos


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
        choices=["relight", "night"],
        default="relight",
        help="Preset: 'relight' uses custom sun, 'night' disables sun and dims sky",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name in config (e.g. waymo/1cams)",
    )
    parser.add_argument(
        "--render_set",
        choices=["test", "full"],
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
    if args.opts is not None:
        opts_cfg = OmegaConf.from_cli(args.opts)
        cfg = OmegaConf.merge(cfg, opts_cfg)

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
    print(f"Resumed at step {trainer.step}")

    # ------------------ Configure lighting ------------------
    sky_model = trainer.models.get("Sky")
    if sky_model is None:
        print("Warning: No Sky model found in trainer; skipping lighting overrides.")

    new_sun_dir = None

    if args.mode == "night" and sky_model is not None:
        # Night preset: sun below horizon, dim sky dome and dim environment light.
        # The sky dome is rendered via SH coefficients; the scene is lit via
        # sky_intensity.  We scale both down together so the sky and ground
        # stay visually consistent.
        new_sun_dir = torch.tensor([0.0, -1.0, 0.0], device="cuda", dtype=torch.float32)
        sky_model.sun_intensity.data[:] = 0.01
        # Dim the sky dome SH (was learned for daylight)
        sky_model.shs.data[:] *= 0.15
        # Environment light for PBR — higher than physically strict so the
        # scene remains visible after gamma correction.
        sky_model.sky_intensity.data[:] = torch.tensor(
            [0.15, 0.15, 0.20], device=sky_model.sky_intensity.device, dtype=torch.float32
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

    # ------------------ Monkey-patch collect_gaussians for relighting ------------------
    if new_sun_dir is not None and trainer.pbr:
        print(f"Recomputing visibility for sun direction: {new_sun_dir.cpu().tolist()}")

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
    target_dataset = (
        dataset.test_image_set
        if args.render_set == "test" and dataset.test_image_set is not None
        else dataset.full_image_set
    )

    print(f"Rendering {args.render_set} set ({len(target_dataset)} images) ...")
    render_results = render_images(
        trainer=trainer,
        dataset=target_dataset,
        compute_metrics=True,
        compute_error_map=False,
    )

    # ------------------ Save videos ------------------
    candidate_keys = [
        "rendered_pbr",
        "rendered_albedos",
        "rendered_roughness",
        "rendered_normal",
        "rendered_intensity",
    ]
    # Only keep keys that were actually produced by the model
    render_keys = [k for k in candidate_keys if k in render_results]

    num_cams = dataset.pixel_source.num_cams
    num_timestamps = len(target_dataset) // num_cams if num_cams > 0 else len(target_dataset)

    video_path = os.path.join(args.output_dir, f"{args.mode}_{args.render_set}.mp4")
    # rendered_pbr is linear radiance; apply sRGB gamma so dark scenes are
    # visible on standard monitors (otherwise values <0.02 look pure black).
    if "rendered_pbr" in render_results:
        render_results["rendered_pbr"] = [
            torch.clamp(img, 0.0, 1.0) ** (1.0 / 2.2) for img in render_results["rendered_pbr"]
        ]

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
