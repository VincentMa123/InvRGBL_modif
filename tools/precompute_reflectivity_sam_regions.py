import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger()


def load_sam_generator(args: argparse.Namespace, device: torch.device):
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint does not exist: {args.sam_checkpoint}")

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment_anything is required only for offline region precomputation. "
            "Install SAM and pass --sam_checkpoint, or disable "
            "data.pixel_source.load_region_maps during training."
        ) from exc

    if args.sam_model_type not in sam_model_registry:
        choices = ", ".join(sorted(sam_model_registry.keys()))
        raise ValueError(f"Unknown --sam_model_type '{args.sam_model_type}'. Available: {choices}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamAutomaticMaskGenerator(sam)


def reflectivity_to_rgb(reflectivity: torch.Tensor) -> np.ndarray:
    ref = reflectivity.detach().float().squeeze().cpu().numpy()
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(ref)
    if finite.any():
        vals = ref[finite]
        lo, hi = np.percentile(vals, [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = 0.0, 1.0

    if hi <= lo:
        norm = np.zeros_like(ref, dtype=np.float32)
    else:
        norm = np.clip((ref - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    gray = (norm * 255.0).round().astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def masks_to_label_map(masks, shape: Tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    next_label = 1
    for mask in sorted(masks, key=lambda item: int(item.get("area", 0))):
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        if segmentation.shape != shape:
            raise ValueError(f"SAM mask shape {segmentation.shape} does not match image shape {shape}")
        new_pixels = segmentation & (labels == 0)
        if not new_pixels.any():
            continue
        labels[new_pixels] = next_label
        next_label += 1
    return labels


def build_dataset_and_trainer(args: argparse.Namespace, device: torch.device):
    from omegaconf import OmegaConf

    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    log_dir = os.path.dirname(args.resume_from)
    cfg_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find experiment config next to checkpoint: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    cfg.data.pixel_source.load_region_maps = False
    cfg.trainer.render.eval_use_pbr_rgb = False

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
    if hasattr(trainer, "render_each_class"):
        trainer.render_each_class = False
    return cfg, dataset, trainer


def get_split_dataset(dataset, split: str):
    if split == "full":
        return dataset.full_image_set
    if split == "train":
        return dataset.train_image_set
    if split == "test":
        if dataset.test_image_set is None:
            raise ValueError("Requested --split test, but this experiment has no test image set")
        return dataset.test_image_set
    raise ValueError(f"Unsupported split: {split}")


def limit_split_dataset(split_dataset, max_frames: Optional[int], num_cams: int):
    if max_frames is None or max_frames <= 0:
        return split_dataset
    from datasets.base.split_wrapper import SplitWrapper

    max_images = int(max_frames) * int(num_cams)
    limited_indices = split_dataset.split_indices[:max_images]
    return SplitWrapper(
        datasource=split_dataset.datasource,
        split_indices=limited_indices,
        split=split_dataset.split,
    )


def global_index_to_frame_cam(dataset, img_idx: int) -> Tuple[int, int]:
    unique_cam_idx, frame_idx = dataset.pixel_source.parse_img_idx(img_idx)
    for cam_id in dataset.pixel_source.camera_list:
        if unique_cam_idx == dataset.pixel_source.camera_data[cam_id].unique_cam_idx:
            timestep = dataset.start_timestep + frame_idx
            return timestep, cam_id
    raise ValueError(f"Could not map image index {img_idx} to a camera id")


def move_tensors_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_device(item, device) for item in value)
    return value


@torch.no_grad()
def precompute_regions(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_generator = load_sam_generator(args, device)
    _cfg, dataset, trainer = build_dataset_and_trainer(args, device)

    split_dataset = get_split_dataset(dataset, args.split)
    split_dataset = limit_split_dataset(split_dataset, args.max_frames, dataset.pixel_source.num_cams)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(dataset.data_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Saving reflectivity SAM region labels to %s", output_dir)
    logger.info("Processing %d image(s) from %s split", len(split_dataset), args.split)

    if hasattr(trainer, "invalidate_visibility_frames"):
        frame_indices = [
            split_dataset.split_indices[i] // dataset.pixel_source.num_cams
            for i in range(len(split_dataset))
        ]
        trainer.invalidate_visibility_frames(frame_indices, full=False)

    for local_idx in tqdm(range(len(split_dataset)), desc="Precomputing SAM regions", dynamic_ncols=True):
        image_infos, cam_infos = split_dataset[local_idx]
        image_infos = move_tensors_to_device(image_infos, device)
        cam_infos = move_tensors_to_device(cam_infos, device)

        outputs = trainer(image_infos, cam_infos)
        if "rendered_reflectivity" not in outputs:
            raise RuntimeError("Trainer output does not contain rendered_reflectivity")

        reflectivity_rgb = reflectivity_to_rgb(outputs["rendered_reflectivity"])
        masks = sam_generator.generate(reflectivity_rgb)
        label_map = masks_to_label_map(masks, reflectivity_rgb.shape[:2])

        global_idx = split_dataset.split_indices[local_idx]
        timestep, cam_id = global_index_to_frame_cam(dataset, global_idx)
        save_path = os.path.join(output_dir, f"{timestep:03d}_{cam_id}.npy")
        np.save(save_path, label_map)

        if (local_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    logger.info("Finished SAM region precomputation")


def main(args: argparse.Namespace) -> None:
    log_dir = os.path.dirname(args.resume_from)
    try:
        from utils.logging import setup_logging
    except ImportError:
        logging.basicConfig(level=logging.INFO)
    else:
        setup_logging(output=log_dir, level=logging.INFO)
    precompute_regions(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Precompute SAM regions from rendered stage-1 reflectivity")
    parser.add_argument("--resume_from", type=str, required=True, help="stage-1 checkpoint path")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM checkpoint path")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM model type, e.g. vit_h")
    parser.add_argument("--split", choices=["full", "train", "test"], default="full")
    parser.add_argument("--max_frames", type=int, default=None, help="process only the first N timesteps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="region_sam_reflectivity",
        help="output directory, relative to the scene directory unless absolute",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="OmegaConf overrides")
    main(parser.parse_args())
