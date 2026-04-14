import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision
from diffusers import DDIMScheduler
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Batch RGB-X material extraction")
    parser.add_argument("--rgbx_root", required=True, help="Path to cloned RGB-X repo")
    parser.add_argument("--input_dir", required=True, help="Directory containing input RGB images")
    parser.add_argument("--output_albedo_dir", required=True, help="Directory for albedo JPG outputs")
    parser.add_argument("--output_roughness_dir", required=True, help="Directory for roughness JPG outputs")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing outputs")
    return parser.parse_args()


def resize_for_model(photo: torch.Tensor):
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    ratio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / ratio)
    else:
        new_width = max_side
        new_height = int(new_width * ratio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

    resized = torchvision.transforms.Resize((new_height, new_width))(photo)
    return resized, old_height, old_width, new_height, new_width


def collect_images(input_dir: Path):
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(input_dir.glob(pattern))
    return sorted(set(image_paths))


def main():
    args = parse_args()

    rgbx_root = Path(args.rgbx_root).resolve()
    rgb2x_root = rgbx_root / "rgb2x"
    sys.path.insert(0, str(rgb2x_root))

    from load_image import load_exr_image, load_ldr_image
    from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

    input_dir = Path(args.input_dir)
    output_albedo_dir = Path(args.output_albedo_dir)
    output_roughness_dir = Path(args.output_roughness_dir)
    output_albedo_dir.mkdir(parents=True, exist_ok=True)
    output_roughness_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for RGB-X but torch.cuda.is_available() is false")

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=dtype,
        cache_dir=str(rgb2x_root / "model_cache"),
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "roughness": "Roughness",
    }

    for image_path in tqdm(image_paths, desc="RGB-X materials", leave=True):
        stem = image_path.stem
        albedo_path = output_albedo_dir / f"{stem}.jpg"
        roughness_path = output_roughness_dir / f"{stem}.jpg"
        if args.skip_existing and albedo_path.exists() and roughness_path.exists():
            continue

        suffix = image_path.suffix.lower()
        if suffix == ".exr":
            photo = load_exr_image(str(image_path), tonemapping=True, clamp=True).to(device)
        else:
            photo = load_ldr_image(str(image_path), from_srgb=True).to(device)

        photo, old_height, old_width, new_height, new_width = resize_for_model(photo)

        outputs = {}
        for aov_name in ("albedo", "roughness"):
            generated = pipe(
                prompt=prompts[aov_name],
                photo=photo,
                num_inference_steps=args.inference_steps,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]
            outputs[aov_name] = torchvision.transforms.Resize((old_height, old_width))(generated)

        outputs["albedo"].save(albedo_path, quality=95)
        outputs["roughness"].save(roughness_path, quality=95)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    main()