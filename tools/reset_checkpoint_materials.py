#!/usr/bin/env python3
"""Reset selected material logits in an InvRGB+L checkpoint."""

import argparse
import os

import torch


def logit_like(tensor: torch.Tensor, value: float) -> torch.Tensor:
    filled = torch.full_like(tensor, float(value))
    return torch.logit(filled.clamp(1e-4, 1.0 - 1e-4))


def reset_materials(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if "models" not in ckpt:
        raise KeyError(f"{args.input} does not look like a trainer checkpoint: missing 'models'")

    roughness_count = 0
    reflectivity_count = 0
    for class_name, state in ckpt["models"].items():
        if not isinstance(state, dict):
            continue

        if "_roughness" in state:
            state["_roughness"] = logit_like(state["_roughness"], args.roughness)
            roughness_count += 1
            print(f"Reset {class_name}._roughness to {args.roughness}")

        if args.reflectivity is not None and "_reflectivity" in state:
            state["_reflectivity"] = logit_like(state["_reflectivity"], args.reflectivity)
            reflectivity_count += 1
            print(f"Reset {class_name}._reflectivity to {args.reflectivity}")

    if roughness_count == 0:
        raise RuntimeError("No _roughness tensors were found in the checkpoint")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(ckpt, args.output)
    print(
        f"Saved {args.output} "
        f"(roughness tensors: {roughness_count}, reflectivity tensors: {reflectivity_count})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reset roughness/reflectivity logits in a checkpoint")
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--roughness", type=float, default=0.7)
    parser.add_argument(
        "--reflectivity",
        type=float,
        default=None,
        help="Optional reflectivity reset value. Omit to preserve checkpoint reflectivity.",
    )
    reset_materials(parser.parse_args())
