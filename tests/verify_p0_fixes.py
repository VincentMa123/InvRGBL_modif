#!/usr/bin/env python3
"""Verify P0 paper-code alignment fixes."""

import sys
import re
from pathlib import Path

import yaml

# Resolve paths relative to project root (parent of tests/)
PROJECT_ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "tests" else Path(__file__).parent


def check_pbr_py():
    """Verify LiDAR rendering equation has specular term, cosθ, and /π."""
    pbr_code = (PROJECT_ROOT / "models/trainers/pbr.py").read_text()
    
    lidar_fn = re.search(
        r"def rendering_equation_lidar\(.*?\):(.*?)(?=\n### USING ###|\ndef )",
        pbr_code,
        re.DOTALL,
    )
    assert lidar_fn, "rendering_equation_lidar not found"
    fn_body = lidar_fn.group(1)
    
    checks = [
        ("base_color / np.pi", "/ np.pi" in fn_body),
        ("F0 = 0.04", "F0 = 0.04" in fn_body),
        ("specular numerator min(1, 2cos^2θ)", "torch.clamp(2 * cos2_theta, max=1.0)" in fn_body),
        ("cosθ factor", "* cos_theta" in fn_body),
        ("distance squared falloff", "(view_dists ** 2)" in fn_body),
        ("NO f_s = 0", "f_s = 0" not in fn_body),
    ]
    
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False
    return all_pass


def check_base_py():
    """Verify view_dists is raw Euclidean distance and compute_losses structure is correct."""
    base_code = (PROJECT_ROOT / "models/trainers/base.py").read_text()
    
    no_sigmoid = "torch.sigmoid(view_dists/10)" not in base_code
    print(f"  [{'PASS' if no_sigmoid else 'FAIL'}] view_dists uses raw Euclidean distance (no sigmoid)")
    
    stage2_idx = base_code.find("if self.step > self.freeze_step:")
    after_stage2 = base_code[stage2_idx:]
    intensity_in_stage2 = "intensity_loss" in after_stage2
    print(f"  [{'PASS' if intensity_in_stage2 else 'FAIL'}] intensity_loss is in Stage 2 block")
    
    has_pbr_pre = 'pbr_loss_name = "pbr" if stage2_active else "pbr_pre"' in base_code
    print(f"  [{'PASS' if has_pbr_pre else 'FAIL'}] image-space PBR has pre/post-freeze loss weights")

    no_ispbr_skip = 'if not use_ispbr_train and "rendered_pbr" in outputs' not in base_code
    print(f"  [{'PASS' if no_ispbr_skip else 'FAIL'}] pbr_loss is not skipped during image-space PBR training")
    
    no_intensity_stage1 = True
    before_stage2 = base_code[:stage2_idx]
    if "loss_dict[\"intensity_loss\"]" in before_stage2 or "loss_dict['intensity_loss']" in before_stage2:
        no_intensity_stage1 = False
    print(f"  [{'PASS' if no_intensity_stage1 else 'FAIL'}] intensity_loss NOT in Stage 1")
    
    return no_sigmoid and intensity_in_stage2 and has_pbr_pre and no_ispbr_skip and no_intensity_stage1


def check_configs():
    """Verify freeze_step and loss weights match paper supplementary."""
    all_pass = True
    for cfg_path in [PROJECT_ROOT / "configs/invrgbl.yaml", PROJECT_ROOT / "configs/invrgbl_static.yaml"]:
        cfg = yaml.safe_load(cfg_path.read_text())
        trainer = cfg.get("trainer", {})
        
        freeze = trainer.get("freeze_step")
        ok_freeze = freeze == 15000
        print(f"  [{'PASS' if ok_freeze else 'FAIL'}] {cfg_path}: freeze_step = {freeze}")
        if not ok_freeze:
            all_pass = False
        
        losses = trainer.get("losses", {})
        checks = {
            "pbr_pre": 0.1,
            "pbr": 0.5,
            "albedo_pre": 0.1,
        }
        for key, expected in checks.items():
            val = losses.get(key)
            actual = val.get("w") if isinstance(val, dict) else val
            ok = actual == expected
            print(f"  [{'PASS' if ok else 'FAIL'}] {cfg_path.name}: {key} = {actual} (expected {expected})")
            if not ok:
                all_pass = False
        envmap_ao = trainer.get("render", {}).get("envmap_ao", {})
        ao_ok = (
            envmap_ao.get("enabled") is True
            and envmap_ao.get("radius") == 5
            and envmap_ao.get("strength") == 0.35
            and envmap_ao.get("specular_strength") == 0.2
            and envmap_ao.get("depth_bias") == 0.02
        )
        print(f"  [{'PASS' if ao_ok else 'FAIL'}] {cfg_path.name}: EnvMap AO defaults configured")
        if not ao_ok:
            all_pass = False
    return all_pass


def test_lidar_rendering():
    """Numerical sanity check on the LiDAR rendering equation."""
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f"  [SKIP] Missing dependency: {e}. Run in project conda env for numerical test.")
        return True
    
    from models.trainers.pbr import rendering_equation_lidar
    
    N = 10
    base_color = torch.full((N, 1), 0.5)
    roughness = torch.full((N, 1), 0.1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(N, 3)
    viewdirs = torch.tensor([[0.0, 0.0, 1.0]]).expand(N, 3)
    view_dists = torch.full((N, 1), 10.0)
    
    out = rendering_equation_lidar(base_color, roughness, normals, viewdirs, view_dists)
    pure_diffuse = (0.5 / np.pi) * 1.0 / 100.0
    ok = out.mean().item() > pure_diffuse * 2.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Frontal output > pure diffuse ({out.mean().item():.6f} > {pure_diffuse:.6f})")
    
    viewdirs_graze = torch.tensor([[1.0, 0.0, 0.0]]).expand(N, 3)
    out_graze = rendering_equation_lidar(base_color, roughness, normals, viewdirs_graze, view_dists)
    ok2 = out_graze.mean().item() < out.mean().item()
    print(f"  [{'PASS' if ok2 else 'FAIL'}] Grazing < frontal ({out_graze.mean().item():.6f} < {out.mean().item():.6f})")
    
    roughness_high = torch.full((N, 1), 0.99)
    out_rough = rendering_equation_lidar(base_color, roughness_high, normals, viewdirs, view_dists)
    ok3 = abs(out_rough.mean().item() - pure_diffuse) < pure_diffuse * 0.5
    print(f"  [{'PASS' if ok3 else 'FAIL'}] High roughness ≈ pure diffuse ({out_rough.mean().item():.6f} ≈ {pure_diffuse:.6f})")
    
    return ok and ok2 and ok3


def check_losses_py():
    losses_code = (PROJECT_ROOT / "models/losses.py").read_text()
    
    has_abs = "torch.abs(dx)" in losses_code and "torch.abs(dy)" in losses_code
    print(f"  [{'PASS' if has_abs else 'FAIL'}] Uses absolute differences (|dx|) instead of dx**2")
    
    has_sigma = "sigma=1.0" in losses_code and "sigma ** 2" in losses_code
    print(f"  [{'PASS' if has_sigma else 'FAIL'}] Exposes configurable sigma parameter")
    
    return has_abs and has_sigma


def check_relightgs_py():
    relightgs_code = (PROJECT_ROOT / "models/gaussians/relightgs.py").read_text()
    
    has_init_intensity = "init_intensity: torch.Tensor = None" in relightgs_code
    print(f"  [{'PASS' if has_init_intensity else 'FAIL'}] create_from_pcd accepts init_intensity")
    
    has_roughness_map = "init_roughness = 1.0 - intensity_norm" in relightgs_code
    print(f"  [{'PASS' if has_roughness_map else 'FAIL'}] Maps intensity_norm to initial roughness")
    
    no_todo = "#TODO: use lidar intensity to initalize roughness" not in relightgs_code
    print(f"  [{'PASS' if no_todo else 'FAIL'}] TODO comment removed/resolved")
    
    return has_init_intensity and has_roughness_map and no_todo


def check_albedo_leak_controls():
    """Verify image-space PBR cannot drive albedo through RGB reconstruction."""
    base_code = (PROJECT_ROOT / "models/trainers/base.py").read_text()
    pbr_code = (PROJECT_ROOT / "models/trainers/pbr.py").read_text()

    assigns_pbr = "rendered_pbr = pbr_rgb" in base_code
    print(f"  [{'PASS' if assigns_pbr else 'FAIL'}] image-space PBR is stored in rendered_pbr")

    eval_only_rgb = "if not self.training:\n                        rendered_rgb = pbr_rgb" in base_code
    print(f"  [{'PASS' if eval_only_rgb else 'FAIL'}] rendered_rgb is replaced by PBR only outside training")

    detaches_materials = (
        "pbr_albedo = rendered_albedos.detach() if self.training else rendered_albedos" in base_code
        and "pbr_normal = rendered_normal.detach() if self.training else rendered_normal" in base_code
        and "pbr_reflectivity = rendered_reflectivity.detach() if self.training else rendered_reflectivity" in base_code
    )
    print(f"  [{'PASS' if detaches_materials else 'FAIL'}] PBR loss detaches material maps during training")

    base_color_trainable = (
        '"_base_color",' not in base_code.split("frozen_attrs = (", 1)[1].split(")", 1)[0]
        and '"base_color",' not in base_code.split("frozen_components = {", 1)[1].split("}", 1)[0]
        and 'for attr_name in ("_base_color", "_reflectivity"):' in base_code
    )
    print(f"  [{'PASS' if base_color_trainable else 'FAIL'}] _base_color remains trainable in Stage 2")

    ao_controls = (
        "def compute_screen_space_ao" in pbr_code
        and "env_diffuse = env_diffuse * ao" in pbr_code
        and "env_specular = env_specular * specular_ao.clamp" in pbr_code
        and "ao_map=ao_map" in base_code
    )
    print(f"  [{'PASS' if ao_controls else 'FAIL'}] EnvMap diffuse/specular receive screen-space AO")

    return assigns_pbr and eval_only_rgb and detaches_materials and base_color_trainable and ao_controls


def main():
    print("=" * 60)
    print("Verifying P0 paper-code alignment fixes")
    print("=" * 60)
    
    results = []
    
    print("\n1. models/trainers/pbr.py (LiDAR BRDF):")
    results.append(check_pbr_py())
    
    print("\n2. models/trainers/base.py (schedule & loss structure):")
    results.append(check_base_py())
    
    print("\n3. Config files (freeze_step & loss weights):")
    results.append(check_configs())
    
    print("\n4. Numerical test of rendering_equation_lidar:")
    results.append(test_lidar_rendering())
    
    print("\n5. models/losses.py (neighborhood_smoothness_loss):")
    results.append(check_losses_py())
    
    print("\n6. models/gaussians/relightgs.py (P2: roughness from LiDAR intensity init):")
    results.append(check_relightgs_py())

    print("\n7. image-space PBR albedo leak controls:")
    results.append(check_albedo_leak_controls())
    
    print("\n" + "=" * 60)
    if all(results):
        print("All P0 verification checks PASSED.")
        sys.exit(0)
    else:
        print("Some P0 verification checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
