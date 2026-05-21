#!/bin/bash
#SBATCH -J relight_invrgbl
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=titanrtx02,titanrtx03,titanv01
#SBATCH -w rtx8000
#SBATCH -o work_dirs/%x.%j.out

cd ~/project2/InvRGBL_modif
mkdir -p logs

source ~/.bashrc
conda activate InvRGBL

# Fix broken CUDA_HOME on some nodes
unset CUDA_HOME

set -euo pipefail

CHECKPOINT=work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/checkpoint_final.pth
DATA_ROOT=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training

# Dynamic OBB/contact-shadow debug:
# - save_images writes per-frame PNGs for the individual mask channels
# - background contact receiver traces the OBB footprint on the static road map
# - contact apply threshold keeps that footprint from darkening the car surface
python -u tools/relight.py \
    --resume_from "${CHECKPOINT}" \
    --name sunset_box_contact_debug_5f \
    --preset sunset \
    --split full \
    --max_frames 50 \
    --save_images \
    --dynamic_box_sun_visibility \
    --dynamic_box_skip_inside_boxes \
    --dynamic_box_receiver_static_only \
    --dynamic_box_receiver_dynamic_opacity_threshold 0.5 \
    --dynamic_box_shadow_strength 0.7 \
    --dynamic_box_contact_shadow_strength 1.0 \
    --dynamic_box_contact_shadow_receiver background \
    --dynamic_box_contact_shadow_height 1.2 \
    --dynamic_box_contact_shadow_softness 0.7 \
    --dynamic_box_contact_shadow_dynamic_opacity_threshold 0.8 \
    --dynamic_box_contact_shadow_apply_dynamic_opacity_threshold 0.8 \
    --env_diffuse_scale 0.5 \
    --env_specular_scale 0.7 \
    --keys rgbs,rendered_pbr,rendered_albedos,rendered_sun_visibility,rendered_dynamic_box_sun_visibility,rendered_dynamic_box_contact_shadow,depths \
    data.data_root="${DATA_ROOT}" \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=50


# ---------- Night relight ----------
# Uncomment the next block to run night relighting instead of sunset.
# The night preset sets sun_intensity=0 and uses a dark env_constant.
# Keep the same veil-reduction flags for consistency.
#
# python tools/relight.py \
#     --name night_5f \
#     --preset night \
#     "${COMMON_RELIGHT_OPTS[@]}"
