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

cd /lab/haoq_lab/cse12310504/InvRGBL
mkdir -p logs

source ~/.bashrc
conda activate InvRGBL

# Fix broken CUDA_HOME on some nodes
unset CUDA_HOME

set -euo pipefail

CHECKPOINT=work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/checkpoint_final.pth
DATA_ROOT=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training

  python -u tools/relight.py \
    --resume_from work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/checkpoint_final.pth \
    --name sunset_box_sunvis_5f \
    --preset sunset \
    --split full \
    --max_frames 50 \
    --dynamic_box_sun_visibility \
    --dynamic_box_receiver_static_only \
    --dynamic_box_receiver_dynamic_opacity_threshold 0.5 \
    --dynamic_box_shadow_strength 0.7 \
    --dynamic_box_contact_shadow_strength 0.65 \
    --dynamic_box_contact_shadow_height 0.75 \
    --dynamic_box_contact_shadow_softness 0.35 \
    --keys rgbs,rendered_pbr,rendered_sun_visibility,rendered_dynamic_box_sun_visibility,rendered_dynamic_box_contact_shadow \
    data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
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
