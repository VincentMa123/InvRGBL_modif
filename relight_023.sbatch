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

COMMON_RELIGHT_OPTS=(
    --resume_from "${CHECKPOINT}"
    --split full
    --max_frames 5
    --keys rgbs,rendered_pbr,rendered_albedos,rendered_sun_visibility
    --env_diffuse_scale 0.02
    --env_specular_scale 0.1
    --env_diffuse_mode neutral
    --force_dielectric
    --min_roughness 0.25
    --sky_scale 0.04
    --exposure_scale 0.35
    --disable_affine
    data.data_root="${DATA_ROOT}"
    data.scene_idx=23
    data.start_timestep=0
    data.end_timestep=50
)

# ---------- Sunset relight ----------
python tools/relight.py \
    --name sunset_5f \
    --preset sunset \
    "${COMMON_RELIGHT_OPTS[@]}"

# ---------- Night relight ----------
# Uncomment the next block to run night relighting instead of sunset.
# The night preset sets sun_intensity=0 and uses a dark env_constant.
# Keep the same veil-reduction flags for consistency.
#
# python tools/relight.py \
#     --name night_5f \
#     --preset night \
#     "${COMMON_RELIGHT_OPTS[@]}"
