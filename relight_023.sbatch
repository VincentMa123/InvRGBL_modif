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

# python -u tools/relight.py \
#   --config configs/invrgbl.yaml \
#   --checkpoint work_dirs/invrgbl/waymo023_dyn_1cam/checkpoint_final.pth \
#   --output_dir work_dirs/invrgbl/waymo023_dyn_1cam/night \
#   --mode night \
#   --render_set full \
#   --fps 10 \
#   data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
#   data.scene_idx=23 \
#   data.start_timestep=0 \
#   data.end_timestep=50


  python tools/relight.py \
    --checkpoint work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/checkpoint_final.pth \
    --output_dir work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/relight_afternoon_no_rigid_simple_envmap \
    --mode sunset \
    --render_set full \
    --visibility_samples 128 \
    --disable_rigid_nodes \
    --simple_envmap \
    --max_frames 1 \
    data.scene_idx=23 \
    data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training


# python tools/relight.py \
#       --checkpoint work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/checkpoint_final.pth \
#       --output_dir work_dirs/invrgbl/waymo023_dyn_1cam_ispbr/relight_afternoon \
#       --mode afternoon \
#       --render_set train \
#       --max_frames 10 \
#       data.scene_idx=23 \
#       data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
    


# ------------------ Custom sun relighting (uncomment to use) ------------------
# python -u tools/relight.py \
#   --config configs/invrgbl.yaml \
#   --checkpoint work_dirs/invrgbl/waymo023_dyn_1cam/checkpoint_final.pth \
#   --output_dir work_dirs/invrgbl/waymo023_dyn_1cam/relight_afternoon \
#   --sun_direction -0.5 -0.8 0.3 \
#   --sun_intensity 2.5 \
#   --render_set full \
#   --fps 10 \
#   data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
#   data.scene_idx=23 \
#   data.start_timestep=0 \
#   data.end_timestep=50
