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

# ------------------ Nighttime relighting ------------------
python -u tools/relight.py \
  --config configs/invrgbl.yaml \
  --checkpoint work_dirs/invrgbl/waymo023_dyn_1cam/checkpoint_final.pth \
  --output_dir work_dirs/invrgbl/waymo023_dyn_1cam/night \
  --mode night \
  --render_set full \
  --fps 10 \
  data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
  data.scene_idx=23 \
  data.start_timestep=0 \
  data.end_timestep=50

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
