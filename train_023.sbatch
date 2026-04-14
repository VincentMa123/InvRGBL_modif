#!/bin/bash
#SBATCH -J training
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH -w rtx8000
#SBATCH -o work_dirs/%x.%j.out

cd /lab/haoq_lab/cse12310504/InvRGBL
mkdir -p logs

source ~/.bashrc
conda activate InvRGBL

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -u tools/train.py \
  --config_file configs/invrgbl.yaml \
  --output_root ./work_dirs \
  --project invrgbl \
  --run_name waymo023_dyn_1cam \
  dataset=waymo/1cams \
  data.data_root=/lab/haoq_lab/cse12310504/InvRGBL/data/waymo/processed/training \
  data.scene_idx=23 \
  data.start_timestep=0 \
  data.end_timestep=50 \
  logging.vis_freq=0